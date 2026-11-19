#!/usr/bin/env bash
# Resumable chain. Kill / reboot any time, then rerun (RAM-capped, survives VS Code):
#   systemd-run --user --unit=lowbpw-fast --collect -p MemoryMax=16G -p MemoryHigh=14G -p MemorySwapMax=1G -p OOMScoreAdjust=1000 --setenv=TAG=tcq12 --setenv=HSEQ=24 --setenv=RSEQ=8 --setenv=QSEQ=32 --setenv=ROUNDS=1 /bin/bash -c "cd /mnt/common/Code/low-bpw-quants && exec research/overnight.sh >> $HOME/low-bpw-work/overnight.log 2>&1"
# or plainly:
#   nohup research/overnight.sh >> ~/low-bpw-work/overnight.log 2>&1 &
# Every step skips work whose output already exists; at most the step in flight is redone
# (one score layer ~8.5 min, one bank quantize ~40-60 min, one eval ~20 min).
# 1) score missing layers (8 tcq formats) -> 2) teacher + dense-6-only evals ->
# 3) round 1 = NatScale + layer-scale round 1, minimal tuning -> 4) rounds 2..: measured layer-scale,
# reallocate, requantize, eval; the lowest-KL bank is kept, the others deleted.
# Each bank must be <= 1.5 WHOLE-MODEL bpw (research/budget.py) before it is evaluated.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:$HOME/low-bpw-work/deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/blaze/.venvs/vllmServer/bin/python; M=nvidia/Qwen3.6-35B-A3B-NVFP4; W=$HOME/low-bpw-work
# Fast test profile: TAG=tcq12 HSEQ=24 RSEQ=8 QSEQ=32 ROUNDS=1 (L12 formats, see logs/06)
TAG=${TAG:-tcq}; HSEQ=${HSEQ:-48}; RSEQ=${RSEQ:-16}; QSEQ=${QSEQ:-64}
T=$W/teacher-wikitext2-test.pt; WIKI=calib/wikitext-test.parquet; S=$W/scores-$TAG; BITS=6; ROUNDS=${ROUNDS:-3}
log() { echo "[$(date +%F' '%T)] $*"; }
kl() { $PY -c "import json,sys;print(json.load(open(sys.argv[1]))['kl'])" "$1"; }

# 1) score: wait for a score already running, then score whatever layers are missing or unreadable
while ps -eo args | grep -q "[l]ow_bpw_quants score"; do sleep 60; done
missing=$($PY - "$S" <<'EOF'
import sys, numpy as np
from pathlib import Path
bad = []
for l in range(40):
    p = Path(sys.argv[1]) / f"layer-{l:03d}.npz"
    try:
        np.load(p)["M"]
    except Exception:
        p.unlink(missing_ok=True); bad.append(str(l))
print(" ".join(bad))
EOF
)
if [ -n "$missing" ]; then
  log "scoring layers: $missing"
  $PY -m low_bpw_quants score $M calib/tokens-64x1024.pt $S --formats $(cat $W/formats-$TAG.txt) \
      --hessian-seqs $HSEQ --replay-seqs $RSEQ --prior 2 --expert-batch 16 --layers $missing
fi
log "score complete: $(ls $S/layer-*.npz | wc -l)/40"

target=$($PY research/budget.py $M $BITS target | sed -n 's/^routed target bpw \([0-9.]*\).*/\1/p')
log "dense_bits=$BITS -> routed target $target (whole-model 1.5)"

# 2) teacher + dense-only cost floor
[ -f $W/eval-teacher.json ] || { rm -f $T; $PY -m low_bpw_quants evaluate $M $WIKI --teacher-cache $T --output $W/eval-teacher.json; }
[ -f $W/eval-dense$BITS-only.json ] || LOW_BPW_DENSE_BITS=$BITS $PY -m low_bpw_quants evaluate $M $WIKI \
    --teacher-cache $T --output $W/eval-dense$BITS-only.json
log "dense-$BITS only: $(tr -d '\n ' < $W/eval-dense$BITS-only.json)"

# 3-4) rounds
scale=calib/layer-scale-natscale.json
for r in $(seq 1 $((ROUNDS + 1))); do
  alloc=$W/alloc-$TAG-r$r.npz; bank=$W/$TAG-r$r.gguf; out=$W/eval-$TAG-r$r.json; next=$W/layer-scale-$TAG-r$r.json
  if [ ! -f $out ]; then
    [ -f $alloc.json ] || $PY -m low_bpw_quants allocate $M $S calib/gradients.npz $alloc --target-bpw $target --layer-scale $scale
    [ -f $bank ] || $PY -m low_bpw_quants quantize $M calib/tokens-64x1024.pt $alloc $bank --hessian-seqs $QSEQ --passes 1
    $PY research/budget.py $M $BITS $bank | tail -1   # non-zero exit (stops the chain) if over 1.5 whole-model
    LOW_BPW_DENSE_BITS=$BITS $PY -m low_bpw_quants evaluate $M $WIKI --bank $bank --teacher-cache $T --output $out.tmp
    mv $out.tmp $out
    log "round $r (scale $scale): $(tr -d '\n ' < $out)"
  fi
  [ $r -gt $ROUNDS ] && break
  if [ ! -f $next ]; then
    $PY -m low_bpw_quants layer-scale $M calib/tokens-64x1024.pt $bank $alloc $S calib/gradients.npz $next.tmp --previous $scale
    mv $next.tmp $next
  fi
  scale=$next
  # keep only the lowest-KL bank (later rounds' layer-scale needs only its own bank)
  best=$(for f in $W/eval-$TAG-r*.json; do echo "$(kl $f) ${f%.json}"; done | sort -g | head -1 | awk '{print $2}' | sed "s/eval-$TAG/$TAG/").gguf
  for b in $W/$TAG-r*.gguf; do [ "$b" = "$best" ] || rm -f $b; done
done
best=$(for f in $W/eval-$TAG-r*.json; do echo "$(kl $f) $f"; done | sort -g | head -1)
log "BEST (kl, eval): $best"
