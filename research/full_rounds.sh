#!/usr/bin/env bash
# Resumable tuning rounds on the full-quality profile (after research/full_exp.sh): per round, measure layer-scale
# on the current best bank -> reallocate -> quantize (L14, 64 seqs) -> eval with router mask; keep only the best bank.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:$HOME/low-bpw-work/deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=${PY:-/home/blaze/.venvs/vllmServer/bin/python}; M=${MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}; W=${WORK:-$HOME/low-bpw-work}
T=calib/tokens-64x1024.pt; S=$W/scores-full14z; ROUNDS=${ROUNDS:-3}
log() { echo "[$(date +%F' '%T)] $*"; }
kl() { $P -c "import json,sys;print(json.load(open(sys.argv[1]))['kl'])" "$1"; }
target=$($P research/budget.py $M 6 target | sed -n 's/^routed target bpw \([0-9.]*\).*/\1/p')
prev_scale=${LAYER_SCALE:-calib/layer-scale-tcq12-r1.json}; prev_alloc=$W/alloc-full14z.npz; prev_bank=$W/full14z.gguf
for r in $(seq 2 $((ROUNDS + 1))); do
  LS=$W/layer-scale-full-r$r.json; A=$W/alloc-full14z-r$r.npz; B=$W/full14z-r$r.gguf; O=$W/eval-full14z-r$r-mask.json
  if [ ! -f $LS ]; then
    $P -m low_bpw_quants layer-scale $M $T $prev_bank $prev_alloc $S calib/gradients.npz $LS.tmp --previous $prev_scale
    mv $LS.tmp $LS
  fi
  [ -f $A.json ] || $P -m low_bpw_quants allocate $M $S calib/gradients.npz $A --target-bpw $target --layer-scale $LS
  if [ ! -f $O ]; then
    [ -f $B ] || $P -m low_bpw_quants quantize $M $T $A $B --hessian-seqs 64 --passes 1
    $P research/budget.py $M 6 $B | tail -1
    LOW_BPW_DENSE_BITS=6 LOW_BPW_MASK_DROPPED=1 $P -m low_bpw_quants evaluate $M calib/wikitext-test.parquet --bank $B \
        --teacher-cache $W/teacher-wikitext2-test.pt --output $O.tmp
    mv $O.tmp $O
  fi
  log "full round $r: $(tr -d '\n ' < $O)"
  prev_scale=$LS
  # the next round measures on the best bank so far; delete the others
  best=$(for f in $W/eval-full14z-mask.json $W/eval-full14z-r*-mask.json; do [ -f $f ] && echo "$(kl $f) $f"; done | sort -g | head -1 | awk '{print $2}')
  case $best in
    */eval-full14z-mask.json) prev_bank=$W/full14z.gguf; prev_alloc=$W/alloc-full14z.npz ;;
    *) n=${best##*-r}; n=${n%-mask.json}; prev_bank=$W/full14z-r$n.gguf; prev_alloc=$W/alloc-full14z-r$n.npz ;;
  esac
  for b in $W/full14z.gguf $W/full14z-r*.gguf; do [ "$b" = "$prev_bank" ] || [ ! -f $b ] || rm -f $b; done
  log "best so far: $best"
done
