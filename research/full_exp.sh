#!/usr/bin/env bash
# Resumable full-quality run of the logs/08 recipe: richer ladder (10 trellis formats + zero), 2x calibration.
# systemd-run --user --unit=lowbpw-full --collect -p MemoryMax=16G -p MemoryHigh=14G -p MemorySwapMax=1G \
#   -p OOMScoreAdjust=1000 /bin/bash -c "cd /mnt/common/Code/low-bpw-quants && exec research/full_exp.sh >> ~/low-bpw-work/overnight.log 2>&1"
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:$HOME/low-bpw-work/deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=${PY:-/home/blaze/.venvs/vllmServer/bin/python}; M=${MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}; W=${WORK:-$HOME/low-bpw-work}; T=calib/tokens-64x1024.pt
# Other machine/model: MODEL=<checkpoint dir> WORK=<output dir> PY=<python> [EB=<expert batch, e.g. 64 on H200>] [LAYER_SCALE=<json>]
# Needs gguf + accelerate + pyarrow + transformers>=5.19 (qwen3_5_moe). Then run research/full_rounds.sh with the same env for tuning rounds.
log() { echo "[$(date +%F' '%T)] $*"; }
FMTS="rtcq4x1l12 rtcq2x1l12 rtcq4x3l12 rtcq2x2l12 rtcq4x5l12 rtcq2x3l12 rtcq4x7l12 rtcq2x4l12 rtcq2x5l12 rtcq2x6l12"
SPLIT="--hessian-seqs 48 --replay-seqs 16 --prior 2"
S0=$W/scores-full12; Z=$W/scores-fullzero; S=$W/scores-full14z
A=$W/alloc-full14z.npz; B=$W/full14z.gguf; O=$W/eval-full14z-mask.json
NL=$($P -c "from low_bpw_quants.checkpoint import Checkpoint;from pathlib import Path;print(Checkpoint(Path(\"$M\")).layers)")
mkdir -p $W
[ -f $W/teacher-wikitext2-test.pt ] || $P -m low_bpw_quants evaluate $M calib/wikitext-test.parquet --teacher-cache $W/teacher-wikitext2-test.pt --output $W/eval-teacher.json
count() { ls $1/layer-*.npz 2>/dev/null | wc -l; }
missing() { $P -c "import sys,numpy as np;from pathlib import Path
bad=[]
for l in range(int(sys.argv[2])):
    p=Path(sys.argv[1])/f'layer-{l:03d}.npz'
    try: np.load(p)['M']
    except Exception: p.unlink(missing_ok=True); bad.append(str(l))
print(' '.join(bad))" $1 $NL; }
m=$(missing $S0); [ -z "$m" ] || $P -m low_bpw_quants score $M $T $S0 --formats $FMTS $SPLIT --expert-batch ${EB:-16} --layers $m
log "full score complete: $(count $S0)/$NL"
m=$(missing $Z); [ -z "$m" ] || $P -m low_bpw_quants score $M $T $Z --formats zero $SPLIT --expert-batch 16 --layers $m
[ "$(count $S)" = $NL ] || $P research/add_zero.py $M $S0 $Z $S l12 l14r256
target=$($P research/budget.py $M 6 target | sed -n 's/^routed target bpw \([0-9.]*\).*/\1/p')
[ -f $A.json ] || $P -m low_bpw_quants allocate $M $S calib/gradients.npz $A --target-bpw $target \
    --layer-scale ${LAYER_SCALE:-calib/layer-scale-tcq12-r1.json}
[ -f $B ] || $P -m low_bpw_quants quantize $M $T $A $B --hessian-seqs 64 --passes 1
$P research/budget.py $M 6 $B | tail -1
[ -f $O ] || { LOW_BPW_DENSE_BITS=6 LOW_BPW_MASK_DROPPED=1 $P -m low_bpw_quants evaluate $M calib/wikitext-test.parquet \
    --bank $B --teacher-cache $W/teacher-wikitext2-test.pt --output $O.tmp; mv $O.tmp $O; }
log "full14z + mask: $(tr -d '\n ' < $O)"
