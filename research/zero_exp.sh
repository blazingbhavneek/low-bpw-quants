#!/usr/bin/env bash
# Resumable: 0-bpw "zero" (dropped expert projection) as an extra allocation option, L14 R256 bank.
# Run as: systemd-run --user --unit=lowbpw-zero --collect -p MemoryMax=16G -p MemoryHigh=14G -p MemorySwapMax=1G \
#   -p OOMScoreAdjust=1000 /bin/bash -c "cd /mnt/common/Code/low-bpw-quants && exec research/zero_exp.sh >> ~/low-bpw-work/overnight.log 2>&1"
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:$HOME/low-bpw-work/deps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/blaze/.venvs/vllmServer/bin/python; M=nvidia/Qwen3.6-35B-A3B-NVFP4; W=$HOME/low-bpw-work
log() { echo "[$(date +%F' '%T)] $*"; }
Z=$W/scores-zero; S=$W/scores-tcq14z; A=$W/alloc-tcq14z.npz; B=$W/tcq14z.gguf; O=$W/eval-tcq14z.json
[ "$(ls $Z/layer-*.npz 2>/dev/null | wc -l)" = 40 ] || $P -m low_bpw_quants score $M calib/tokens-64x1024.pt $Z \
    --formats zero --hessian-seqs 24 --replay-seqs 8 --prior 2 --expert-batch 16   # same split as scores-tcq12
[ "$(ls $S/layer-*.npz 2>/dev/null | wc -l)" = 40 ] || $P research/add_zero.py $M $W/scores-tcq12 $Z $S l12 l14r256
target=$($P research/budget.py $M 6 target | sed -n 's/^routed target bpw \([0-9.]*\).*/\1/p')
[ -f $A.json ] || $P -m low_bpw_quants allocate $M $S calib/gradients.npz $A --target-bpw $target \
    --layer-scale $W/layer-scale-tcq12-r1.json
[ -f $B ] || $P -m low_bpw_quants quantize $M calib/tokens-64x1024.pt $A $B --hessian-seqs 32 --passes 1
$P research/budget.py $M 6 $B | tail -1
[ -f $O ] || { LOW_BPW_DENSE_BITS=6 $P -m low_bpw_quants evaluate $M calib/wikitext-test.parquet --bank $B \
    --teacher-cache $W/teacher-wikitext2-test.pt --output $O.tmp; mv $O.tmp $O; }
log "zero-format L14: $(tr -d '\n ' < $O)"

# round 2: measured layer-scale on the zero bank -> reallocate -> requantize -> eval
LS=$W/layer-scale-tcq14z.json; A2=$W/alloc-tcq14z-r2.npz; B2=$W/tcq14z-r2.gguf; O2=$W/eval-tcq14z-r2.json
[ -f $LS ] || { $P -m low_bpw_quants layer-scale $M calib/tokens-64x1024.pt $B $A $S calib/gradients.npz $LS.tmp \
    --previous $W/layer-scale-tcq12-r1.json; mv $LS.tmp $LS; }
[ -f $A2.json ] || $P -m low_bpw_quants allocate $M $S calib/gradients.npz $A2 --target-bpw $target --layer-scale $LS
[ -f $B2 ] || $P -m low_bpw_quants quantize $M calib/tokens-64x1024.pt $A2 $B2 --hessian-seqs 32 --passes 1
$P research/budget.py $M 6 $B2 | tail -1
[ -f $O2 ] || { LOW_BPW_DENSE_BITS=6 $P -m low_bpw_quants evaluate $M calib/wikitext-test.parquet --bank $B2 \
    --teacher-cache $W/teacher-wikitext2-test.pt --output $O2.tmp; mv $O2.tmp $O2; }
log "zero-format L14 round 2: $(tr -d '\n ' < $O2)"
