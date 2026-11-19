# 08 — Current best recipe (whole-model ≤1.5 bpw), for the 600B run

Best on Qwen3.6-35B-A3B (superseded by logs/09: full14z, KL 0.4101; use the 10-format ladder and 48/16 + 64 seqs):
- `tcq14z.gguf` + router mask;
- whole model 1.4997 bpw; WikiText-2 KL 0.4256, top-1 76.8%, PPL 8.93 (teacher 6.31).

## Steps

1. **Dense:** attention (FP8 source) and embed_tokens → 6-bit RTN, FP16 scale per 32 (6.5 bpw). Costs KL 0.012.
   - `research/budget.py CKPT 6 target` gives the routed bpw that makes the whole model exactly 1.5.
2. **Score** every expert with the L12 R64 trellis ladder plus `zero`:
   - formats `rtcq{4x1,2x1,4x3,2x2,2x3,2x4,2x6}l12` + `zero`;
   - `--hessian-seqs 24 --replay-seqs 8 --prior 2 --expert-batch 16`;
   - ~100 s/layer here (256 experts, 3×512×2048).
   - Zero can be scored separately (4 min) and merged with `research/add_zero.py`, which also retargets the bytes to L14 R256.
3. **Allocate:** NatScale (`allocate --target-bpw T --layer-scale S`).
   - One measured layer-scale round is enough; further rounds are noise.
   - The allocator's zero-prior fix is required.
4. **Quantize** with L14 R256 formats (`--hessian-seqs 32 --passes 1`): ~55 s/layer, 37 min total here.
   - Do not use `--sequential`; it gave no gain.
5. **Inference:** mask fully dropped experts out of router top-k.

## Gains

| step | KL |
|---|---|
| rvq → trellis | about −20% expert error at the same rate |
| NatScale + layer-scale (r1) | 0.468 |
| L14 encode | 0.458 |
| zero format | 0.427 |
| router mask | **0.4256** |

## Did not help

- extra layer-scale rounds
- sequential quantize
- data-trained LUTs
- row-rate allocation
- L16 (40× slower kernel)

## Speed notes for 600B

The fixes:
- R=64 tiles + 4 warps at L12;
- block LDLQ (one triangular solve per 16-col group, bit-identical).

Score went 527 → ~100 s/layer. The final L14 quantize is 1 format/expert, so it is cheap relative to scoring.

Run each step under `systemd-run --user ... MemoryMax=…` so a host OOM never takes down the desktop.

## Run it on another machine (e.g. H200)

```sh
git clone https://github.com/blazingbhavneek/low-bpw-quants && cd low-bpw-quants   # this branch
pip install gguf accelerate pyarrow "transformers>=5.19"                          # torch + triton already there
export MODEL=/path/to/Qwen3.6-35B-A3B-NVFP4 WORK=$HOME/low-bpw-work PY=$(which python) EB=64
research/full_exp.sh     # teacher cache -> score (10 fmts + zero) -> allocate -> L14 bank -> eval with mask
research/full_rounds.sh  # optional: 3 layer-scale tuning rounds; keeps only the best bank
```

- Both scripts are resumable: rerun them after any interruption.
- Inputs ship in `calib/`: tokens, gradients, wikitext-test.parquet, layer-scale-tcq12-r1.json.
- Trellis LUTs are retrained on the first run and cached in `~/.cache/low_bpw_quants`. That takes a few minutes per format.
- Reference on an RTX 5060 Ti 16 GB:
  - full_exp: 3.6 h (score 236 s/layer);
  - result: whole 1.4998 bpw, KL 0.4101, top-1 77.6%, PPL 8.85.
- `EB` = experts per scoring batch. Raise it on big GPUs; 16 fits in 16 GB.
- For a different model, pass `LAYER_SCALE=` (or run one layer-scale round first). The bundled file was measured on Qwen3.6-35B-A3B.
