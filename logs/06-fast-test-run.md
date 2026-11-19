# 06 — Fast test run (L12 trellis, ~2.5 h)

Why: the full L14 chain is about 13 h on this GPU (score 527 s/layer).
This model is a test bed for the 600B model, so I cut iterations to about 1/5 and kept every pipeline stage:
score → NatScale allocate → quantize → eval → measured layer-scale → reallocate → eval.

## Changes vs the full profile (`TAG=tcq12 HSEQ=24 RSEQ=8 QSEQ=32 ROUNDS=1`)

- **Trellis L14 → L12** (4096 states).
  - Measured Viterbi time for all 8 formats on a 4096×2048 matrix: 3.47 s at L14, 0.88 s at L12 (4×). L10 is no faster.
  - Cost: about +0.5% error at 0.5 bpw and +3.6% at 1.5 bpw (screen in 03).
- **Dropped 2x5 (2.52 bpw).** Ladder: 0.27 / 0.52 / 0.77 / 1.02 / 1.52 / 2.02 / 3.02.
- **Fewer sequences:**
  - score Hessian 48→24 and replay 16→8 per source;
  - quantize Hessian 64→32.
- **Rounds:** r1 (NatScale + old layer-scale) and r2 (measured layer-scale). The r1→r2 KL drop is the "tuning works" signal.
- **Unchanged:**
  - whole-model 1.5 bpw budget: dense 6-bit RTN at 6.5 bpw, routed target 1.15164;
  - `budget.py` gate;
  - same WikiText-2 teacher eval.
- **Outputs:** `~/low-bpw-work/{scores-tcq12,alloc-tcq12-r*,tcq12-r*.gguf,eval-tcq12-r*.json}`.
  - The 3 L14 layers already in `scores-tcq/` are kept for the full run.

## Score speed (per layer, 256 experts × 7–8 formats)

| setup | s/layer |
|---|---|
| L14, R=256 tiles, 8 fmts (full profile) | 527 |
| L12, R=256, 7 fmts, fewer seqs | 283 |
| + R=64 tiles, num_warps 4 at L12 | 136 |
| + block LDLQ (per-group triangular solve, bit-identical) | **112** (4.7× vs full) |

Note: score's `layer N done (Xs)` is cumulative since process start, not per layer.

Why R=256 was slow:
- Each LDLQ step launches only (16 experts × rows/R) tiles, about 32 programs.
- Each program walks a 2048-step serial chain, so the kernel is latency-bound: ~2 ms per call, with most SMs idle.
- R=64 gives 4× more programs, each with a 4× shorter chain.
- At L12, 4 warps beat 8–16, because there are fewer cross-warp reductions per step.
- R=64 adds +0.007 bpw of start-state overhead. That is exact-bytes accounted.

For the 600B model:
- use L12 R64, or try L14 R64 with 8–16 warps;
- block LDLQ removed the per-column launch storm (it also speeds up `quantize`).

Ops: the 13:14 restart plus side benchmarks caused a host OOM, which killed VS Code.
Jobs now run via `systemd-run --user ... MemoryMax=16G OOMScoreAdjust=1000`. Peak so far is 13 GB, page cache included.

## Results (WikiText-2, 8192 predictions; teacher PPL 6.309)

| bank | whole-model bpw | routed bpw | KL | top-1 | PPL |
|---|---|---|---|---|---|
| dense 6-bit only, experts original | — | — | 0.0123 | 95.3% | 6.320 |
| **r1**: NatScale + old layer-scale, L12 trellis | **1.4996** | 1.1512 | **0.468** | 74.9% | 9.25 |
| **r2**: measured layer-scale | **1.4996** | 1.1512 | **0.464** | 75.1% | 9.21 |

For reference, earlier rvq banks at 2.0 whole-model (1.5 routed):
- uniform rvq8x12: KL 0.453
- best NatScale: KL 0.359

So r1 nearly matches uniform rvq while using 0.5 bpw less over the whole model.

### r1 allocation (share of expert projections)

| bpw | share |
|---|---|
| 0.27 | 22% |
| 0.52 | 8% |
| 0.77 | 10% |
| 1.02 | 15% |
| 1.52 | 22% |
| 2.02 | 20% |
| 3.02 | 2% (mostly down) |

- This is the predicted shape: a large cold tail far below 1.5, and hot experts at 2–3.
- Bank: 4,635,286,688 B routed. Built in ~10 min (block LDLQ). Eval takes ~2.5 min.

### r1 → r2

- KL −0.9% (0.468 → 0.464), top-1 +0.2 pt.
- The allocation barely moved:
  - 0.27-bpw share 22.5% → 23.1%;
  - other formats within ±0.7 pt.
- Layer-scale has converged. More rounds will not pay; the overnight loop of rounds is not worth its time.
- Measured per-layer KL is about 0.008–0.012 in early layers.
- Total chain wall time: 13:37 → 15:40 (score 62 min, then 1 h for evals and two rounds).

### Follow-up levers on the r2 allocation (same whole-model budget)

| variant | whole bpw | KL | top-1 | PPL | verdict |
|---|---|---|---|---|---|
| r2 (baseline) | 1.4996 | 0.4640 | 75.1% | 9.21 | |
| `--sequential` quantize (later layers see quantized earlier layers + 6-bit dense) | 1.4996 | 0.4641 | 75.5% | 9.29 | no gain, dropped |
| **same allocation re-encoded L14 R256** | **1.4904** | **0.4581** | **75.8%** | 9.24 | KL −1.3%, kept (best) |

- The L14 R256 bank is 40 MB smaller than L12 R64 (less start-state overhead). That leaves 0.0096 bpw of unused slack, about 41 MB. Reallocating at L14 sizes would spend it.
- Each lever gave ≤1.3%, so the codec and allocation tweaks are close to saturated at this budget.

### Dense bytes at dense_bits=6 (1.86 GB = 29% of the 6.5 GB budget)

| tensor | stored size | handling |
|---|---|---|
| embed_tokens | BF16, 1017 MB | 413 MB at 6.5 bpw |
| linear_attn in_proj_qkv / z / out_proj | FP8, 1007 MB | re-quantized to 6.5 bpw (≥6-bit rule) |
| self_attn q/o/k/v | FP8, 273 MB | re-quantized to 6.5 bpw (≥6-bit rule) |
| lm_head | NVFP4, 286 MB | kept as stored, ~4.5 bpw |
| shared experts | NVFP4, ~70 MB | kept as stored |
| router `mlp.gate` | BF16, 42 MB | kept as stored |

Remaining dense levers are small:
- router to 6-bit saves ~25 MB;
- embeddings below 6-bit (a lookup table, not attention) save up to ~130 MB, i.e. routed +0.03 bpw.
