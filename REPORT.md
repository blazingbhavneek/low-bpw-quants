# 1.5 bpw routed-expert quantization: experiment report (2026-11-18)

Test model: `nvidia/Qwen3.6-35B-A3B-NVFP4` (40 layers, 256 routed experts per layer, hidden 2048,
expert FFN 512, 8 experts routed per token) on one 16 GB RTX 5060 Ti. Work target:
`thinkingmachines/Inkling-2-Small` (NVFP4 experts in layers 0-39, MXFP8 experts in layers 40-47) on one H200.

## 1. Headline

| Bank (all ≈1.5 routed bpw) | routed bpw | whole text model bpw | KL (nats/token) | top-1 agreement | PPL |
|---|---:|---:|---:|---:|---:|
| decoded NVFP4 teacher | 4.5 (source) | | 0 | 1 | 6.288 |
| v1 failed baseline (binary256 / scalar128) | 1.4999 | 2.018 | 1.0936 | 0.6080 | 16.51 |
| uniform rvq8x12 | 1.5158 | 2.033 | 0.4531 | 0.7386 | 9.00 |
| NatScale over the rvq family | 1.4999 | 2.018 | 0.3843 | 0.7766 | 8.48 |
| **NatScale + measured per-layer calibration (best so far)** | **1.4999** | **2.018** | **0.3591** | **0.7798** | **8.27** |

KL fell from 1.094 to 0.359, a 67% cut. Perplexity went from 16.5 to 8.27; the teacher's is 6.29. Top-1 agreement went from 60.8% to 78.0%. Every bank is exactly the same size: routed bpw is the GGUF file size × 8 / 32,212,254,720 routed weights, headers and padding included.

"Whole text model bpw" adds the 2,705,517,784 bytes of the dense text weights, which stay in source precision. That covers attention, router, shared experts, norms, BF16 embeddings and lm_head: 2,448,355,968 parameters in all. Vision and MTP heads are excluded. The figure is (bank + dense) × 8 / (routed + dense params). The dense part alone costs about 0.5 bpw of the whole-model figure.

What moved the numbers, in order of size:

1. **The codec.** Randomized-Hadamard incoherence plus 8-dim universal-Gaussian VQ with LDLQ, Hessian least-squares scale refit and Hessian shrinkage (`rvq`) replaced binary256/scalar128. It dominates SAGE, GSQ and IQ1 at every rate (§4).
2. **The allocation.** NatScale contrast calibration plus a joint gate/up/down error, solved exactly under the byte budget, is 15% lower KL than uniform at a lower size (§5).
3. **Measured per-layer calibration of the NatScale proxy.** It gives another 6.6% lower KL (§7).

## 2. Setup and evaluation protocol

- **Evaluation** (unchanged from the handoff, reused for every row; `src/low_bpw_quants/evaluate.py`):
  - WikiText-2 raw test split, 32 evenly spaced nonoverlapping 512-token windows; the last 256 predictions of each window are scored, 8,192 in total.
  - KL(teacher ‖ student) is computed over the full vocabulary in FP64.
  - The teacher is the decoded-weight NVFP4 model with FP32 expert math, and its final hidden states are cached.
  - Both sides use the same Python text model, dense weights and router. Routes are recomputed independently.
  - My re-run of the v1 bank gives KL 1.0936 / top-1 0.6080 / PPL 16.506 against teacher PPL 6.288. The handoff reported 1.0956 / 0.6073 / 16.51 against 6.307; the small differences come from the reimplemented runtime.
- **Calibration data** (`calib/tokens-64x1024.pt`): 4 sources × 64 sequences × 1024 tokens. The sources are ultrachat, magpie and writingprompts (NatScale's chat sources) plus C4. Each source is one calibration fold.
  - The Hessians use the first 48 sequences per source, and the last 16 are a held-out replay for scoring.
  - Final quantization uses all 64.
  - The tuning and layer-calibration validation uses the last 4 per source.
  - WikiText is never used for calibration, so every WikiText number is out of domain.
- **Hardware notes:**
  - The full-model forward/backward fits 16 GB only with BF16 expert decode, per-layer activation checkpointing and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - Quantize peaks at about 12.5 GB.
  - Transformers falls back to the reference PyTorch gated-delta-rule and causal-conv kernels (no `flash-linear-attention`), which slows the layer walks.

## 3. Pipeline as implemented

| Stage | What it does | Qwen cost (5060 Ti) |
|---|---|---|
| `tokens` | packs the 4 sources into 64 × 1024 rows each | seconds |
| `gradients` | full-model forward/backward over 16 sequences per source. `grad_tap` accumulates squared CE-gradient energy at every routed expert output and counts routed events per source. Output `calib/gradients.npz` | tens of minutes |
| `score` | one layer-by-layer walk; details below | **1.77 h** (6,380 s, about 160 s/layer, 7 formats) |
| `allocate` | NatScale objective, per-expert lower convex hull over the 343 triples, then a global greedy by distortion reduction per byte under an exact byte budget (2 MiB reserve), with boundary rounding | seconds (CPU) |
| `quantize` | walks again with Hessians on all 64 sequences per source, quantizes every projection with its assigned format (expert batches of 16), and streams the GGUF | **32 min** |
| `layer-scale` (new) | KL on held-out calibration sequences when only layer L comes from the bank, divided by the proxy's layer-L distortion at that allocation | 25 min |
| `evaluate` | WikiText KL / top-1 / PPL against the cached teacher | about 1 min |
| `tune` (negative result) | end-to-end KL distillation of FP16 row scales | 40 s/step |

**Score stage, in detail:**
- Per expert it builds `H = Σ r² x xᵀ / n` for the gate/up input and the down input, where r is the router weight.
- It shrinks each Hessian toward its diagonal with `prior` = 2 pseudo-tokens per input dimension: `H ← (1-α)H + α diag(H)`, with `α = prior·d/(n + prior·d)`.
- It keeps up to 256 held-out replay inputs per expert and fold.
- It quantizes gate, up and down with every candidate format.
- For every (gate, up, down) format triple it measures the held-out **joint** expert-output error `M[e, fold, cg, cu, cd]`, with the SiLU between the projections.
- It also records exposure (routed counts per fold), weight variance and a spectral tail index α per projection.

**GGUF bank, version 2** (`low_bpw_moe`):
- One int8 tensor `blk.{L}.routed_experts` per layer holds experts in their original router ID order, each as gate, up, down payloads back to back.
- Each payload starts with an 8-byte self-describing header (format code + parameters).
- Metadata: format table, `[L, E, 3]` assignments, per-format byte sizes `[3, K]`, source arch, hidden/FFN sizes and gate/up caps.
- VQ codebooks are stored once each as FP16 tensors `vq.{dim}x{bits}`.
- Version 1 banks (header-less binary256/scalar128) still load.

## 4. Codec study (layer 12, real experts, held-out joint expert-output relative error)

`calib/bench-L12.pt` holds the real layer-12 experts and routed inputs. Every projection of the expert is quantized with the format, and the table reports `‖y_q − y‖² / ‖y‖²` of the full expert output on held-out tokens. All rows use Hessian shrinkage prior 2 unless noted. Bits are honest: every scale, label, position and header is counted.

| format | bpw | joint error | notes |
|---|---:|---:|---|
| binary256 (v1 control) | 1.25 | 0.656 | hand-designed sign + 2 magnitudes + 2 outliers |
| sage0 | 1.125 | 0.690 | SAGE-PTQ, fixed-rate layout, no salient weights |
| sage2 | 1.23 | 0.575 | 2 salient weights per block |
| rsage2 | 1.23 | 0.555 | + Hadamard rotation |
| scalar128 (v1 control) | 2.125 | 0.270 | |
| gsq2 | 2.125 | 0.258 | GSQ grid, GPTQ init |
| gsq2odd | 2.125 | 0.2315 | zero-free grid s·{−3,−1,1,3} |
| rgsq2odd | 2.125 | 0.216 | + Hadamard rotation |
| iq1_s (ggml) | 1.5625 | 0.48-0.486 | Hessian diagonal as imatrix + GPTQ feedback between 256-blocks |
| iq1_m (ggml) | 1.75 | 0.413 | |
| iq2_xxs (ggml) | 2.0625 | 0.314 | no prior |
| rvq8x6 | 0.77 | 0.72 | |
| rvq8x8 | 1.02 | 0.566 | 0.552 with 2 coordinate-descent passes |
| rvq8x10 | 1.27 | 0.429 | |
| **rvq8x12** | **1.52** | **0.316** | p1 0.312, p2 0.311 |
| rvq8x14 | 1.77 | 0.231 | |
| rvq8x16 | 2.02 | 0.167 | |
| rvq4x8 | 2.02 | 0.190 | |
| rvq4x10 | 2.52 | 0.099 | |

Findings:
- **SAGE-PTQ** is the requested sub-1.5 path. At about 1.2 honest bpw it reaches 0.555-0.575, worse than rvq8x8 at 1.02 bpw (0.566 → 0.552). The paper's bpw leaves out the per-weight table label. Charged literally, that layout costs about 5 bpw; the fixed-rate layout here charges one label per 32 inputs.
- **GSQ (+RCO)** is the requested above-1.5 path. Its best variant, rgsq2odd, gives 0.216 at 2.125 bpw. rvq8x16 gives 0.167 at 2.02 bpw.
- **GSQ's Gumbel-softmax refinement and the VQ coordinate-descent passes** gain only 1-3% on held-out data; they overfit the Hessian sample. Passes are kept at 1 in quantize.
- **IQ1 (llama.cpp iq1_s / iq1_m)** sits well above rvq at equal rate: 0.48 at 1.56 bpw against rvq8x12's 0.316 at 1.52. The end-to-end uniform iq1_s control is queued (§8).
- **Hessian shrinkage toward the diagonal** is a large win for rarely routed experts, whose few routed tokens give rank-deficient Hessians. Priors 4 and 8 are slightly worse than 2.
- **Conclusion:** the rvq family (randomized Hadamard on the input dim, a universal Gaussian k-means codebook, per-row FP16 scale, LDLQ feedback, Hessian least-squares scale refit) dominates at every rate. SAGE and GSQ remain implemented as ablation formats (`sage*`, `split*`, `gsq2*`, `int*`), and so do the ggml IQ types (`iq1_s`, `iq1_m`, `iq2_xxs`, `tq1_0`, via `libggml-base`).

**rvq{d}x{b} details** (`codecs.VQ` inside `codecs.Rotated`):
- W' = W Rᵀ with R = H_n diag(signs)/√n, Sylvester Hadamard with seed 0, applied on the input dimension, and H' = R H Rᵀ. Decoding multiplies by R; at inference the activations can be rotated instead.
- The codebook is k-means on N(0, I_d) with 2^b entries (64 samples per entry, 30 iterations). It is cached in `~/.cache/low_bpw_quants` and copied into each bank.
- Each row is normalized by an FP16 scale, and each 8-column group snaps to the nearest codeword. The nearest-codeword search is an exact FP32 Triton kernel for codebooks of at least 1024 entries.
- LDLQ/GPTQ error feedback runs within 128-column blocks, with block feedback between them.
- Row scales are then refit by Hessian least squares, `s = ⟨W, U⟩_H / ⟨U, U⟩_H`.
- bpw = b/d + 16/cols. For gate/up, cols = 2048 adds 0.008 bpw; for down, cols = 512 adds 0.031 bpw.

## 5. End-to-end runs and ablations (WikiText-2, 8,192 predictions)

| run | routed bpw | KL | top-1 | PPL | what it tests |
|---|---:|---:|---:|---:|---|
| v1 baseline | 1.4999 | 1.0936 | 0.6080 | 16.51 | handoff's failed bank |
| uniform rvq8x12 | 1.5158 | 0.4531 | 0.7386 | 9.00 | codec alone, no allocation |
| NatScale, `--pool max` | 1.4999 | 0.4289 | 0.7715 | 8.83 | exposure = max over sources (domain-robust protection): **worse** |
| NatScale, `--drop rvq8x6` | 1.4999 | 0.4222 | 0.7670 | 8.81 | floor at 1.02 bpw (no 0.77 bpw option): **worse** |
| NatScale (mean pool) | 1.4999 | 0.3843 | 0.7766 | 8.48 | the paper's objective |
| NatScale + `--layer-scale` (round 1) | 1.4999 | **0.3591** | **0.7798** | **8.27** | proxy × measured/proxy per layer |

All rvq runs use the 7-format set rvq8x6/8/10/12/14/16 + rvq4x10 (0.77-2.52 bpw), a 1.5 bpw target and quantize `--passes 1`.

**NatScale allocation anatomy at 1.5 bpw:**
- **Format use** across 30,720 projections: rvq8x6 25.8%, rvq8x8 10.8%, rvq8x10 11.0%, rvq8x12 10.9%, rvq8x14 10.2%, rvq8x16 16.3%, rvq4x10 15.1%. The allocation is strongly bimodal, and the expert-mean bpw at the 0/10/50/90/100th percentiles is 0.75/0.75/1.42/2.5/2.5.
- **Per projection:** gate 1.49, up 1.42, down 1.54 bpw.
- **Bits follow calibration exposure:** the experts in the lowest 10% of exposure average 0.80 bpw (90% of them all-rvq8x6), and those in the top 10% average 2.42 bpw.
- **Exposure is domain specific:** 26% of experts have more than a 10× spread in routing frequency across the 4 sources.
  - Protecting each expert by its most-using source (`--pool max`) still hurt WikiText.
  - So did removing the 0.77 bpw option (`--drop rvq8x6`).
  - Pushing rarely routed experts very low and spending the bits elsewhere is the right trade, even out of domain.
- **The contrast strength τ** (Jeffreys target, default 1 nat, giving β = 0.173) barely matters. Format use is nearly unchanged for τ from 0.25 to 4, and the allocation changes in 14-43% of projections. The expert term `E` (exposure × error) dominates.

## 6. Negative result: end-to-end scale tuning (`tune`)

KL distillation of the FP16 row scales against the decoded-weight teacher's full next-token distribution on the 240 training calibration sequences:
- Codes, codebooks and dense weights stay fixed, so the bank's size and layout don't change.
- The student rotates activations instead of weights: y = s · (C[idx] (R x)).
- Training uses Adam on log s, per-layer checkpointing and chunked lm_head KL. It runs at about 40 s/step at 4 × 1024 tokens.

| variant | held-out calib KL before | after 20 steps |
|---|---:|---:|
| per-row scales (31 M params), lr 1e-3 | 0.0645 | 0.0673 (worse) |
| per expert-projection scale (30 k params), lr 1e-3 | 0.0645 | 0.0658 (worse) |

The Hessian least-squares refit already leaves the scales near the end-to-end optimum. Most experts see about 128 tokens per step, so the remaining gradient signal is noise and Adam's near-sign steps random-walk away from that optimum. I dropped this. The command stays as a documented ablation.

The tuning also turned up a domain gap: held-out calibration KL is about 6× lower than WikiText KL (0.0645 vs 0.384). The chat data is low-entropy for this model.

## 7. Per-layer sensitivity: the NatScale proxy gets the layer trend backwards

`layer-scale` measures the held-out calibration KL (16 sequences, the last 4 of each source) when **only layer L's** routed experts come from the bank and every other layer stays at the teacher. NatScale bank, 1.5 bpw:

| layer | 0 | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 24 | 28 | 32 | 36 | 38 | 39 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KL_L ×10³ | 10.6 | 9.6 | 7.9 | 7.7 | 9.4 | 7.6 | 7.4 | 6.4 | 5.1 | 4.8 | 3.6 | 3.0 | 2.6 | 2.3 | 2.1 | 1.8 | 1.6 | 2.4 |
| NatScale bpw | 0.99 | 1.31 | 1.33 | 1.28 | 1.38 | 1.42 | 1.42 | 1.44 | 1.44 | 1.47 | 1.46 | 1.49 | 1.47 | 1.57 | 1.69 | 1.70 | 1.56 | 1.64 |

- **Early layers dominate the KL** even though NatScale gave them the fewest bits.
- **NatScale's layer-L distortion proxy** `Σ_i w_i A_i E_i` is nearly flat across layers, which is what an equal-marginal greedy produces. Its Spearman correlation with the measured `KL_L` is **−0.91**.
- **The Fisher objective** `g_i f_i M_i` correlates **+0.71**. Its allocation leans the opposite way, giving early layers more bits.
- **Not additive:** the single-layer KLs sum to 0.182, about 3× the joint KL of 0.0645. Downstream layers partly repair upstream errors.

**Fix:** `layer_scale_L = measured KL_L / proxy_L`, normalized to mean 1, multiplies layer L's distortion in `allocate`. This is calibrated on held-out calibration data, never on WikiText.
- **Round-1 scale:** 3.27 at layer 0, about 1.6-2.2 for layers 1-10, 1.0 around layer 14 and 0.3-0.5 for layers 28-39 (`calib/layer-scale-natscale.json`).
- **Effect on bits:** early layers rise to 1.6-1.8 bpw and deep layers drop to 1.3-1.45. Layer 0 is still at 1.43, because its experts' structural prior and exposure are low.
- **Result:** KL 0.3843 → 0.3591, top-1 0.7766 → 0.7798, PPL 8.48 → 8.27, at the identical 1.4999 bpw.

## 8. Runs still in the overnight queue

They run one after another on the 16 GB GPU, and each bank takes about 40 min. The `keep_best` step keeps only the lowest-KL new bank on disk, because only about 9 GB is free.

| run | what it tests | status |
|---|---|---|
| `natscale-seq-1.50` | `quantize --sequential`: after each layer is quantized, its outputs are recomputed with the quantized experts, so later layers' Hessians and routes see the deployed activations | running (layer-calibrated allocation) |
| `fisher-1.50` | the Fisher objective `g f M`, no tempering, no prior | queued |
| layer-scale round 2 → `natscale-cal2-1.50` | re-measure per-layer KL on the best bank, geometric mean with round 1, re-allocate | queued |
| `score9` → `natscale9-1.50` | wider format range: adds rvq8x4 (0.535 bpw, error 0.76 on random data) and rvq4x12 (3.035 bpw, 0.052). The allocator already piles 26% onto the cheapest and 15% onto the dearest format, so both ends of the range bind | queued (2.5 h re-score) |
| `uniform-iq1_s` | the requested IQ1 base-quant control, ggml iq1_s at 1.5625 bpw | queued last |

Results will be appended to `results/*.json`. The rule for adopting them: take the lowest-KL configuration, and re-run `layer-scale` whenever the operating point changes.

## 9. How to run this on Inkling-2-Small (one H200)

Inkling-2-Small has about 587 B routed weights, 18× Qwen's. At 1.5 bpw the routed bank alone is about 110 GB. Order of work:

1. **Find the real budget first.** Run `inspect` on the checkpoint.
   - It reports `routed_weights`, `dense_params` and `dense_bytes`, and whether shards are missing.
   - The deployable routed bpw is set by memory, not by taste: `target_bpw ≈ (141 GB − dense − KV cache − runtime workspace) × 8 / routed_weights`.
   - KL falls steeply with bits (rvq8x12 → rvq8x14 cuts the expert error from 0.316 to 0.231). So pick the **largest target that fits**, then let the allocator spend it.
   - Report routed bpw and whole-model bpw separately, as above.
2. **Check per-layer memory before any walk.** Today's code holds, on the GPU, one layer's decoded experts in FP32 (`4·E·3·ffn·hidden` bytes) plus per-expert Hessians (`4·E·(hidden² + ffn²)` bytes).
   - For Qwen those are 3.2 GB and 4.6 GB.
   - Plug in Inkling's `E`, `hidden` and `ffn` from `inspect`. If a full layer doesn't fit next to the dense model, chunk the experts in `Stats` and `score_layer` (several passes per layer, or Hessians kept in BF16 or on the CPU), and decode the bank layer in BF16. This is the main code change needed for Inkling.
   - The MXFP8 expert layers (40-47) already decode through `checkpoint.decode_mxfp8`. The gate/up soft-caps (`mlp_gate_softcap`, `mlp_up_softcap`) are carried through `runtime.activate` and the bank metadata.
3. **Build calibration tokens** with the Inkling tokenizer and chat template (`tokens`). Keep 64 × 1024 per source; more sources are better for routing coverage on a model this size.
4. **Run `gradients`** (one forward/backward over 16 sequences per source). On an H200, BF16 expert decode and per-layer checkpointing should fit comfortably.
5. **Score a few layers first** to time it: `score ... --layers 0 20 40 47`.
   - Qwen scoring took 160 s/layer with 7 formats on the 5060 Ti. Scale that by Inkling's per-layer routed weights (about 15× Qwen's per layer) and the H200's speed.
   - If the full run is too long, cut the format set to the four or five points the allocator used most: for example rvq8x6, rvq8x8, rvq8x12, rvq8x16 and rvq4x10. Or cut `--replay-seqs` to 8 per source.
6. **`allocate --target-bpw <from step 1>`** (NatScale, mean pool), then `quantize --hessian-seqs 64 --passes 1`, then `evaluate` against a cached teacher.
7. **Layer calibration:** `layer-scale` on that bank, then re-allocate with `--layer-scale`, re-quantize and re-evaluate. A second round uses `--previous`.
   - Expect the MXFP8 layers 40-47 to measure as sensitive. The publisher kept them at higher precision, and the calibration will move bits there if so.
8. Then test `quantize --sequential` (pending its Qwen result) and the wider format set.

Don't spend H200 time on SAGE, GSQ or IQ1 for this model. Their codec-level gap to rvq is large and consistent (§4).

## 10. Next things to implement and check (prioritized)

1. **Inference kernels for the rvq bank** (llama.cpp). Today only the Python runtime reads `low_bpw_moe` banks.
   - A CUDA path needs three pieces:
     - a fast Walsh-Hadamard transform of each activation (in place, O(n log n)), or of the down-projection input;
     - a per-row FP16 scale;
     - a gather-dot over an `2^b × 8` FP16 codebook: 64 KB for b = 12, which fits in shared memory or L1.
   - Mixed formats per expert can be grouped by format at load time.
   - The `inkling2-rvq` llama.cpp branch already has the routed-expert plumbing (strided expert IDs, group dispatch, gains). Its RVQ2X8 kernels use LUTs over two 256-entry books. That does not scale to 4096-entry books (a 1 M-entry LUT per token for gate), so the gather-dot is the right kernel for rvq8x12.
2. **Kernel-friendly residual formats as allocator options.**
   - A two-stage residual VQ with two 256-entry books is exactly RVQ2X8 (2 bpw), and 8 + 4 bits gives 1.5 bpw. Both map straight onto the existing LUT kernels.
   - Score them with the same joint-error machinery; the allocator can then trade quality for kernel cost.
   - The universal-Gaussian codebook plus Hadamard rotation should transfer, and per-layer trained books may do better.
3. **Trellis-coded quantization (QTIP-style)** at 1-2 bpw. It typically cuts Gaussian MSE by another 10-20% over 8-dim k-means VQ at the same rate. This is the next codec-level gain.
4. **Intra-expert sequential quantization:** build the down-projection Hessian from the *quantized* gate/up outputs, because at deployment the down projection sees quantized `h`. This needs a second tap pass per layer; the `walk(..., redo=True)` hook added today is the mechanism.
5. **Iterate the layer calibration** (round 2 is queued) and calibrate per projection type: down vs gate/up. The proxy might also mis-weight projections.
6. **Finer format grid:** rvq8x7, 9, 11 and 13 give more hull points, at the cost of re-scoring.
7. **Broader calibration domains** (code, math, encyclopedic text) for routing coverage. Calibration KL is 6× below WikiText KL, and routing is domain specific.
8. **Quants left unchecked end to end:**
   - the uniform iq1_s control (queued);
   - the user's original SAGE (<1.5) + GSQ (>1.5) mixture under NatScale. It is dominated at codec level, so it is skipped unless needed as a paper comparison;
   - iq1_m and iq2_xxs mixtures;
   - AQLM-style learned additive codebooks;
   - expert merging or pruning for experts with near-zero exposure. Those already sit at 0.77 bpw, and rvq8x4 at 0.52 bpw is being scored.
9. **Fine-tuning that can work:** PV-tuning-style code updates (discrete) with larger data, or tuning only the most-routed experts' scales. Plain scale-only Adam failed (§6).

## 11. llama.cpp: the `inkling2-rvq` branch

`https://github.com/blazingbhavneek/llama.cpp/tree/inkling2-rvq`, commit `739cd6cb1`, a single commit on top of `8ca5a80` ("llama : prune (#30565)").
- That base is the merge base the work branch was developed on.
- The commit is the branch's own diff from that base: 81 files, +7,877 / −127 lines. The resulting source tree is byte-identical to the work checkout: tree `b84adc0f…`.
- `llama_diff.txt` was taken against a newer `origin/master` (9fba51d). Applying it would also revert about 82 upstream commits, and those reverts are the "junk": deleted parsers and converters, CI and doc rollbacks. None of that, no untracked leftover examples and no build output is in the branch.

Contents:
- **ggml:** type 150 `RVQ2X8`, and CPU/CUDA routed matmuls (`ggml-cuda/rvq.cu`) with strided expert IDs and zero output for IDs outside the group.
- **CUDA kernels:** banded flash attention (`fattn-banded.cu`), plus the argsort, ssm-conv and MMA changes used by the Inkling graph.
- **llama:** Inkling and Inkling 2 (`src/models/inkling*.cpp`), with pruned routers, variable expert widths, relative-bias table, short convolutions, decode graph reuse and KV-cache changes.
- **mtmd:** the Inkling 2 BF16 vision projector.
- **Converters, gguf-py mappings and chat templates.**
- **Tests:** RVQ backend ops, flash-attention bias and banded-hash tests, arch tests.

Rebasing onto the fork's current `master` (178 commits newer) conflicts in 14 files. Most are add/add conflicts, because upstream has since added its own Inkling 2 support (`conversion/inkling2.py`, `src/models/inkling2.cpp`, `common/parsers/inkling.cpp`). The rest are in `llama-arch`, `llama-kv-cache`, `llama-model`, `llama-vocab`, `tensor_mapping.py` and `fattn-mma-f16.cuh`. Port by keeping upstream's Inkling 2 and re-applying the RVQ type, kernels, banded attention and loader deltas on top.

## 12. Reproduce

```sh
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=nvidia/Qwen3.6-35B-A3B-NVFP4
python -m low_bpw_quants tokens    $M calib calib/tokens-64x1024.pt
python -m low_bpw_quants gradients $M calib/tokens-64x1024.pt calib/gradients.npz
python -m low_bpw_quants score     $M calib/tokens-64x1024.pt calib/scores \
    --formats rvq8x6 rvq8x8 rvq8x10 rvq8x12 rvq8x14 rvq8x16 rvq4x10 --hessian-seqs 48 --replay-seqs 16 --prior 2
python -m low_bpw_quants allocate  $M calib/scores calib/gradients.npz calib/alloc-natscale-1.50.npz --target-bpw 1.5
python -m low_bpw_quants quantize  $M calib/tokens-64x1024.pt calib/alloc-natscale-1.50.npz natscale.gguf --hessian-seqs 64 --passes 1
python -m low_bpw_quants evaluate  $M calib/wikitext-test.parquet --bank natscale.gguf --teacher-cache teacher.pt
# layer calibration (the best row)
python -m low_bpw_quants layer-scale $M calib/tokens-64x1024.pt natscale.gguf calib/alloc-natscale-1.50.npz \
    calib/scores calib/gradients.npz calib/layer-scale-natscale.json
python -m low_bpw_quants allocate  $M calib/scores calib/gradients.npz calib/alloc-natscale-cal-1.50.npz \
    --target-bpw 1.5 --layer-scale calib/layer-scale-natscale.json
```

The small reproducibility files are committed under `calib/`:
- `gradients.npz`;
- every `alloc-*.npz` and its `.json` report;
- `layer-scale-natscale.json`;
- `tokens-64x1024.pt`.

The parquet sources, per-layer `scores/` (111 MB), bench tensors and banks (6 GB each) are not committed.

Files:
- `src/low_bpw_quants/codecs.py`: all formats, with a self-check under `__main__`.
- `allocation.py`: NatScale terms, hull and greedy, layer proxy.
- `score.py`: walk statistics and joint scoring.
- `bank.py`: GGUF writer and reader, sequential quantize.
- `calibrate.py`: tokens, gradients, layer sensitivity.
- `tune.py`: scale distillation.
- `runtime.py`: text runtime and walk.
- `evaluate.py`.
- `checkpoint.py`: NVFP4 and MXFP8 readers and dense accounting.
- Tests: `PYTHONPATH=src python -m unittest tests.test_quantizer` covers the payload, bank, allocation, sequential path and the tuned-expert forward plus scale write-back.
