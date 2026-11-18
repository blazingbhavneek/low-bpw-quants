# Low-bit MoE expert quantization

Routed experts of an NVFP4 MoE checkpoint (test model `nvidia/Qwen3.6-35B-A3B-NVFP4`; work target
`thinkingmachines/Inkling-2-Small`, NVFP4 plus MXFP8 experts in layers 40-47) are packed into a
mixed-precision GGUF expert bank at about 1.5 routed bits per weight. Dense weights (attention,
router, shared experts, embeddings) stay in the source checkpoint's precision. Stock llama.cpp does
not have a loader or kernels for this bank; the Python runtime here evaluates it.

## Pipeline

```sh
export PYTHONPATH=src
M=nvidia/Qwen3.6-35B-A3B-NVFP4
python -m low_bpw_quants inspect   $M                                # routed / dense parameter accounting
python -m low_bpw_quants tokens    $M calib calib/tokens-64x1024.pt   # 4 sources x 64 x 1024 tokens
python -m low_bpw_quants gradients $M calib/tokens-64x1024.pt calib/gradients.npz
python -m low_bpw_quants score     $M calib/tokens-64x1024.pt calib/scores \
    --formats rvq8x6 rvq8x8 rvq8x10 rvq8x12 rvq8x14 rvq8x16 rvq4x10
python -m low_bpw_quants allocate  $M calib/scores calib/gradients.npz calib/alloc.npz --target-bpw 1.5
python -m low_bpw_quants quantize  $M calib/tokens-64x1024.pt calib/alloc.npz bank.gguf --hessian-seqs 64 --passes 1
python -m low_bpw_quants evaluate  $M calib/wikitext-test.parquet --bank bank.gguf --teacher-cache teacher.pt
python -m low_bpw_quants layer-scale $M calib/tokens-64x1024.pt bank.gguf calib/alloc.npz calib/scores calib/gradients.npz \
    calib/layer-scale.json                                          # then allocate --layer-scale, quantize, evaluate
```

1. **gradients**: one full-model forward/backward per calibration batch (BF16 expert decode,
   per-layer checkpointing; fits 16 GB). It records the squared CE gradient energy at each
   expert's output and how often each expert is routed per source. These give NatScale's
   sensitivity `g`.
2. **score**: one layer-by-layer walk. For each expert it builds the input Hessians on 48
   sequences per source and keeps 16 sequences per source as a held-out replay. Each candidate
   format quantizes gate, up and down. Every (gate, up, down) combination is then scored by the
   held-out *joint* output error of the whole expert, with the SiLU between the projections.
3. **allocate**: NatScale: `d_i(c) = w_i * A_i(c) * E_i(c)`, where:
   - `w_i = g_i^beta` is contrast-calibrated so the expert weighting has Jeffreys divergence
     `tau` = 1 nat from uniform;
   - `E_i` is the exposure-weighted held-out error;
   - `A_i` is the structural prior.

   A per-expert lower convex hull feeds a global greedy that fills an exact byte budget.
   `--objective fisher` is the ablation `g_i f_i M_i`.
4. **quantize**: requantizes every projection with its assigned format on all 64 sequences per
   source and streams the GGUF.
5. **layer-scale**: measures the held-out KL when only layer L comes from the bank, then rescales
   each layer's proxy distortion by measured / predicted. NatScale's per-layer proxy is
   anti-correlated with the measured sensitivity. `--previous` iterates.
6. **tune** (ablation, it did not help): end-to-end KL distillation of the FP16 row scales, with
   codes fixed.
7. **evaluate**: on WikiText-2 test, 32 nonoverlapping 512-token windows, it scores the last
   256 predictions of each (8,192 predictions). It reports KL to the decoded-weight teacher,
   top-1 agreement and perplexity.

## Formats

| name | what | bpw |
|---|---|---|
| `rvq{d}x{b}` | randomized-Hadamard incoherence on the input dim, `d`-dim VQ on a universal Gaussian k-means codebook of 2^b entries (stored once per bank), per-row FP16 scale, LDLQ error feedback, Hessian least-squares scale refit | b/d + 16/cols |
| `gsq2`, `gsq2odd` | GSQ's 2-bit grid, group 128, FP16 scale; `odd` = zero-free {-3,-1,1,3} | 2.125 |
| `sage{cap}` | SAGE-PTQ in a fixed-rate layout (sign plane, per-matrix magnitude table, table label per 32 inputs, `cap` salient weights per block) | 1.1-1.3 |
| `iq1_s`, `iq1_m`, `iq2_xxs`, `tq1_0` | llama.cpp ggml types (via `libggml-base`), Hessian diagonal as imatrix, GPTQ feedback between 256-blocks | 1.56 / 1.75 / 2.06 / 1.69 |
| `binary256`, `scalar128` | the failed v1 baseline formats, kept as controls | 1.25 / 2.125 |

Every payload starts with an 8-byte self-describing header. A bank has:
- one `int8` tensor per layer, `blk.{L}.routed_experts`, holding experts in their original ID
  order, each as gate, up, down back to back;
- metadata giving the format table, the per-(layer, expert, projection) assignment and per-format
  byte sizes, from which the offsets follow;
- one FP16 `vq.{d}x{b}` codebook tensor per VQ format.

Version 1 banks still load.

## Results (Qwen3.6-35B-A3B, routed experts only)

WikiText-2 test, 8,192 predictions, KL to the decoded NVFP4 teacher (PPL 6.288). Routed bpw is the
bank file size over the 32.2 B routed weights; whole text model bpw adds the 2.71 GB of dense weights.

| bank | routed bpw | whole bpw | KL | top-1 | PPL |
|---|---:|---:|---:|---:|---:|
| v1 baseline (binary256 / scalar128) | 1.4999 | 2.018 | 1.094 | 0.608 | 16.51 |
| uniform rvq8x12 | 1.5158 | 2.033 | 0.453 | 0.739 | 9.00 |
| NatScale over rvq8x6..rvq4x10 | 1.4999 | 2.018 | 0.384 | 0.777 | 8.48 |
| NatScale + measured layer scale | 1.4999 | 2.018 | **0.359** | **0.780** | **8.27** |

[REPORT.md](REPORT.md) has the codec study, every ablation (including negative results), the
per-layer diagnosis, the Inkling plan and the next steps.

## Tests

```sh
PYTHONPATH=src python -m unittest tests.test_quantizer       # payload, bank, allocation and tune round trips
PYTHONPATH=src python -m low_bpw_quants.codecs               # every format: size, decode == quantizer, error
PYTHONPATH=src python -m low_bpw_quants.allocation           # greedy vs brute force, contrast beta
```
