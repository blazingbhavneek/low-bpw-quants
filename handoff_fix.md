# Handoff for fixing the 1.5 bpw MoE quantizer

The user rejected the current Qwen result on perplexity, KL divergence, and top-token agreement. They asked for this handoff and are taking the fix to another agent. The existing GGUF is a reproducible failed quality baseline. This project is independent of the older Inkling RVQ work in `handoff_llama.md` and `llama_diff.txt`.

## User request

- Use the requested **>1.5 bpw GCO RSQ** quantization for the more important routed experts. The earlier model and papers in this thread identify the published methods as **GSQ + RCO**. The user's latest message uses the wording “GCO RSQ.” [ISTA-DASLab model card](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) · [GSQ](https://arxiv.org/abs/2604.18556) · [RCO](https://arxiv.org/abs/2605.00649)
- Use the **<1.5 bpw SAGE-PTQ** path for the other routed experts. SAGE-PTQ estimates the number of inlier groups from a sampled graph, uses adaptive saliency thresholds, and gives binary inliers and multibit salient weights separate quantization paths. A deployable budget must include the actual group lookup, positions, scales, and other metadata. [SAGE-PTQ](https://arxiv.org/abs/2606.05429)
- Choose which experts receive more bits using their importance and the joint distortion of their gate, up, and down projections. The user pointed to NatScale's contrast calibration for this. [NatScale](https://arxiv.org/abs/2610.20353)
- Test first on the local `nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint on a 16 GB RTX 5060 Ti. The work target is `thinkingmachines/Inkling-2-Small` on one H200.

## What I actually implemented

I substituted a fixed block binary codec and a basic four-level scalar codec for the requested paper quantizers. I implemented a NatScale-inspired allocation proxy. The resulting file did not meet the requested quality and does not implement the full method combination.

| Piece | Implementation used in the shipped file |
| --- | --- |
| Low precision | `binary256`, a hand-designed **1.25 bpw** format. Each 256-weight row block stores 32 sign bytes, two FP16 magnitudes, an 8-bit selector for eight contiguous 32-weight subgroups, two U8 outlier positions, and a U8 shared outlier magnitude relative to the larger binary scale. The two largest magnitudes are selected per block. Group membership is split around the block's inlier mean magnitude. It has no SAGE graph partition, adaptive salient fraction, or separately quantized salient values. |
| High precision | `scalar128`, a hand-designed **2.125 bpw** format. Each 128-weight block uses 2-bit codes on `[-3, -1, 1, 3]` and one FP16 scale. The shipped file uses four deterministic scale-fitting iterations from weight magnitudes. There is no GPTQ initialization or official GSQ encode path in this run. |
| Optional refinement | `refine_scalar` is an independent, GSQ-inspired Gumbel-Softmax routine over the same fixed grid, fitted against projection outputs on replay inputs. It is opt-in via `--gsq-steps`. **The shipped run used `--gsq-steps 0`.** The cloned GSQ code was never called by the encoder. |
| Expert selection | Each entire expert gets one codec across gate, up, and down. The allocator measures joint output MSE on the saved routed inputs, multiplies by a projection-variance and precision-decay prior, and computes the improvement from upgrading. Optional spectral-tail indices default to one and were absent in this run. It fits a global power transform for the measured sensitivities at 1 nat of contrast, then ranks positive weighted gains. Equal-size experts make this rank-based selection exact for this two-choice local proxy. The RCO clone was never called and the allocator does not optimize end-to-end model loss. |
| Calibration | A decoded-weight reference ran a teacher forward and backward on ten short locally authored passages. It stored up to 12 input vectors per expert and measured the mean squared next-token-loss gradient at routed expert outputs. Coverage was **8,375 of 10,240 experts (81.8%)**. Experts with no replay were skipped in scoring and stayed at 1.25 bpw. The fitted exponent was **0.317581**. |
| Output and inference | A custom `low_bpw_moe` **routed-expert GGUF bank**. Each `blk.N.routed_experts` tensor is an I8 byte container. Metadata assigns one codec per expert in original router ID order. The Python runtime reconstructs selected weights and combines the bank with source dense weights and routing. There are no corresponding stock llama.cpp type registrations, kernels, or full-model GGUF assembly. |

The actual file is [`artifacts/Qwen3.6-35B-A3B-routed-1.50bpw.gguf`](artifacts/Qwen3.6-35B-A3B-routed-1.50bpw.gguf). It is **6,039,595,840 bytes** for **32,212,254,720 routed weights**, or **1.49994985 routed bpw** including GGUF metadata and padding. It uses 2.125 bpw for **2,925 experts** and 1.25 bpw for **7,315 experts**, across all 40 layers. The reported budget excludes dense model weights, embeddings, vision weights, and speculative heads.

## Failed quality baseline

| Metric | Result |
| --- | ---: |
| Mean full-vocabulary KL, decoded NVFP4 reference → quantized | 1.095606 nats/token |
| Top-token agreement | 60.72998% (4,975 / 8,192) |
| Decoded NVFP4 reference perplexity | 6.30681 |
| Quantized perplexity | 16.51213 |

The benchmark ran the last 256 next-token predictions from each of 32 evenly spaced, nonoverlapping 512-token windows of the raw WikiText-2 test split. Both paths received the same token prefixes and used the same Python text model, dense weights, tokenizer, and routing code. They recomputed routes independently. KL used full-vocabulary FP64 log-softmax and reduction. The teacher is the decoded-weight NVFP4 reference with FP32 expert math. The native ModelOpt activation-quantized inference path was outside this comparison. The earlier 39-token smoke test was much smaller and should not replace this baseline. [WikiText-2 test split](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/wikitext-2-raw-v1/test-00000-of-00001.parquet)

## Code and reproduction

- [`src/low_bpw_quants/codecs.py`](src/low_bpw_quants/codecs.py) contains `pack_binary`, `initial_scalar`, `refine_scalar`, `encode`, and `decode`.
- [`src/low_bpw_quants/allocation.py`](src/low_bpw_quants/allocation.py) contains replay loading, `contrast_weights`, joint `score_candidates`, and `allocate`.
- [`src/low_bpw_quants/bank.py`](src/low_bpw_quants/bank.py) contains the scoring and packing pass, exact byte budgeting, GGUF layout, reader, and routed replay.
- [`src/low_bpw_quants/checkpoint.py`](src/low_bpw_quants/checkpoint.py) reads sharded safetensors. Qwen uses per-expert NVFP4 with ModelOpt block and outer scales. Inkling stores stacked experts and interleaves gate/up rows. Its official checkpoint uses NVFP4 experts in layers 0–39 and MXFP8 experts in layers 40–47. Both Inkling expert formats were checked against real weight slices and the dense mapping was checked against official shard headers. A full Inkling run on H200 remains unverified.
- [`src/low_bpw_quants/reference.py`](src/low_bpw_quants/reference.py) contains the text-only runtime and gradient capture. Dense parameters come from the source checkpoint. Inference can map the expert bank from disk or preload it.
- `research/GSQ` at `03fc16484c369e3127225615d5e03e8d3a6043e3` and `research/RCO` at `9a1e09c07d468109cbe60a1b87d5036034a79d10` are local reference clones. I found no verified public SAGE-PTQ or NatScale implementation at the time.
- Scratch repro files are `/tmp/low-bpw-qwen15.jzCRCp/calibration.txt` and its `replay/` directory, plus `/tmp/low-bpw-kld.kAllbx/evaluate.py` and `metrics.json`. The dataset is `/tmp/low-bpw-wikitext2-test.parquet` with SHA-256 `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`. These paths are temporary. The benchmark ran entirely on CUDA after the unrelated `llama-tts` job finished.

The original packing command ran from the repository root. Change the output path to reproduce it alongside the existing bank.

```sh
PYTHONPATH=src:/tmp/low-bpw-deps:/tmp/low-bpw-hf523 \
  /home/blaze/.venvs/vllmServer/bin/python -B -m low_bpw_quants quantize \
  nvidia/Qwen3.6-35B-A3B-NVFP4 \
  artifacts/Qwen3.6-35B-A3B-routed-1.50bpw.gguf \
  --target-bpw 1.50 --replay /tmp/low-bpw-qwen15.jzCRCp/replay --device cuda
```

## Fix priorities

1. Validate the requested SAGE and GSQ candidates on real routed inputs and count their actual serialized overhead before running a full checkpoint. Keep `binary256` and `scalar128` as failed-baseline controls.
2. Compare allocations at the same routed byte budget. Distinguish codec improvements from expert selection by holding the codecs or assignments fixed in separate comparisons. Get broader routed calibration coverage and explicitly account for experts absent from replay.
3. Reuse the same 8,192-prediction evaluation for direct comparison. The earlier small expert-MSE win over Q1_0 did not establish model quality. The current run gives no ablation identifying how much of the regression came from the binary codec, scalar codec, or allocation proxy.
4. Report routed bpw and whole-model bpw separately if producing a deployable full GGUF. Preserve original expert IDs and verify the chosen runtime on the H200 before claiming Inkling deployment compatibility.
