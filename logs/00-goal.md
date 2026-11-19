# 00 — Goal and budget (read first)
Goal: least-lossy Qwen3.6-35B-A3B at ≤1.5 bpw for the WHOLE model (all 34,660,610,688 params)
-> total file ≤ 6,498,864,504 B. Inside it: cold experts <1.5 bpw, hot experts >1.5 (up to 2–3 bpw),
allocated by NatScale (+ measured layer-scale). Metrics: WikiText-2 KL (lower), top-1 (higher), PPL.
Every result reports whole-model bpw first.

Budget arms:
- A: dense (attn/embed/lm_head/router, 2,705,517,784 B NVFP4/BF16) untouched -> routed 3,793,346,720 B = 0.942 routed bpw.
- B: dense requantized to ~4 bpw -> routed ≈1.3 bpw. Measure; dense runs on every token.

No bank so far meets this: all earlier banks (KL 0.359 here, 0.354 work PC) are 2.018 whole-model bpw.
Env: GPU RTX 5060 Ti 16 GB; python /home/blaze/.venvs/vllmServer/bin/python, PYTHONPATH=src:$HOME/low-bpw-work/deps;
big outputs in ~/low-bpw-work (/mnt/common ~5 GB free).

## User constraint (2026-11-19): attention etc ≥6 bits
Text dense today: embed_tokens 1017 MB BF16 | attention (linear_attn in_proj_qkv/z/out, self_attn q/k/v/o) 1279 MB FP8 |
lm_head + shared experts NVFP4 (source precision, untouched) | router/norms BF16 (tiny, untouched). Vision/MTP not counted.
- Arm C (chosen default): embed + attention -> 6-bit RTN, FP16 scale / 32 (6.5 bpw).
  dense ≈ 1,862 MB -> routed budget ≈ 4,637 MB = **1.152 routed bpw** at 1.5 whole-model.
  Implemented as `LOW_BPW_DENSE_BITS=6` (runtime.fake_quant at load; teacher cache must be built without it).
- First measure: dense-6 alone (no bank) KL vs teacher = the dense cost floor.
