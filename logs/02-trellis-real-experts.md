# 03 — Trellis codec on real experts (layer 12 bench, 16 experts, held-out joint error)
Codec `Trellis(V,kV,L,g=16,R=64)` in codecs.py (code 9), names `rtcq{V}x{kV}l{L}` (rotated).
Triton Viterbi kernel (state cost in registers, uint8 backpointers), LDLQ feedback per column
inside 16-col groups, each group split in 64-row tiles = one trellis sequence (T=512 steps @V=2).
Decode = sliding L-bit window over bitstream (random access, cheap kernel later). Roundtrip exact.
Bench script: research/bench.py (mean-per-expert matches REPORT §4: rvq8x12 0.3157 vs 0.316).

| bpw | rvq (VQ8) | rtcq L=12 | Δ |
|---|---|---|---|
| 0.5 | rvq8x4 0.8719 | rtcq2x1 0.7886 / rtcq4x2 0.7920 | −9.6% |
| 1.0 | rvq8x8 0.5657 | rtcq2x2 0.4780 | −15.5% |
| 1.5 | rvq8x12 0.3157 | rtcq2x3 0.2612 | −17.3% |
(mean-per-expert; tcq pays +0.016 bpw tile-start overhead; speed ≈ VQ, ~1 s / 16 experts / format)
