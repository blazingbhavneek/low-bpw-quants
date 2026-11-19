# 04 — Codec screening (bench L01/L12/L26, 48 experts, avg mean-per-expert held-out joint error)
Harness: research/bench.py LAYERS fmt... (formats by name or expression, e.g. "Rotated(Trellis(2,3,14,g=8))").

| variant (at ~1.5 bpw unless noted) | avg err | verdict |
|---|---|---|
| rvq8x12 (old codec) | 0.2817 | baseline |
| rtcq2x3 L14 g16 R256 | **0.2256** | −19.9% — chosen |
| V=4 instead of V=2 at same rate (1.0/1.5) | +0.5%/+2% | V=2 wins and is faster; V=4 only for in-between rates |
| L=12 / 13 | 0.2337 / 0.2290 | L14 better; L16 −1.5% but 40x slower kernel -> L14 final |
| LDLQ group g=8 / 32 / 64 | 0.2253 / 0.2260 / 0.2275 | flat — keep g=16 |
| data-trained LUT (layers 5/20/35) | 0.2252 (−0.2%) | rotated weights ≈ Gaussian; universal LUT kept |
| Hessian-weighted Viterbi | not run: col-weight CV 13% -> ≤1% possible | skipped |
| within-matrix row-rate allocation (oracle-ish, held-out split) | L01 −0.8%, L12 +4.0%, L26 +3.5% | rejected |

Kernel: LUT stored [V,S] (coalesced) -> 1.6x (V2) to 3x (V4) faster Viterbi. Exact match vs reference.

Final ladder (avg err): tcq4x1 0.27 bpw 0.908 | 2x1 0.52 0.738 | 4x3 0.77 0.570 | 2x2 1.02 0.424 | 4x5 1.27 0.315 |
2x3 1.52 0.226 | 4x7 1.77 0.168 | 2x4 2.02 0.118 | 2x5 2.52 0.061 | 2x6 3.02 0.032
vs rvq: 0.77 0.670 | 1.27 0.385 | 1.77 0.205 | 2.52 0.087 (−15..−30% everywhere).
Score set (8 fmts, drop 4x5/4x7 — slow V=4, NatScale mixes neighbours): 4x1 2x1 4x3 2x2 2x3 2x4 2x5 2x6.
