# 02 — Bit-shift trellis (QTIP-style TCQ) vs 8-dim VQ, iid N(0,1)
Script: research/tcq_proto.py. State = last L bits, shift in k·V bits/step, emit LUT[state] (V dims),
LUT k-means-trained by alternating Viterbi/centroid (6 iters), measured on held-out seq (T=128 steps).

| bpw | VQ8 (current rvq) | TCQ L=12 | TCQ L=16 | R(D) bound | best Δ vs VQ |
|---|---|---|---|---|---|
| 0.50 (V4) | 0.6201 | 0.5178 | 0.5555* | 0.500 | −16.5% |
| 0.75 (V4) | 0.4540 | 0.3758 | 0.3837* | 0.354 | −17.2% |
| 1.00 (V2) | 0.3314 | 0.2656 | 0.2599 | 0.250 | −21.6% |
| 1.50 (V2) | 0.1726 | 0.1384 | 0.1299 | 0.125 | −24.7% |
| 2.00 (V2) | 0.0886 | 0.0721 | 0.0662 | 0.0625 | −25.3% |
*L=16 V=4 under-trained (64K×4 LUT, few samples) — fix with more data.

Takeaway: VQ8 sits 1.0–1.5 dB above R(D); trellis gets to 0.15–0.25 dB. ~20–25% less MSE at every rate,
far beyond trained books (2%). This is the main lever. Next: real-weight codec with Hessian (block LDLQ) + held-out joint error.
