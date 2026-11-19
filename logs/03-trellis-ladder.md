# 04 — Trellis rate ladder vs rvq (layer 12 bench, mean-per-expert held-out joint error)
L=14 state bits, R=256-row tiles (start-state overhead 0.004 bpw). L=15/16 add only −0.3%/−1.5% and are
5–50x slower in the register-resident kernel (spills) -> L=14 chosen.

| bpw | rvq | rtcq (L14 R256) | Δ |
|---|---|---|---|
| 0.52 | rvq8x4 0.8719 | tcq4x2 0.7882 | −9.6% |
| 0.77 | rvq8x6 0.7206 | tcq4x3 0.6226 | −13.6% |
| 1.02 | rvq8x8 0.5657 | tcq2x2 0.4709 | −16.8% |
| 1.27 | rvq8x10 0.4287 | tcq4x5 0.3533 | −17.6% |
| 1.52 | rvq8x12 0.3157 | tcq2x3 0.2534 | −19.7% |
| 1.77 | rvq8x14 0.2308 | tcq4x7 0.1897 | −17.8% |
| 2.02 | rvq8x16 0.1669 | tcq2x4 0.1338 | −19.8% |
| 2.52 | rvq4x10 0.0988 | tcq2x5 0.0692 | −30.0% |
| 3.02 | rvq4x12 0.0514 | tcq2x6 0.0361 | −29.8% |

Every rate wins; biggest at the top (2.5–3 bpw) where the hot experts sit — supports the
"cold <1.5 / hot 2–3 bpw" mixture. Bank format: LUT stored as `tcq.{V}x{kV}l{L}` FP16 tensor.
Next: 40-layer score of the tcq ladder -> NatScale allocation at the WHOLE-MODEL 1.5 bpw budget (see 00-goal.md).
Output in ~/low-bpw-work/ (scores-tcq/, banks) — /mnt/common has only 5 GB.
