# 09 — Full-quality run (`research/full_exp.sh`)

Recipe from logs/08 with three changes:
- richer ladder: +4x5 (1.25), +4x7 (1.75), +2x5 (2.5); 10 trellis formats + zero;
- 2× calibration: score 48/16, quantize 64;
- layer-scale from tcq12-r1.

L14 R256 bank, router mask on.

Wall time: score 236 s/layer (2 h 37 min), zero score 5 min, bank 45 min, eval 3 min. About 3.6 h total.

| bank | whole bpw | KL | top-1 | PPL |
|---|---|---|---|---|
| tcq14z + mask (7 fmts, fast split) | 1.4997 | 0.4256 | 76.8% | 8.93 |
| **full14z + mask** | **1.4998** | **0.4101** | **77.6%** | **8.85** |

KL −3.6%, top-1 +0.8 pt, PPL −0.08. The intermediate rates and better calibration both count; they were not separated.

## Allocation (share of expert projections)

| bpw | share |
|---|---|
| 0 (dropped) | 15.6% |
| 0.27 | 3.8% |
| 0.52 | 8.0% |
| 0.77 | 9.9% |
| 1.02 | 11.2% |
| 1.25 | 9.9% |
| **1.5** | **13.4%** |
| 1.75 | 9.0% |
| 2.0 | 13.2% |
| 2.5 | 5.7% |
| 3.0 | 0.4% |

- Below 1.5: 58% (including dropped).
- Above 1.5: 28%, mostly 1.75–2.5.
- Cold experts go below 1.5 (many dropped), hot experts go to 2–2.5. Few need 3 bpw once 2.5 exists.

Bank: `~/low-bpw-work/full14z.gguf`, 4,636,055,168 B.
