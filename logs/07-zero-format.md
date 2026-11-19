# 07 — 0-bpw "zero" format (drop cold experts)

Idea:
- At 0.27 bpw (tcq4x1) an expert keeps almost nothing: relative joint error ~0.9.
- So offer 0 bpw as well: the projection decodes to zeros and the expert output is zero.
- Those bytes are freed for hot experts.

## Implementation

- `codecs.Zero`: code 10, header only.
- `research/add_zero.py` merges a zero-only score run into the trellis scores:
  - any (gate, up, down) combo containing `zero` gets error mean‖y0‖²/hidden on the same replay vectors;
  - it also retargets names and bytes from l12 to l14r256 (exact bytes for the final codec).
- The zero-only score run takes 4 min (no Viterbi).
- **Allocator fix:** the NatScale prior used each projection's bpw, so (zero, 3 bpw, …) looked better than all-zero.
  - That produced 176 "partial" experts that paid for useless projections.
  - Now any combo containing zero uses bpw = 0 for all three projections, giving 0 partial experts.
- Chain: `research/zero_exp.sh` (resumable).

## Result (whole-model ≤1.5, dense 6-bit, layer-scale from tcq12-r1, L14 R256 bank)

| bank | whole bpw | KL | top-1 | PPL |
|---|---|---|---|---|
| tcq14-r2 (no zero) | 1.4904 | 0.4581 | 75.8% | 9.24 |
| **tcq14z (zero allowed)** | **1.4997** | **0.4267** | **76.8%** | **8.94** |
| old uniform rvq8x12 at 2.0 whole | 2.0 | 0.453 | 73.9% | 9.00 |

- KL −6.9%, the largest single lever after the trellis codec.
- About 0.6% of that comes from spending the 0.0096 bpw slack.
- Mix (share of projections):

  | bpw | share |
  |---|---|
  | 0 (dropped) | 16.8% (1719 whole experts of 10240) |
  | 0.27 | 4% |
  | 0.52 | 7% |
  | 0.77 | 9.5% |
  | 1.02 | 15% |
  | 1.52 | 23% |
  | 2.02 | 22% |
  | 3.02 | 3% |

## Follow-ups

| variant | whole bpw | KL | top-1 | PPL |
|---|---|---|---|---|
| tcq14z round 2 (layer-scale re-measured on the zero bank) | 1.4997 | 0.4308 | 76.8% | 9.01 |
| **tcq14z + router mask** (`LOW_BPW_MASK_DROPPED=1`) | **1.4997** | **0.4256** | **76.8%** | **8.93** |

- Round 2 is slightly worse, so the layer-scale correction is converged/noise. Keep round 1.
- Router mask: fully dropped experts get logit −inf before top-k, and the weights renormalize over the remaining experts.
  - The token goes to its next-best expert instead of to a zero output.
  - KL −0.25%, at zero cost, and it saves inference compute. Use it as the default for dropped experts.
