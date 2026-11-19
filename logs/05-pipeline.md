# 05 — Pipeline: best codec + NatScale (minimal tuning) -> overnight layer-scale rounds
Script research/overnight.sh (log ~/low-bpw-work/overnight.log), started 2026-11-19 ~12:00 behind the score.
- Score: 8-format tcq ladder (0.27..3.02 bpw), 507 s/layer -> ~5.6 h (done ~17:30).
- Budget: research/budget.py -> routed target 1.15164 bpw with embed+attention at 6.5 bpw (dense 1,861,741,264 B);
  each bank re-checked from file size: whole-model must be ≤1.5 or the chain stops.
- Evals: teacher cache; dense-6-only (expert-free dense cost floor); round 1 = NatScale + rvq layer-scale round 1.
- Rounds 2..4: layer-scale (measured per-layer KL on held-out calib, geometric mean with previous) -> allocate
  -> quantize -> eval; keep lowest-KL bank, delete the rest.
Fixes on the way: evaluate with no bank + LOW_BPW_DENSE_BITS now runs the student (was returning teacher);
bank stores/loads tcq LUTs (`tcq.{V}x{kV}l{L}`), round-trip verified with LUT cache cleared.

## Resumable (2026-11-19 12:2x)
overnight.sh rewritten: every step skips finished outputs (score layers checked by loading each npz,
allocation .json, bank written via .partial rename, eval/layer-scale via .tmp rename). After a shutdown:
`nohup research/overnight.sh >> ~/low-bpw-work/overnight.log 2>&1 &` — loses at most the step in flight.
