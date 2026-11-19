"""Whole-model bpw accounting. usage:
  budget.py CHECKPOINT DENSE_BITS target  -> routed --target-bpw that makes the WHOLE model exactly 1.5
  budget.py CHECKPOINT DENSE_BITS BANK    -> whole-model bpw of BANK + dense; exit 1 if > 1.5
DENSE_BITS=0: dense as stored. Else embed_tokens + FP8 attention weights at DENSE_BITS + 16/32 bpw (runtime.fake_quant)."""
import json, re, sys
from pathlib import Path
from low_bpw_quants.checkpoint import Checkpoint

src, bits, what = Checkpoint(Path(sys.argv[1])), int(sys.argv[2]), sys.argv[3]
routed = 0
params = nbytes = 0
for shard in set(src.index.values()):
    with open(src.path / shard, "rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    for name, info in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for d in info["shape"]:
            count *= d
        size = info["data_offsets"][1] - info["data_offsets"][0]
        if ".mlp.experts." in name:
            if name.startswith("model.language_model.") and not re.search(r"(scale|scale_2|amax)$", name):
                routed += count * (2 if info["dtype"] == "U8" else 1)
            continue
        if not name.startswith(("model.language_model.", "lm_head.")):
            continue
        is_scale = bool(re.search(r"(scale|scale_2|amax|denominator)$", name))
        p = 0 if is_scale else count * (2 if info["dtype"] == "U8" else 1)
        params += p
        requant = bits and (info["dtype"] == "F8_E4M3" and not is_scale or "embed_tokens" in name)
        fp8_scale = bits and is_scale and name.removesuffix("_scale") in header and header[name.removesuffix("_scale")]["dtype"] == "F8_E4M3"
        nbytes += p * (bits + 0.5) / 8 if requant else 0 if fp8_scale else size
total = params + routed
limit = 1.5 * total / 8
print(f"params total {total:,} (dense {params:,}, routed {routed:,}); dense bytes {nbytes:,.0f} at dense_bits={bits}")
if what == "target":
    print(f"routed target bpw {(limit - nbytes) * 8 / routed:.5f}  (whole-model limit {limit:,.0f} B)")
else:
    bank = Path(what).stat().st_size
    whole = (bank + nbytes) * 8 / total
    print(f"bank {bank:,} B  routed {bank * 8 / routed:.4f} bpw  WHOLE-MODEL {whole:.4f} bpw  -> {'OK' if whole <= 1.5 else 'OVER BUDGET'}")
    sys.exit(whole > 1.5)
