"""Held-out joint expert-output error on cached real experts (calib/bench-L{01,12,26}.pt, 16 experts each).
usage: bench.py LAYERS fmt [fmt ...]   e.g. bench.py 01,12,26 rtcq2x3l14r256
prints bpw and mean-per-expert ||y_q-y||^2/||y||^2 per layer + average over layers."""
import sys, time, torch
from low_bpw_quants import codecs
from low_bpw_quants.runtime import expert_output

dev = "cuda"


def load(layer):
    d = torch.load(f"calib/bench-L{layer}.pt", weights_only=False)
    n = d["n"].to(dev)
    hx = codecs.shrink(d["hx"].to(dev) / n.clamp_min(1)[:, None, None], n, 2 * d["hx"].shape[-1])
    hd = codecs.shrink(d["hd"].to(dev) / n.clamp_min(1)[:, None, None], n, 2 * d["hd"].shape[-1])
    W = [w.to(dev) for w in d["weights"]]
    X = [torch.cat([r.to(dev) for r in d["replay"][e]]).float() for e in range(len(n))]
    return W, (hx, hx, hd), X


def error(fmt, W, Hs, X):
    out = [codecs.quantize(fmt, w, h) for w, h in zip(W, Hs)]
    bits = sum(out[p][1][0].numel() for p in range(3)) * 8 / sum(w[0].numel() for w in W)
    per = []
    for e, x in enumerate(X):
        y0 = expert_output(x, [w[e] for w in W])
        y = expert_output(x, [o[0][e] for o in out])
        per.append(((y - y0).square().sum() / y0.square().sum()).item())
    return bits, sum(per) / len(per)


if __name__ == "__main__":
    layers, names = sys.argv[1].split(","), sys.argv[2:]
    data = {l: load(l) for l in layers}
    for name in names:
        fmt = codecs.FORMATS[name] if name in codecs.FORMATS else eval(name, vars(codecs))  # e.g. "Rotated(Trellis(2,3,14,g=8))"
        t = time.time()
        res = [error(fmt, *data[l]) for l in layers]
        avg = sum(r[1] for r in res) / len(res)
        cols = "  ".join(f"L{l} {r[1]:.4f}" for l, r in zip(layers, res))
        print(f"{name:18s} {res[0][0]:.4f} bpw  avg {avg:.4f}  {cols}  ({time.time() - t:.0f}s)", flush=True)
