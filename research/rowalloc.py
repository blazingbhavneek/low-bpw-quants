"""Within-matrix row-rate allocation test (rows quantize independently under input-side LDLQ).
Importance from replay folds 0-1, error measured on folds 2-3. Rates from the V=2 trellis ladder.
usage: rowalloc.py LAYERS BASE_kV"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "research")
import bench
from low_bpw_quants import codecs
from low_bpw_quants.runtime import expert_output

layers, base = sys.argv[1].split(","), int(sys.argv[2])
D = {0: 1.0, 1: 0.52, 2: 0.26, 3: 0.13, 4: 0.066, 5: 0.033, 6: 0.017}  # Gaussian MSE per kV (V=2)
fmts = {k: codecs.Rotated(codecs.Trellis(2, k, 14, R=64)) for k in range(1, 7)}


def allocate(I, base):
    """greedy integer kV per row in [0, 6] with mean == base, minimizing sum I_j D(k_j)."""
    k = torch.full_like(I, base, dtype=torch.long)
    for _ in range(4 * len(I)):  # pairwise moves: +1 on best gain row, -1 on cheapest-loss row
        up = torch.where(k < 6, I * (torch.tensor([D[int(v)] for v in k.tolist()], device=I.device) -
                                     torch.tensor([D[min(6, int(v) + 1)] for v in k.tolist()], device=I.device)), -1)
        down = torch.where(k > 0, I * (torch.tensor([D[max(0, int(v) - 1)] for v in k.tolist()], device=I.device) -
                                       torch.tensor([D[int(v)] for v in k.tolist()], device=I.device)), float("inf"))
        a, b = up.argmax(), down.argmin()
        if up[a] <= down[b] or a == b:
            break
        k[a] += 1; k[b] -= 1
    return k


def quant_rows(W, H, k):
    """quantize each rate class of rows separately (rows in multiples of 64 per class not required here)."""
    Q = torch.zeros_like(W)
    for v in k.unique().tolist():
        rows = (k == v).nonzero().flatten()
        if v == 0:
            continue
        sub = W[rows]
        pad = -len(rows) % 64  # pad rows to the tile height with zeros (cost ignored: test only)
        sub = F.pad(sub, (0, 0, 0, pad))
        q, _ = codecs.quantize(fmts[v], sub[None], H[None])
        Q[rows] = q[0, :len(rows)]
    return Q


for L in layers:
    W, Hs, _ = bench.load(L)
    d = torch.load(f"calib/bench-L{L}.pt", weights_only=False)
    base_err, alloc_err = [], []
    for e in range(len(d["n"])):
        xa = torch.cat([r.cuda() for r in d["replay"][e][:2]]).float()
        xb = torch.cat([r.cuda() for r in d["replay"][e][2:]]).float()
        g, u, dn = (w[e] for w in W)
        G, U = xa @ g.T, xa @ u.T
        s = torch.sigmoid(G); ds = s * (1 + G * (1 - s)); col = dn.square().sum(0)
        Ig = ((U * ds) ** 2).mean(0) * col * g.square().mean(1)
        Iu = (F.silu(G) ** 2).mean(0) * col * u.square().mean(1)
        Id = dn.square().mean(1)
        y0 = expert_output(xb, [g, u, dn])
        qs_base = [codecs.quantize(fmts[base], w[None], H[e][None])[0][0] for w, H in zip((g, u, dn), Hs)]
        qs_alloc = [quant_rows(w, H[e], allocate(I, base)) for w, H, I in zip((g, u, dn), Hs, (Ig, Iu, Id))]
        for qs, acc in ((qs_base, base_err), (qs_alloc, alloc_err)):
            acc.append(((expert_output(xb, qs) - y0).square().sum() / y0.square().sum()).item())
    b, a = sum(base_err) / len(base_err), sum(alloc_err) / len(alloc_err)
    print(f"L{L} kV={base}: uniform {b:.4f}  row-allocated {a:.4f}  ({100 * (a / b - 1):+.1f}%)", flush=True)
