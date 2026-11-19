"""NatScale allocation over per-projection format triples under an exact byte budget.

d_i(c) = w_i * A_i(c) * E_i(c) for expert i and triple c = (gate, up, down) formats:
  w_i   contrast-calibrated gradient weight g_i^beta, beta solving J(q_beta, uniform) = tau nats
  E_i   exposure-weighted held-out joint error  sum_s pi_s f_si sqrt(M_si(c))
  A_i   structural prior  mean_p (alpha~ / alpha_ip) Var(W_ip) 2^(-2 b_p)
Each expert keeps the lower convex hull of (bytes, d); a global greedy takes hull steps by loss
reduction per byte, skipping steps that overflow the budget (boundary rounding).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


def contrast_beta(g: np.ndarray, tau: float = 1.0) -> float:
    """beta with Jeffreys divergence J(q, u) = tau, q ∝ g^(2 beta) over experts with g > 0."""
    x = np.log(g[g > 0])

    def jeffreys(t):
        q = np.exp(t * (x - x.max()))
        q /= q.sum()
        return t * (q @ x - x.mean())

    lo, hi = 0.0, 1.0
    while jeffreys(hi) < tau:
        lo, hi = hi, hi * 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if jeffreys(mid) < tau else (lo, mid)
    return (lo + hi) / 4  # t = 2 beta


def gradient_weights(gradients: Path, hidden: int, tau: float):
    with np.load(gradients) as data:
        energy, events = data["energy"], data["events"].sum(0)
    G = energy / (hidden * np.maximum(events, 1))
    g = G / G[events > 0].mean()
    beta = contrast_beta(g, tau)
    return np.where(g > 0, g, 0) ** beta * (g > 0), beta, g


def hull(cost: np.ndarray, dist: np.ndarray) -> list[int]:
    """Lower convex hull of (cost, dist): cost increasing, dist decreasing, gains per byte decreasing."""
    points = []
    for i in np.lexsort((dist, cost)):
        if points and dist[i] >= dist[points[-1]]:
            continue
        while len(points) >= 2:
            a, b = points[-2], points[-1]
            if (dist[a] - dist[b]) * (cost[i] - cost[b]) <= (dist[b] - dist[i]) * (cost[b] - cost[a]):
                points.pop()
            else:
                break
        points.append(int(i))
    return points


def greedy(costs: list[np.ndarray], dists: list[np.ndarray], budget: int) -> list[int]:
    """Pick one option per item under sum(cost) <= budget, maximizing total distortion reduction."""
    hulls = [hull(c, d) for c, d in zip(costs, dists)]
    pick = [h[0] for h in hulls]
    spent = sum(int(c[p]) for c, p in zip(costs, pick))
    if spent > budget:
        raise ValueError(f"cheapest assignment needs {spent} bytes > budget {budget}")
    steps = []  # (ratio, item, step)
    for i, (h, c, d) in enumerate(zip(hulls, costs, dists)):
        for k in range(1, len(h)):
            steps.append(((d[h[k - 1]] - d[h[k]]) / (c[h[k]] - c[h[k - 1]]), i, k))
    steps.sort(key=lambda s: (-s[0], s[1], s[2]))
    level, frozen = [0] * len(hulls), [False] * len(hulls)
    for _, i, k in steps:
        if frozen[i] or k != level[i] + 1:
            continue
        delta = int(costs[i][hulls[i][k]] - costs[i][hulls[i][k - 1]])
        if spent + delta > budget:
            frozen[i] = True
            continue
        spent += delta
        level[i] = k
        pick[i] = hulls[i][k]
    return pick


def expert_terms(score: dict, tied: bool, power: float = 0.5, pool: str = "mean", allowed=None):
    """Per expert: E_i(c) = sum_s pi_s f_si M_si(c)^power over flattened triples, and byte costs.
    pool "max" takes max_s instead: an expert is protected by the source that routes to it most,
    since routing frequencies are domain specific and the deployment mix is unknown."""
    M, replay_n = score["M"], score["replay_n"]
    E, S, K = M.shape[0], M.shape[1], M.shape[2]
    have = replay_n > 0
    filled = M.copy()
    known = have.any(1)
    layer_median = np.median(M[have], axis=0) if have.any() else np.zeros(M.shape[2:])
    for e in range(E):  # ponytail: folds or experts without replay borrow the expert / layer mean
        filled[e, ~have[e]] = M[e, have[e]].mean(0) if known[e] else layer_median
    f = score["exposure"] / np.maximum(score["tokens"], 1)  # [E, S]
    terms = f[:, :, None, None, None] * filled ** power
    err = terms.max(1) if pool == "max" else terms.sum(1) / S  # [E, K, K, K]
    allowed = range(K) if allowed is None else allowed
    combos = [(c, c, c) for c in allowed] if tied else list(itertools.product(allowed, repeat=3))
    idx = np.array(combos)
    cost = sum(score["bytes"][:, p, idx[:, p]] for p in range(3))  # [E, C]
    return err[:, idx[:, 0], idx[:, 1], idx[:, 2]], cost, idx


def objective_terms(source, scores: Path, gradients: Path, contrast: float, objective: str = "natscale",
                    prior: bool = True, tied: bool = False, pool: str = "mean", drop: tuple = ()):
    """Per layer (d [E, C], bytes [E, C]) of the allocation objective over format triples combos [C, 3].

    objective "natscale": w A E as above; "fisher": g_i f_i M_i, the second-order loss increase
    E[dy^T F dy] with an isotropic Fisher per expert output (no tempering, no structural prior)."""
    weights, beta, g = gradient_weights(gradients, source.hidden, contrast)
    if objective == "fisher":
        weights, prior = g, False
    layers = [dict(np.load(scores / f"layer-{l:03d}.npz")) for l in range(source.layers)]
    names = [str(n) for n in layers[0]["formats"]]
    bpw = layers[0]["bits"]
    ref = np.median(np.concatenate([s["alpha"] for s in layers]))
    allowed = [k for k, n in enumerate(names) if n not in drop]
    zero = [k for k, n in enumerate(names) if n == "zero"]
    terms, combos = [], None
    for l, score in enumerate(layers):
        err, cost, combos = expert_terms(score, tied, 1.0 if objective == "fisher" else 0.5, pool, allowed)
        d = weights[l][:, None] * err
        if prior:
            b = bpw[combos]
            b[np.isin(combos, zero).any(1)] = 0  # a zeroed projection zeroes the whole expert output
            A = ((ref / score["alpha"]) * score["variance"]) @ (2.0 ** (-2 * b)).T / 3  # [E, C]
            d = d * A
        terms.append((d, cost))
    return terms, combos, names, beta


def layer_proxy(choice: np.ndarray, terms, combos: np.ndarray) -> np.ndarray:
    """Per-layer objective sum of an existing allocation (choice [L, E, 3] in the scores' format order)."""
    index = {c: i for i, c in enumerate(map(tuple, combos.tolist()))}
    return np.array([d[np.arange(len(d)), [index[c] for c in map(tuple, ch.tolist())]].sum()
                     for (d, _), ch in zip(terms, choice)])


def allocate(source, scores: Path, gradients: Path, output: Path, target_bpw: float, contrast: float,
             uniform: str | None = None, prior: bool = True, tied: bool = False, reserve: int = 2 ** 21,
             objective: str = "natscale", pool: str = "mean", drop: tuple = (), layer_scale=None) -> dict:
    """Exact-budget allocation of the objective_terms; layer_scale multiplies layer l's distortion
    (measured per-layer sensitivity correction, see the layer-scale command)."""
    from . import codecs

    shapes = source.expected_shapes()
    routed = source.layers * source.experts * sum(r * c for r, c in shapes)
    if uniform:
        fmt = codecs.FORMATS[uniform]
        choice = np.zeros((source.layers, source.experts, 3), dtype=np.int32)
        result = {"formats": [uniform], "choice": choice}
        payload = source.layers * source.experts * sum(fmt.nbytes(s) for s in shapes)
        beta = None
    else:
        terms, combos, names, beta = objective_terms(source, scores, gradients, contrast, objective, prior, tied,
                                                     pool, drop)
        costs, dists = [], []
        for l, (d, cost) in enumerate(terms):
            costs += list(cost)
            dists += list(d if layer_scale is None else d * layer_scale[l])
        budget = int(target_bpw * routed / 8) - reserve
        pick = greedy(costs, dists, budget)
        choice = combos[np.array(pick)].reshape(source.layers, source.experts, 3).astype(np.int32)
        payload = int(sum(c[p] for c, p in zip(costs, pick)))
        result = {"formats": names, "choice": choice}
    np.savez(output, formats=np.array(result["formats"]), choice=result["choice"])
    use = np.bincount(result["choice"].reshape(-1), minlength=len(result["formats"])) / result["choice"].size
    report = {"output": str(output), "payload_bpw": payload * 8 / routed, "beta": beta, "objective": objective,
              "pool": pool, "drop": list(drop), "layer_scale": layer_scale is not None,
              "use": dict(zip(result["formats"], use.round(4).tolist())),
              "per_projection_use": {p: np.bincount(result["choice"][..., i].reshape(-1), minlength=len(result["formats"])).tolist()
                                     for i, p in enumerate(("gate", "up", "down"))}}
    Path(str(output) + ".json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    # Self-check: hull + greedy match brute force on a tiny problem; contrast beta hits tau.
    rng = np.random.default_rng(0)
    costs = [rng.integers(1, 10, 6) * 10 for _ in range(4)]
    dists = [rng.random(6) for _ in range(4)]
    low = sum(int(c.min()) for c in costs)
    for budget in (low, low + 60, low + 150):
        pick = greedy(costs, dists, budget)
        best = min((sum(d[p] for d, p in zip(dists, ps)), ps) for ps in itertools.product(range(6), repeat=4)
                   if sum(c[p] for c, p in zip(costs, ps)) <= budget)
        got = sum(d[p] for d, p in zip(dists, pick))
        assert sum(c[p] for c, p in zip(costs, pick)) <= budget
        print(f"budget {budget}: greedy {got:.3f} vs optimum {best[0]:.3f}")
        assert got <= best[0] * 1.5 + 1e-9
    g = rng.lognormal(0, 1, 1000)
    beta = contrast_beta(g, 1.0)
    x = np.log(g)
    q = np.exp(2 * beta * x)
    q /= q.sum()
    assert abs(2 * beta * (q @ x - x.mean()) - 1.0) < 1e-6
    print("beta", beta)
