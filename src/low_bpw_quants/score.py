"""Layer walk statistics: per-expert Hessians, held-out replay, and NatScale candidate errors."""

from __future__ import annotations

import itertools
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from . import codecs
from .runtime import LayerCache, activate, down_input, expert_output, load_text_model, route_groups, walk


class Stats:
    """Walk tap. Hessian sequences build H = sum r^2 x x^T / n per expert (gate/up input and down
    input); replay sequences are kept per fold for held-out joint error; every token counts as exposure."""

    def __init__(self, source, load, entries: list[tuple[int, bool]], folds: int, replay_cap: int, device: str):
        self.source, self.load, self.entries, self.device = source, load, entries, device
        self.folds, self.replay_cap = folds, replay_cap
        self.entry = 0
        self.tokens = np.zeros(folds, dtype=np.int64)

    def begin(self, layer: int):
        E, Hd, F = self.source.experts, self.source.hidden, self.source.ffn
        self.layer = layer
        self.hx = torch.zeros(E, Hd, Hd, device=self.device)
        self.hd = torch.zeros(E, F, F, device=self.device)
        self.n = torch.zeros(E, device=self.device)
        self.exposure = np.zeros((E, self.folds), dtype=np.int64)
        self.tokens[:] = 0
        self.replay = [[[] for _ in range(self.folds)] for _ in range(E)]

    def __call__(self, layer, x, ids, route):
        fold, is_replay = self.entries[self.entry]
        counts = torch.bincount(ids.reshape(-1), minlength=self.source.experts)
        self.exposure[:, fold] += counts.cpu().numpy()
        self.tokens[fold] += x.shape[0]
        order, tokens, counts = route_groups(ids, self.source.experts)
        scale = route.reshape(-1)[order].float()
        weights = self.load(layer)
        caps = (self.source.gate_cap, self.source.up_cap)
        start = 0
        for e, n in enumerate(counts):
            if not n:
                continue
            sel, r = tokens[start:start + n], scale[start:start + n, None]
            start += n
            xe = x[sel].float()
            if is_replay:
                held = sum(len(t) for t in self.replay[e][fold])
                if held < self.replay_cap:
                    self.replay[e][fold].append(xe[:self.replay_cap - held].bfloat16())
                continue
            h = down_input(xe, weights[0][e], weights[1][e], *caps) * r
            xe = xe * r
            self.hx[e].addmm_(xe.T, xe)
            self.hd[e].addmm_(h.T, h)
            self.n[e] += n

    def hessians(self, experts: slice, prior: float = 2.0):
        """Per-expert Hessians shrunk toward their diagonal by `prior` pseudo-tokens per input dim."""
        n = self.n[experts]
        hx, hd = self.hx[experts] / n.clamp_min(1)[:, None, None], self.hd[experts] / n.clamp_min(1)[:, None, None]
        return codecs.shrink(hx, n, prior * hx.shape[-1]), codecs.shrink(hd, n, prior * hd.shape[-1])

    def replay_inputs(self, expert: int, fold: int):
        parts = self.replay[expert][fold]
        return torch.cat(parts).float() if parts else None


def tail_index(W: torch.Tensor, block: int = 128, top: float = 0.1) -> torch.Tensor:
    """Hill estimate of the ESD power-law exponent over pooled 128x128 block spectra (AlphaQ-style)."""
    B, R, C = W.shape
    blocks = W.reshape(B, R // block, block, C // block, block).permute(0, 1, 3, 2, 4).reshape(B, -1, block, block)
    eig = torch.linalg.eigvalsh(blocks.transpose(-1, -2) @ blocks).reshape(B, -1).clamp_min(1e-30)
    k = max(2, int(top * eig.shape[1]))
    tail = eig.topk(k, -1).values
    return 1 + (k - 1) / torch.log(tail[:, :-1] / tail[:, -1:]).sum(-1)


def score_layer(stats: Stats, layer: int, formats: list, batch: int, prior: float = 2.0, log=print) -> dict:
    """Quantize every expert with every format (GPTQ, no GSQ refinement) and measure NatScale's
    joint held-out error M[e, fold, cg, cu, cd] = mean_t ||y_c - y_0||^2 / hidden."""
    source = stats.source
    E, K, S = source.experts, len(formats), stats.folds
    weights = stats.load(layer)
    caps = (source.gate_cap, source.up_cap)
    M = np.zeros((E, S, K, K, K))
    nbytes = np.zeros((E, 3, K), dtype=np.int64)
    variance, alpha = np.zeros((E, 3)), np.zeros((E, 3))
    replay_n = np.zeros((E, S), dtype=np.int64)
    for start in range(0, E, batch):
        span = slice(start, min(E, start + batch))
        native = [w[span] for w in weights]
        hx, hd = stats.hessians(span, prior)
        hessians = (hx, hx, hd)
        variance[span] = torch.stack([w.var((-1, -2), unbiased=False) for w in native], 1).cpu().numpy()
        alpha[span] = torch.stack([tail_index(w) for w in native], 1).cpu().numpy()
        decoded = []  # [format][projection] -> [b, rows, cols]
        for k, fmt in enumerate(formats):
            per = []
            for p, (w, h) in enumerate(zip(native, hessians)):
                q, raw = codecs.quantize(fmt, w, h)
                per.append(q)
                nbytes[span, p, k] = [r.numel() for r in raw]
            decoded.append(per)
        for b, e in enumerate(range(span.start, span.stop)):
            for s in range(S):
                x = stats.replay_inputs(e, s)
                if x is None:
                    continue
                replay_n[e, s] = len(x)
                y0 = expert_output(x, [w[b] for w in native], *caps)
                gate = [torch.nn.functional.linear(x, decoded[k][0][b]) for k in range(K)]
                up = [torch.nn.functional.linear(x, decoded[k][1][b]) for k in range(K)]
                downs = torch.stack([decoded[k][2][b] for k in range(K)])  # [K, hidden, ffn]
                errors = torch.empty(K, K, K, dtype=torch.float64, device=x.device)
                for cg, cu in itertools.product(range(K), range(K)):
                    y = torch.einsum("nf,khf->knh", activate(gate[cg], up[cu], *caps), downs)
                    errors[cg, cu] = (y - y0).double().square().sum((1, 2))
                M[e, s] = errors.cpu().numpy() / (len(x) * source.hidden)
        log(f"layer {layer}: scored experts {span.start}-{span.stop - 1}")
    return {"M": M, "bytes": nbytes, "variance": variance, "alpha": alpha, "replay_n": replay_n,
            "exposure": stats.exposure.copy(), "tokens": stats.tokens.copy(), "hessian_n": stats.n.cpu().numpy(),
            "formats": np.array([f.name for f in formats]), "bits": np.array([f.bpw for f in formats])}


def calibration_entries(tokens: dict[str, torch.Tensor], hessian_seqs: int, replay_seqs: int, batch: int):
    """Walk inputs as [batch, length] chunks plus (fold, is_replay) per chunk."""
    sequences, entries = [], []
    for fold, rows in enumerate(tokens.values()):
        for is_replay, part in ((False, rows[:hessian_seqs]), (True, rows[hessian_seqs:hessian_seqs + replay_seqs])):
            for chunk in part.split(batch) if len(part) else ():
                sequences.append(chunk)
                entries.append((fold, is_replay))
    return sequences, entries


def run(source, tokens: dict[str, torch.Tensor], output: Path, device: str, formats: list, *,
        hessian_seqs: int, replay_seqs: int, replay_cap: int, batch: int, expert_batch: int,
        layers: list[int] | None = None, on_layer=None, prior: float = 2.0, sequential: bool = False):
    """Walk the calibration set once; score each requested layer into output/layer-NNN.npz.

    `on_layer(stats, layer)` replaces scoring; if it returns decoded (gate, up, down) stacks, the
    layer is rerun with them so later layers see the quantized model (sequential quantization)."""
    output.mkdir(parents=True, exist_ok=True)
    sequences, entries = calibration_entries(tokens, hessian_seqs, replay_seqs, batch)
    wanted = set(range(source.layers) if layers is None else layers)
    with source.open(device) as reader:
        load = LayerCache(reader.layer)
        stats = Stats(source, load, entries, len(tokens), replay_cap, device)
        model = load_text_model(source, device, load, dense_device="cpu")

        def tap(layer, x, ids, route):
            if layer in wanted and stats.entry is not None:
                stats(layer, x, ids, route)

        for block in model.model.layers:
            block.mlp.experts.tap = tap
        started = [monotonic()]

        def before(layer, index):
            stats.entry = index
            if index == 0 and layer in wanted:
                stats.begin(layer)

        def done(layer):
            quantized = None
            if layer in wanted:
                if on_layer is None:
                    result = score_layer(stats, layer, formats, expert_batch, prior)
                    np.savez(output / f"layer-{layer:03d}.npz", **result)
                else:
                    quantized = on_layer(stats, layer)
                del stats.hx, stats.hd, stats.replay
            print(f"layer {layer} done ({monotonic() - started[0]:.0f}s)", flush=True)
            if layer >= max(wanted):
                raise StopWalk
            if quantized is not None:
                load.key, load.value, stats.entry = layer, quantized, None  # replay without the tap
            return quantized is not None

        try:
            walk(model, sequences, device, done, before, redo=sequential)
        except StopWalk:
            pass


class StopWalk(Exception):
    pass
