"""Routed-expert weight formats.

Every format packs one matrix into a self-describing byte payload (8-byte header first) and
is quantized per expert against its input Hessian H = sum r^2 x x^T / n:

* ``gsq2`` / ``int{b}g{g}``  GPTQ-initialised symmetric scalar grid with a signed FP16 scale
  per `g` inputs; the 2-bit/128 case is GSQ's format and can take GSQ's Gumbel refinement.
* ``sage{cap}``  SAGE-PTQ in a fixed-rate layout: sign plane, per-matrix magnitude table sized
  by SAGE's graph/silhouette step, a table label per 32 inputs, and the top-`cap` salient
  weights of each block with 8-bit positions and 4-bit magnitudes on a per-row FP16 scale.
* ``split{n}``  structured SAGE: the input channels carrying the most Hessian energy go to a
  multibit grid, the rest to SAGE binary; one class bit per input channel.
* ``iq1_s`` / ``iq1_m`` / ``iq2_xxs`` / ``tq1_0``  llama.cpp ggml types quantized with the
  Hessian diagonal as importance matrix, with GPTQ error feedback between 256-input blocks.
* ``binary256`` / ``scalar128``  the failed-baseline formats, kept as controls.
"""

from __future__ import annotations

import ctypes
import functools
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HEADER = 8  # bytes: format code, parameters, zero padding


def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Little-endian bitstream of `bits`-wide codes along the last dim, zero-padded to whole bytes."""
    n = codes.shape[-1]
    pad = -(n * bits) % 8
    planes = (codes.long()[..., None] >> torch.arange(bits, device=codes.device)) & 1
    planes = F.pad(planes.reshape(*codes.shape[:-1], n * bits), (0, pad)).reshape(*codes.shape[:-1], -1, 8)
    return (planes << torch.arange(8, device=codes.device)).sum(-1).to(torch.uint8)


def unpack_bits(raw: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    planes = (raw.long()[..., None] >> torch.arange(8, device=raw.device)) & 1
    planes = planes.reshape(*raw.shape[:-1], -1)[..., :count * bits].reshape(*raw.shape[:-1], count, bits)
    return (planes << torch.arange(bits, device=raw.device)).sum(-1)


def f16(x: torch.Tensor) -> torch.Tensor:
    return x.half().float()


def f16_bytes(x: torch.Tensor) -> torch.Tensor:
    return x.half().contiguous().view(torch.uint8).reshape(*x.shape[:-1], -1)


def bytes_f16(raw: torch.Tensor) -> torch.Tensor:
    return raw.contiguous().view(torch.float16).float()


def header(code: int, params, device) -> torch.Tensor:
    return torch.tensor([code, *params] + [0] * (HEADER - 1 - len(params)), dtype=torch.uint8, device=device)


# ---------------------------------------------------------------------------------------------
# GPTQ engine, batched over experts. A format supplies prepare / fit / round / pack.


def damped(H: torch.Tensor, damp: float) -> torch.Tensor:
    H = H.float().clone()
    diag = H.diagonal(dim1=-2, dim2=-1)
    diag[diag == 0] = 1
    diag += damp * diag.mean(-1, keepdim=True)
    return H


def shrink(H: torch.Tensor, n: torch.Tensor, prior: float) -> torch.Tensor:
    """Shrink sample Hessians toward their diagonal by prior/(n + prior) pseudo-tokens, so experts
    with few calibration tokens are not over-fitted by GPTQ's off-diagonal compensation."""
    alpha = (prior / (n.float() + prior))[:, None, None]
    return (1 - alpha) * H + alpha * torch.diag_embed(H.diagonal(dim1=-2, dim2=-1))


def hessian_inverse(H: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """Upper Cholesky factor of the damped inverse Hessian, as in GPTQ."""
    return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(damped(H, damp))), upper=True)


def gptq(fmt, W: torch.Tensor, H: torch.Tensor | None, damp: float = 0.01):
    """GPTQ over W [B, rows, cols]. `fmt.fit` sets block parameters from the error-updated
    weights; `fmt.round` then rounds column by column."""
    W = W.clone().float()
    U = None if H is None else hessian_inverse(H, damp)
    if U is not None:
        W.masked_fill_((H.diagonal(dim1=-2, dim2=-1) == 0)[:, None, :], 0)
    state = fmt.prepare(W, H)
    block = fmt.block
    Q = torch.empty_like(W)
    fitted = []
    for i1 in range(0, W.shape[-1], block):
        i2 = i1 + block
        W1 = W[..., i1:i2].clone()
        params = fmt.fit(state, W1, i1)
        fitted.append(params)
        Err = torch.zeros_like(W1)
        for i in range(block):
            w = W1[..., i]
            q = fmt.round(state, params, w, i)
            Q[..., i1 + i] = q
            if U is not None:
                err = (w - q) / U[:, i1 + i, i1 + i, None]
                W1[..., i:] -= err[..., None] * U[:, None, i1 + i, i1 + i:i2]
                Err[..., i] = err
        if U is not None:
            W[..., i2:] -= Err @ U[:, i1:i2, i2:]
    return Q, (state, fitted)


class GPTQFormat:
    def quantize(self, W, H, **_):
        Q, state = gptq(self, W, H)
        return Q, [p.cpu() for p in self.pack(W, Q, state)]


# ---------------------------------------------------------------------------------------------
# Scalar grid (GSQ's 2-bit format and wider integer grids).


@dataclass(frozen=True)
class Scalar(GPTQFormat):
    bits: int = 2
    group: int = 128
    odd: bool = False  # symmetric grid without zero, s * {-(2^b-1), ..., -1, 1, ..., 2^b-1}
    code = 1

    @property
    def name(self):
        name = "gsq2" if (self.bits, self.group) == (2, 128) else f"int{self.bits}g{self.group}"
        return name + ("odd" if self.odd else "")

    @property
    def block(self):
        return self.group

    @property
    def bpw(self):
        return self.bits + 16 / self.group

    @property
    def group_bytes(self):
        return 2 + self.group * self.bits // 8

    def nbytes(self, shape):
        rows, cols = shape
        return HEADER + rows * (cols // self.group) * self.group_bytes

    def prepare(self, W, H):
        return None

    def level(self, w, s):
        s = torch.where(s == 0, 1.0, s)  # zero groups decode to 0 on both grids
        top = 2 ** self.bits - 1
        if self.odd:
            return ((w / s + top) / 2).round().clamp(0, top)
        return ((w / s).round() + 2 ** (self.bits - 1)).clamp(0, top)

    def value(self, level, s):
        return s * (2 * level - (2 ** self.bits - 1) if self.odd else level - 2 ** (self.bits - 1))

    def quant(self, w, s):
        return self.value(self.level(w, s), s)

    def fit(self, state, W1, start):
        # GPTQ's symmetric MSE grid search (norm 2.4, 80 shrink steps) with the sign-flipped grid.
        xmax = W1.abs().amax(-1)
        top = 2 ** self.bits - 1
        base = torch.where(xmax > 0, (1 if self.odd else 2) * xmax / top, torch.ones_like(xmax))
        best, scale = torch.full_like(xmax, math.inf), base.clone()
        for p in 1 - torch.arange(80, device=W1.device) / 100:
            for sign in (1, -1):
                s = f16(sign * p * base)
                s = torch.where(s.abs() < 1e-7, 1.0, s)
                err = (self.quant(W1, s[..., None]) - W1).abs().pow(2.4).sum(-1)
                better = err < best
                best, scale = torch.where(better, err, best), torch.where(better, s, scale)
        return torch.where(xmax > 0, scale, 0.0)

    def round(self, state, scale, w, i):
        return self.quant(w, scale)

    def codes(self, q, s):
        return self.level(q.reshape(*s.shape, self.group), s[..., None])

    def pack(self, W, Q, packed_state):
        scales = torch.stack(packed_state[1], -1)  # [B, rows, groups]
        out = []
        for q, s in zip(Q, scales):
            body = torch.cat((f16_bytes(s[..., None]), pack_bits(self.codes(q, s), self.bits)), -1)
            out.append(torch.cat((header(self.code, (self.bits, self.group // 32, int(self.odd)), q.device),
                                  body.reshape(-1))))
        return out

    @staticmethod
    def decode(raw, shape):
        rows, cols = shape
        fmt = Scalar(int(raw[1]), 32 * int(raw[2]), bool(raw[3]))
        body = raw[HEADER:].reshape(rows, cols // fmt.group, -1)
        codes = unpack_bits(body[..., 2:], fmt.bits, fmt.group)
        return fmt.value(codes, bytes_f16(body[..., :2])).reshape(rows, cols)

    def quantize(self, W, H, gsq_steps: int = 0, seed: int = 0, **_):
        Q, state = gptq(self, W, H)
        if gsq_steps and self.bits == 2 and not self.odd and H is not None:
            scales = torch.stack(state[1], -1)
            Q, scales = gsq_refine(W, H, Q, scales, self.group, gsq_steps, seed)
            state = (None, list(scales.unbind(-1)))
        return Q, [p.cpu() for p in self.pack(W, Q, state)]


def lion_step(params, grads, moments, lrs, wd, betas=(0.9, 0.95)):
    for p, g, m, lr, decay in zip(params, grads, moments, lrs, wd):
        p.mul_(1 - lr * decay)
        p.add_(torch.sign(m * betas[0] + g * (1 - betas[0])), alpha=-lr)
        m.mul_(betas[1]).add_(g, alpha=1 - betas[1])


def gsq_refine(W, H, Q, scales, group, steps: int, seed: int = 0, damp: float = 0.01,
               std=0.01, strength=6.0, temperature=(2.0, 0.05), kappa=(100.0, 500.0),
               lr=(1e-4, 5e-5), weight_decay=1.0):
    """GSQ Gumbel-softmax refinement of 2-bit codes and scales against ||(W - Wq) L||^2, H = L L^T.

    Hyperparameters follow GSQ's GPTQ-path defaults (Lion, 2,000 steps, constant LR); fewer steps
    scale the learning rates so total logit travel matches. Rows keep the result only if it
    lowers their Hessian-weighted error.
    """
    L = torch.linalg.cholesky(damped(H, damp))
    gen = torch.Generator(device=W.device).manual_seed(seed)
    grid = W.new_tensor((-2.0, -1.0, 0.0, 1.0))
    expand = lambda s: s.repeat_interleave(group, -1)
    z = Q / expand(scales)
    prior = -0.5 * (z[..., None] - grid).square()
    prior = prior - prior.mean(-1, keepdim=True)
    logits = std * (torch.randn(prior.shape, generator=gen, device=W.device) + strength * prior)
    s = scales.clone()
    params, moments = [logits, s], [torch.zeros_like(logits), torch.zeros_like(s)]
    factor = 2000 / steps
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        for step in range(steps):
            t = step / max(1, steps - 1)
            tau = temperature[0] + (temperature[1] - temperature[0]) * t
            k = kappa[0] + (kappa[1] - kappa[0]) * t
            u = torch.rand(logits.shape, generator=gen, device=W.device)
            noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            with torch.enable_grad():
                for p in params:
                    p.requires_grad_(True)
                soft = ((logits * k + noise) / tau).softmax(-1) @ grid
                loss = ((W - soft * expand(s)) @ L).square().sum()
                grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for p in params:
                    p.requires_grad_(False)
                lion_step(params, grads, moments, [lr[0] * factor, lr[1] * factor], [weight_decay, 0.0])
        with torch.no_grad():
            s = f16(s)
            hard = grid[logits.argmax(-1)] * expand(s)
            before = ((W - Q) @ L).square().sum(-1)
            after = ((W - hard) @ L).square().sum(-1)
            keep = (after < before)[..., None]
            return torch.where(keep, hard, Q), torch.where(keep, s, scales)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


# ---------------------------------------------------------------------------------------------
# SAGE-PTQ, fixed-rate deployable layout.


def kmeans(x: torch.Tensor, k: int, gen: torch.Generator, iters: int = 50, restarts: int = 4):
    best, best_labels = math.inf, None
    for _ in range(restarts):
        centers = x[torch.randperm(len(x), generator=gen)[:k]]
        labels = None
        for _ in range(iters):
            new = torch.cdist(x, centers).argmin(1)
            if labels is not None and torch.equal(new, labels):
                break
            labels = new
            counts = torch.bincount(labels, minlength=k)[:, None]
            sums = torch.zeros_like(centers).index_add_(0, labels, x)
            centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
        inertia = float((x - centers[labels]).square().sum())
        if inertia < best:
            best, best_labels = inertia, labels
    return best_labels


def silhouette(x: torch.Tensor, labels: torch.Tensor) -> float:
    d = torch.cdist(x, x)
    onehot = F.one_hot(labels).to(d.dtype)
    sizes = onehot.sum(0)
    mean = d @ onehot / sizes.clamp_min(1)
    own = labels[:, None]
    a = mean.gather(1, own).squeeze(1) * sizes[labels] / (sizes[labels] - 1).clamp_min(1)
    b = mean.scatter(1, own, math.inf).amin(1)
    s = (b - a) / torch.maximum(a, b).clamp_min(1e-12)
    return float(torch.where(sizes[labels] > 1, s, 0).mean())


def graph_group_count(values: torch.Tensor, candidates, neighbours: int = 5, samples: int = 512, seed: int = 0) -> int:
    """SAGE Alg. 1: spectral clustering of a sampled kNN graph; the count with the best silhouette."""
    gen = torch.Generator().manual_seed(seed)
    values = values.cpu()
    x = values[torch.randperm(len(values), generator=gen)[:samples]].reshape(-1, 1).double()
    d = torch.cdist(x, x)
    knn = d.topk(neighbours + 1, largest=False).indices[:, 1:]
    sigma = d.gather(1, knn[:, -1:]).squeeze(1).clamp_min(1e-12)  # self-tuning scale
    mask = torch.zeros_like(d, dtype=torch.bool).scatter_(1, knn, True)
    A = torch.where(mask | mask.T, torch.exp(-d.square() / (sigma[:, None] * sigma[None, :])), 0)
    deg = A.sum(1).clamp_min(1e-12).rsqrt()
    _, vectors = torch.linalg.eigh(torch.eye(len(x), dtype=x.dtype) - deg[:, None] * A * deg[None, :])
    scores = {}
    for r in candidates:
        scores[r] = silhouette(x, kmeans(F.normalize(vectors[:, :r], dim=1), r, gen))
    return max(scores, key=scores.get)


def lloyd(values: torch.Tensor, k: int, iters: int = 30) -> torch.Tensor:
    """1-D Lloyd-Max levels started from equal-mass quantiles (SAGE Eq. 4, corrected)."""
    values = values.float()
    if values.numel() > 2 ** 24:
        values = values[torch.randperm(values.numel(), device=values.device)[:2 ** 24]]
    levels = torch.quantile(values, (torch.arange(k, device=values.device) + 0.5) / k)
    for _ in range(iters):
        labels = (values[:, None] - levels).abs().argmin(1)
        sums = torch.zeros_like(levels).index_add_(0, labels, values)
        counts = torch.bincount(labels, minlength=k)
        levels = torch.where(counts > 0, sums / counts.clamp_min(1), levels)
    return levels.sort().values


@dataclass(frozen=True)
class Sage(GPTQFormat):
    """SAGE-PTQ with every lookup, position, and scale stored in fixed-size blocks."""

    cap: int = 2           # salient weights per block (8-bit positions + 4-bit magnitudes)
    lut_bits: int = 4      # label width per 32 inputs; table size from SAGE's graph step
    graph: bool = True     # False: all 2**lut_bits entries (Lloyd)
    block: int = 256
    sub: int = 32
    code = 2

    @property
    def name(self):
        name = f"sage{self.cap}"
        if self.lut_bits != 4:
            name += f"l{self.lut_bits}"
        if self.block != 256:
            name += f"b{self.block}"
        return name + ("" if self.graph else "f")

    @property
    def label_bytes(self):
        return -(-(self.block // self.sub) * self.lut_bits // 8)

    @property
    def block_bytes(self):
        return self.block // 8 + self.label_bytes + self.cap + (self.cap + 1) // 2

    @property
    def bpw(self):
        return 8 * self.block_bytes / self.block

    def nbytes(self, shape):
        rows, cols = shape
        return HEADER + 32 + rows * (2 * (self.cap > 0) + (cols // self.block) * self.block_bytes)

    def prepare(self, W, H):
        B, rows, cols = W.shape
        mag = W.abs().reshape(B, rows, -1, self.block)
        if self.cap:
            top = mag.topk(self.cap, -1)
            row_scale = f16(top.values.amax((-1, -2)))
            row_scale = torch.where(row_scale > 0, row_scale, 1.0)  # all-zero rows
            mask = torch.zeros_like(mag, dtype=torch.bool).scatter_(-1, top.indices, True)
        else:
            row_scale, mask = torch.ones(B, rows, device=W.device), torch.zeros_like(mag, dtype=torch.bool)
        inlier = torch.where(mask, 0, mag).reshape(B, rows, -1, self.sub)
        counts = (~mask).reshape(B, rows, -1, self.sub).sum(-1)
        sub_means = inlier.sum(-1) / counts.clamp_min(1)
        size_max = 2 ** self.lut_bits
        tables = []
        for b in range(B):
            values = sub_means[b][counts[b] > 0]
            zero = bool((values == 0).any())  # dead neurons: keep an exact zero level
            values = values[values > 0]
            size = size_max
            if self.graph and values.numel() > 2 * size_max:
                size = graph_group_count(values, range(size_max // 2 + 1, size_max + 1), seed=b)
            size -= zero
            levels = lloyd(values, size) if values.numel() else torch.zeros(1, device=W.device)
            if zero:
                levels = torch.cat((levels.new_zeros(1), levels))
            tables.append(F.pad(levels, (0, 16 - len(levels)), value=float(levels[-1])))
        return row_scale, f16(torch.stack(tables))

    def fit(self, state, W1, start):
        row_scale, table = state
        mag = W1.abs()
        salient = torch.zeros_like(mag, dtype=torch.bool)
        if self.cap:
            salient.scatter_(-1, mag.topk(self.cap, -1).indices, True)
        inlier = torch.where(salient, 0, mag).reshape(*mag.shape[:-1], -1, self.sub)
        means = inlier.sum(-1) / (~salient).reshape(inlier.shape).sum(-1).clamp_min(1)
        labels = (means[..., None] - table[:, None, None, :]).abs().argmin(-1)
        values = table.gather(1, labels.reshape(len(table), -1)).reshape(labels.shape)
        return salient, values.repeat_interleave(self.sub, -1), labels

    def round(self, state, params, w, i):
        row_scale, _ = state
        salient, values, _ = params
        sign = torch.where(w >= 0, 1.0, -1.0)
        level = (w.abs() / row_scale * 15).round().clamp(0, 15) * row_scale / 15
        return sign * torch.where(salient[..., i], level, values[..., i])

    def pack(self, W, Q, packed_state):
        (row_scale, table), fitted = packed_state
        salient = torch.cat([p[0] for p in fitted], -1)
        labels = torch.cat([p[2] for p in fitted], -1)
        out = []
        for b, q in enumerate(Q):
            rows, cols = q.shape
            blocks = cols // self.block
            parts = [pack_bits((q >= 0).reshape(rows, blocks, self.block), 1),
                     pack_bits(labels[b].reshape(rows, blocks, -1), self.lut_bits)]
            if self.cap:
                pos = salient[b].reshape(rows, blocks, self.block).nonzero()[:, 2].reshape(rows, blocks, self.cap)
                mag = q.abs().reshape(rows, blocks, self.block).gather(-1, pos)
                codes = (mag / row_scale[b][:, None, None] * 15).round().clamp(0, 15)
                parts += [pos.to(torch.uint8), pack_bits(codes, 4)]
            body = torch.cat(parts, -1).reshape(rows, -1)
            if self.cap:
                body = torch.cat((f16_bytes(row_scale[b][:, None]), body), -1)
            out.append(torch.cat((header(self.code, (self.cap, self.lut_bits, self.block // 32, self.sub), q.device),
                                  f16_bytes(table[b][None])[0], body.reshape(-1))))
        return out

    @staticmethod
    def decode(raw, shape):
        rows, cols = shape
        cap, lut_bits, block, sub = int(raw[1]), int(raw[2]), 32 * int(raw[3]), int(raw[4])
        fmt = Sage(cap=cap, lut_bits=lut_bits, block=block, sub=sub)
        table = bytes_f16(raw[HEADER:HEADER + 32])
        body = raw[HEADER + 32:].reshape(rows, -1)
        row_scale = None
        if cap:
            row_scale, body = bytes_f16(body[:, :2])[:, 0], body[:, 2:]
        blocks = cols // block
        body = body.reshape(rows, blocks, -1)
        sign = unpack_bits(body[..., :block // 8], 1, block) * 2 - 1
        at = block // 8
        labels = unpack_bits(body[..., at:at + fmt.label_bytes], lut_bits, block // sub)
        at += fmt.label_bytes
        values = table[labels].repeat_interleave(sub, -1)
        if cap:
            pos = body[..., at:at + cap].long()
            codes = unpack_bits(body[..., at + cap:], 4, cap)
            values = values.scatter(-1, pos, codes * row_scale[:, None, None] / 15)
        return (sign * values).reshape(rows, cols)


# ---------------------------------------------------------------------------------------------
# Structured SAGE: Hessian-salient input channels on a multibit grid, the rest SAGE binary.


@dataclass(frozen=True)
class Split(GPTQFormat):
    salient: int = 128        # salient input channels (a multiple of 64)
    high: Scalar = field(default_factory=lambda: Scalar(bits=4, group=64))
    low: Sage = field(default_factory=lambda: Sage(cap=0, block=64))
    code = 5
    block = 64

    @property
    def name(self):
        return f"split{self.salient}" + ("" if self.high.bits == 4 else f"i{self.high.bits}")

    @property
    def bpw(self):  # nominal for 2048 inputs; exact sizes come from nbytes
        f = self.salient / 2048
        return f * self.high.bpw + (1 - f) * self.low.bpw

    def nbytes(self, shape):
        rows, cols = shape
        k = min(self.salient, cols)
        low = self.low.nbytes((rows, cols - k)) if cols > k else 0
        return HEADER + cols // 8 + self.high.nbytes((rows, k)) + low

    def quantize(self, W, H, **_):
        B, rows, cols = W.shape
        k = min(self.salient, cols)
        importance = W.square().sum(1)
        if H is not None:
            importance = importance * H.diagonal(dim1=-2, dim2=-1)
        chosen = torch.zeros(B, cols, dtype=torch.bool, device=W.device)
        chosen.scatter_(1, importance.topk(k, -1).indices, True)
        perm = torch.argsort((~chosen).int(), dim=1, stable=True)  # salient first, original order kept
        Wp = W.gather(2, perm[:, None, :].expand(B, rows, cols))
        Hp = None if H is None else H.gather(1, perm[:, :, None].expand(B, cols, cols)).gather(
            2, perm[:, None, :].expand(B, cols, cols))
        Qp, (state, fitted) = gptq(_Regions(self, k), Wp, Hp)
        nh = k // self.block
        high = self.high.pack(Wp[..., :k], Qp[..., :k], (state[0], [p for _, p in fitted[:nh]]))
        low = (self.low.pack(Wp[..., k:], Qp[..., k:], (state[1], [p for _, p in fitted[nh:]]))
               if k < cols else [None] * B)
        Q = torch.empty_like(Qp).scatter_(2, perm[:, None, :].expand(B, rows, cols), Qp)
        out = []
        for b in range(B):
            parts = [header(self.code, (k // 64,), W.device), pack_bits(chosen[b].int(), 1), high[b]]
            if low[b] is not None:
                parts.append(low[b])
            out.append(torch.cat(parts).cpu())
        return Q, out

    @staticmethod
    def decode(raw, shape):
        rows, cols = shape
        k = 64 * int(raw[1])
        at = HEADER + cols // 8
        chosen = unpack_bits(raw[HEADER:at], 1, cols).bool()
        high = Scalar(int(raw[at + 1]), 32 * int(raw[at + 2]), bool(raw[at + 3]))
        hb = high.nbytes((rows, k))
        Wp = [Scalar.decode(raw[at:at + hb], (rows, k))]
        if k < cols:
            Wp.append(Sage.decode(raw[at + hb:], (rows, cols - k)))
        perm = torch.argsort((~chosen).int(), stable=True)
        return torch.empty(rows, cols, device=raw.device).index_copy_(1, perm, torch.cat(Wp, 1))


class _Regions:
    """GPTQ adapter: the first `k` (permuted) inputs use the high format, the rest the low one."""

    def __init__(self, split: Split, k: int):
        self.split, self.k, self.block = split, k, split.block

    def prepare(self, W, H):
        k = self.k
        return (self.split.high.prepare(W[..., :k], None),
                self.split.low.prepare(W[..., k:], None) if k < W.shape[-1] else None)

    def fit(self, state, W1, start):
        high = start < self.k
        fmt = self.split.high if high else self.split.low
        return high, fmt.fit(state[0 if high else 1], W1, start)

    def round(self, state, params, w, i):
        high, inner = params
        fmt = self.split.high if high else self.split.low
        return fmt.round(state[0 if high else 1], inner, w, i)


# ---------------------------------------------------------------------------------------------
# llama.cpp ggml types, Hessian diagonal as importance matrix, GPTQ feedback between 256-blocks.

GGML_TYPES = {"iq1_s": (19, 50), "iq1_m": (29, 56), "iq2_xxs": (16, 66), "tq1_0": (34, 54)}
_ggml = None


def ggml():
    global _ggml
    if _ggml is None:
        root = Path(os.environ.get("LOW_BPW_GGML", Path(__file__).resolve().parents[2] / "llama.cpp/build-cpu/bin"))
        lib = ctypes.CDLL(str(root / "libggml-base.so"))
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        for name in GGML_TYPES:
            getattr(lib, f"dequantize_row_{name}").argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        _ggml = lib
    return _ggml


_POOL = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2))


def ggml_quantize(kind: str, W: np.ndarray, importance: np.ndarray | None) -> np.ndarray:
    """W [rows, cols] float32 (cols % 256 == 0) -> raw ggml blocks [rows, cols/256*block bytes]."""
    type_id, size = GGML_TYPES[kind]
    rows, cols = W.shape
    W = np.ascontiguousarray(W, dtype=np.float32)
    out = np.empty((rows, cols // 256 * size), dtype=np.uint8)
    imp = None if importance is None else np.ascontiguousarray(importance, dtype=np.float32)
    lib = ggml()
    step = max(1, -(-rows // 14))

    def chunk(r0):
        r1 = min(rows, r0 + step)
        lib.ggml_quantize_chunk(type_id, W[r0:r1].ctypes.data, out[r0:r1].ctypes.data, 0, r1 - r0, cols,
                                None if imp is None else imp.ctypes.data)

    list(_POOL.map(chunk, range(0, rows, step)))
    return out


def ggml_dequantize(kind: str, raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    out = np.empty((rows, cols), dtype=np.float32)
    getattr(ggml(), f"dequantize_row_{kind}")(raw.ctypes.data, out.ctypes.data, rows * cols)
    return out


@dataclass(frozen=True)
class GGML:
    kind: str = "iq1_s"
    feedback: bool = True  # GPTQ error feedback between 256-input blocks
    code = 6

    @property
    def name(self):
        return self.kind + ("" if self.feedback else "-rtn")

    @property
    def bpw(self):
        return GGML_TYPES[self.kind][1] * 8 / 256

    def nbytes(self, shape):
        rows, cols = shape
        return HEADER + rows * cols // 256 * GGML_TYPES[self.kind][1]

    def quantize(self, W, H, **_):
        B, rows, cols = W.shape
        size = GGML_TYPES[self.kind][1]
        importance = None if H is None else H.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
        W = W.clone().float()
        U = hessian_inverse(H) if (H is not None and self.feedback) else None
        Q = torch.empty_like(W)
        raw = np.empty((B, rows, cols // 256 * size), dtype=np.uint8)
        for i1 in range(0, cols, 256):
            i2 = i1 + 256
            W1 = W[..., i1:i2].clone()
            host = W1.cpu().numpy()
            for b in range(B):
                imp = None if importance is None else importance[b, i1:i2].cpu().numpy()
                blocks = ggml_quantize(self.kind, host[b], imp)
                raw[b, :, i1 // 256 * size:i2 // 256 * size] = blocks
                Q[b, :, i1:i2] = torch.from_numpy(ggml_dequantize(self.kind, blocks, rows, 256)).to(W.device)
            if U is not None:
                Err = torch.zeros_like(W1)
                for i in range(256):
                    err = (W1[..., i] - Q[..., i1 + i]) / U[:, i1 + i, i1 + i, None]
                    W1[..., i:] -= err[..., None] * U[:, None, i1 + i, i1 + i:i2]
                    Err[..., i] = err
                W[..., i2:] -= Err @ U[:, i1:i2, i2:]
        code = list(GGML_TYPES).index(self.kind)
        return Q, [torch.cat((header(self.code, (code,), "cpu"), torch.from_numpy(r.reshape(-1)))) for r in raw]

    @staticmethod
    def decode(raw, shape):
        kind = list(GGML_TYPES)[int(raw[1])]
        rows, cols = shape
        values = ggml_dequantize(kind, raw[HEADER:].cpu().numpy(), rows, cols)
        return torch.from_numpy(values).to(raw.device)


# ---------------------------------------------------------------------------------------------
# Incoherence processing and vector quantization on a universal Gaussian codebook.


@functools.lru_cache(maxsize=None)
def rotation(n: int, seed: int = 0, device: str = "cpu") -> torch.Tensor:
    """Orthogonal R (x' = R x): randomized Hadamard H_n diag(signs) / sqrt(n) for powers of two."""
    gen = torch.Generator().manual_seed(seed * 1_000_003 + n)
    signs = torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0)
    if n & (n - 1) == 0:
        R = torch.ones(1, 1)
        while len(R) < n:
            R = torch.cat((torch.cat((R, R), 1), torch.cat((R, -R), 1)))
        R = R * signs / math.sqrt(n)
    else:  # ponytail: dense random orthogonal; a kernel wants Hadamard-friendly sizes (e.g. 12 * 2^k)
        R = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))[0].float()
    return R.to(device)


@dataclass(frozen=True)
class Rotated:
    """Quantize W R^T against R H R^T (QuIP-style incoherence); decode multiplies back by R."""
    inner: object
    seed: int = 0
    code = 7

    @property
    def name(self):
        return "r" + self.inner.name

    @property
    def bpw(self):
        return self.inner.bpw

    def nbytes(self, shape):
        return HEADER + self.inner.nbytes(shape)

    def quantize(self, W, H, **options):
        R = rotation(W.shape[-1], self.seed, str(W.device))
        Q, raw = self.inner.quantize(W @ R.T, None if H is None else R @ H @ R.T, **options)
        return Q @ R, [torch.cat((header(self.code, (self.seed,), "cpu"), r)) for r in raw]

    @staticmethod
    def decode(raw, shape):
        return decode(raw[HEADER:], shape) @ rotation(shape[1], int(raw[1]), str(raw.device))


CODEBOOKS: dict[tuple[int, int], torch.Tensor] = {}


def nearest(x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Index of the nearest codeword for each row of x (exact FP32 distances)."""
    norms = C.square().sum(1)
    if x.is_cuda and len(C) >= 1024 and C.shape[1] <= 16 and len(C) % 128 == 0:
        return _nearest_triton(x, C, norms)
    step = max(1, 2 ** 27 // len(C))
    return torch.cat([(norms - 2 * part @ C.T).argmin(1) for part in x.split(step)])


def _nearest_triton(x, C, norms):
    import triton
    import triton.language as tl

    global _NEAREST
    if "_NEAREST" not in globals():
        @triton.jit
        def kernel(X, C, N, out, n, k, D: tl.constexpr, BQ: tl.constexpr, BC: tl.constexpr):
            q = tl.program_id(0) * BQ + tl.arange(0, BQ)
            d = tl.arange(0, D)
            xs = tl.load(X + q[:, None] * D + d[None, :], mask=q[:, None] < n, other=0.0)
            best = tl.full([BQ], float("inf"), tl.float32)
            arg = tl.zeros([BQ], tl.int32)
            for c0 in range(0, k, BC):
                c = c0 + tl.arange(0, BC)
                dist = tl.load(N + c)[None, :] - 2 * tl.dot(xs, tl.load(C + c[None, :] * D + d[:, None]),
                                                             input_precision="ieee")
                m = tl.min(dist, 1)
                a = tl.argmin(dist, 1).to(tl.int32) + c0
                better = m < best
                best, arg = tl.where(better, m, best), tl.where(better, a, arg)
            tl.store(out + q, arg, mask=q < n)

        _NEAREST = kernel
    D = 16
    xs = F.pad(x.float(), (0, D - x.shape[1])).contiguous()
    cs = F.pad(C.float(), (0, D - C.shape[1])).contiguous()
    out = torch.empty(len(xs), dtype=torch.int32, device=x.device)
    BQ = 64
    _NEAREST[(triton.cdiv(len(xs), BQ),)](xs, cs, norms.float().contiguous(), out, len(xs), len(cs), D=D, BQ=BQ, BC=128)
    return out.long()


def codebook(dim: int, bits: int, device) -> torch.Tensor:
    """Universal k-means codebook for N(0, I_dim), trained once and cached; banks store a copy."""
    key = (dim, bits)
    if key not in CODEBOOKS:
        path = Path(os.environ.get("LOW_BPW_CACHE", Path.home() / ".cache/low_bpw_quants")) / f"vq{dim}x{bits}.pt"
        if path.is_file():
            CODEBOOKS[key] = torch.load(path)
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            gen = torch.Generator(device=dev).manual_seed(bits)
            k = 2 ** bits
            x = torch.randn(k * 64, dim, generator=gen, device=dev)
            C = x[:k].clone()
            for _ in range(30):
                labels = nearest(x, C)
                sums = torch.zeros_like(C).index_add_(0, labels, x)
                counts = torch.bincount(labels, minlength=k)[:, None]
                C = torch.where(counts > 0, sums / counts.clamp_min(1), C)
            CODEBOOKS[key] = C.half().cpu()
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(CODEBOOKS[key], path)
    return CODEBOOKS[key].to(device).float()


@dataclass(frozen=True)
class VQ:
    """`dim`-dimensional VQ on a universal Gaussian codebook of 2^bits entries, per-row FP16 scale,
    GPTQ/LDLQ feedback across columns, then Hessian least-squares refit of the row scales."""
    dim: int = 8
    bits: int = 12
    passes: int = 0  # coordinate-descent passes after LDLQ (each keeps only improving codes)
    code = 8
    block = 128

    @property
    def name(self):
        return f"vq{self.dim}x{self.bits}" + (f"p{self.passes}" if self.passes else "")

    @property
    def bpw(self):
        return self.bits / self.dim

    def row_bytes(self, cols):
        return 2 + -(-cols // self.dim * self.bits // 8)

    def nbytes(self, shape):
        rows, cols = shape
        return HEADER + rows * self.row_bytes(cols)

    def quantize(self, W, H, **_):
        B, rows, cols = W.shape
        C = codebook(self.dim, self.bits, W.device)
        W0 = W.float()
        W = W0.clone()
        U = None if H is None else hessian_inverse(H)
        if U is not None:
            W.masked_fill_((H.diagonal(dim1=-2, dim2=-1) == 0)[:, None, :], 0)
        s = f16(W.square().mean(-1).sqrt())
        safe = torch.where(s > 0, s, 1.0)[..., None]
        idx = torch.empty(B, rows, cols // self.dim, dtype=torch.long, device=W.device)
        Q = torch.empty_like(W)
        for i1 in range(0, cols, self.block):
            i2 = min(cols, i1 + self.block)
            W1 = W[..., i1:i2].clone()
            Err = torch.zeros_like(W1)
            for j in range(0, i2 - i1, self.dim):
                k = nearest((W1[..., j:j + self.dim] / safe).reshape(-1, self.dim), C).reshape(B, rows)
                idx[..., (i1 + j) // self.dim] = k
                Q[..., i1 + j:i1 + j + self.dim] = C[k] * s[..., None]
                if U is not None:
                    for i in range(j, j + self.dim):
                        err = (W1[..., i] - Q[..., i1 + i]) / U[:, i1 + i, i1 + i, None]
                        W1[..., i:] -= err[..., None] * U[:, None, i1 + i, i1 + i:i2]
                        Err[..., i] = err
            if U is not None:
                W[..., i2:] -= Err @ U[:, i1:i2, i2:]
        Hf = None if H is None else H.float()

        def refit(s):  # Hessian least-squares row scales for the current codes
            unit = C[idx].reshape(B, rows, cols)
            weighted = unit if Hf is None else unit @ Hf
            num, den = (W0 * weighted).sum(-1), (unit * weighted).sum(-1)
            return unit, torch.where((den > 0) & (s > 0), f16(num / den.clamp_min(1e-30)), s)

        unit, s = refit(s)
        safe = torch.where(s > 0, s, 1.0)[..., None]
        for _ in range(self.passes if Hf is not None else 0):
            Q = unit * s[..., None]
            EH = (W0 - Q) @ Hf
            Hd = damped(Hf, 0.01)
            for j in range(0, cols, self.dim):
                J = slice(j, j + self.dim)
                step = torch.linalg.solve(Hd[:, J, J], EH[..., J].transpose(1, 2)).transpose(1, 2)
                k = nearest(((Q[..., J] + step) / safe).reshape(-1, self.dim), C).reshape(B, rows)
                delta = C[k] * s[..., None] - Q[..., J]
                gain = 2 * (delta * EH[..., J]).sum(-1) - torch.einsum("brd,bde,bre->br", delta, Hf[:, J, J], delta)
                delta = delta * (gain > 0)[..., None]
                idx[..., j // self.dim] = torch.where(gain > 0, k, idx[..., j // self.dim])
                Q[..., J] += delta
                EH -= delta @ Hf[:, J, :]
            unit, s = refit(s)
            safe = torch.where(s > 0, s, 1.0)[..., None]
        Q = unit * s[..., None]
        out = []
        for b in range(B):
            body = torch.cat((f16_bytes(s[b][:, None]), pack_bits(idx[b], self.bits)), -1)
            out.append(torch.cat((header(self.code, (self.dim, self.bits), W.device), body.reshape(-1))).cpu())
        return Q, out

    @staticmethod
    def decode(raw, shape):
        rows, cols = shape
        dim, bits = int(raw[1]), int(raw[2])  # passes only change how codes were chosen
        body = raw[HEADER:].reshape(rows, -1)
        idx = unpack_bits(body[:, 2:], bits, cols // dim)
        return (codebook(dim, bits, raw.device)[idx] * bytes_f16(body[:, :2])[:, :, None]).reshape(rows, cols)


# ---------------------------------------------------------------------------------------------
# Bit-shift trellis-coded quantization (QTIP-style). The state is the last L bits of the code
# stream; each step shifts in kV bits and emits LUT[state] (V dims), so decode is a sliding
# window over the bits (random access). Viterbi finds the MSE-optimal path per tile.

_VIT = {}


def _viterbi_kernels():
    if _VIT:
        return _VIT
    import triton
    import triton.language as tl

    @triton.jit
    def fwd(X, LUT, BP, END, T, S: tl.constexpr, P: tl.constexpr, V: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        s = tl.arange(0, S)
        x0 = X + pid * T * V
        cost = tl.zeros([S], tl.float32)
        for j in tl.static_range(V):
            e0 = tl.load(x0 + j) - tl.load(LUT + j * S + s)
            cost += e0 * e0
        for t in range(1, T):
            d = tl.zeros([S], tl.float32)
            for j in tl.static_range(V):
                e = tl.load(x0 + t * V + j) - tl.load(LUT + j * S + s)
                d += e * e
            c2 = tl.reshape(cost, (P, S // P))
            m = tl.min(c2, 0)
            a = tl.argmin(c2, 0)
            tl.store(BP + (pid * T + t) * (S // P) + tl.arange(0, S // P), a.to(tl.uint8))
            cost = tl.reshape(tl.broadcast_to(m[:, None], (S // P, P)), (S,)) + d
        tl.store(END + pid, tl.argmin(cost, 0).to(tl.int32))

    @triton.jit
    def back(BP, END, OUT, T, SP: tl.constexpr, SH: tl.constexpr, KV: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        s = tl.load(END + pid)
        for i in range(T):
            t = T - 1 - i
            tl.store(OUT + pid * T + t, s)
            top = tl.load(BP + (pid * T + t) * SP + (s >> KV)).to(tl.int32)
            s = (s >> KV) | (top << SH)

    _VIT.update(fwd=fwd, back=back)
    return _VIT


def viterbi(x: torch.Tensor, lut: torch.Tensor, L: int, kV: int) -> torch.Tensor:
    """x [N, T, V] -> MSE-optimal state sequence [N, T] (int64) of the bit-shift trellis."""
    k = _viterbi_kernels()
    N, T, V = x.shape
    S, P = 1 << L, 1 << kV
    x = x.float().contiguous()
    lut = lut.float().T.contiguous()  # [V, S]: coalesced per-component loads
    out = torch.empty(N, T, dtype=torch.int32, device=x.device)
    chunk = max(1, (1 << 28) // (T * S // P))  # backpointer buffer <= 256 MiB
    for a in range(0, N, chunk):
        n = min(chunk, N - a)
        bp = torch.empty(n, T, S // P, dtype=torch.uint8, device=x.device)
        end = torch.empty(n, dtype=torch.int32, device=x.device)
        k["fwd"][(n,)](x[a:a + n], lut, bp, end, T, S=S, P=P, V=V, num_warps=4 if L <= 12 else 8 if L <= 13 else 16 if L <= 14 else 32)
        k["back"][(n,)](bp, end, out[a:a + n], T, SP=S // P, SH=L - kV, KV=kV, num_warps=1)
    return out.long()


def trellis_lut(V: int, kV: int, L: int, device) -> torch.Tensor:
    """Universal LUT for N(0, I) trained by alternating Viterbi and centroid updates; cached."""
    key = ("tcq", V, kV, L)
    if key not in CODEBOOKS:
        path = Path(os.environ.get("LOW_BPW_CACHE", Path.home() / ".cache/low_bpw_quants")) / f"tcq{V}x{kV}l{L}.pt"
        if path.is_file():
            CODEBOOKS[key] = torch.load(path)
        else:
            dev = "cuda"
            gen = torch.Generator(device=dev).manual_seed(1000 + L * 16 + kV)
            S = 1 << L
            x = torch.randn(max(256, S // 8), 512, V, generator=gen, device=dev)  # ~>=256 visits/state
            lut = torch.randn(S, V, generator=gen, device=dev)
            for _ in range(12):
                st = viterbi(x, lut, L, kV).flatten()
                sums = torch.zeros_like(lut).index_add_(0, st, x.reshape(-1, V))
                cnt = torch.bincount(st, minlength=S)[:, None]
                lut = torch.where(cnt > 0, sums / cnt.clamp_min(1), lut)
            CODEBOOKS[key] = lut.half().cpu()
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(CODEBOOKS[key], path)
    return CODEBOOKS[key].to(device).float()


def _tiles(X, R, V):  # [B, rows, g] -> [B*rows/R, R*g/V, V], row-major inside each R x g tile
    B, rows, g = X.shape
    return X.reshape(B * rows // R, R * g // V, V)


@dataclass(frozen=True)
class Trellis:
    """Bit-shift trellis on a universal Gaussian LUT, per-row FP16 scale, LDLQ feedback between
    `g`-column groups; each group is cut into R-row tiles, one trellis sequence per tile."""
    V: int = 2
    kV: int = 3
    L: int = 12
    g: int = 16
    R: int = 64
    code = 9
    block = 128

    @property
    def name(self):
        return f"tcq{self.V}x{self.kV}l{self.L}" + (f"r{self.R}" if self.R != 64 else "")

    @property
    def bpw(self):
        return self.kV / self.V

    def tile_bytes(self):
        T = self.R * self.g // self.V
        return -(-(self.L + (T - 1) * self.kV) // 8)

    def nbytes(self, shape):
        rows, cols = shape
        return HEADER + 2 * rows + (cols // self.g) * (rows // self.R) * self.tile_bytes()

    def quantize(self, W, H, **_):
        B, rows, cols = W.shape
        lut = trellis_lut(self.V, self.kV, self.L, W.device)
        W0 = W.float()
        W = W0.clone()
        U = None if H is None else hessian_inverse(H)
        if U is not None:
            W.masked_fill_((H.diagonal(dim1=-2, dim2=-1) == 0)[:, None, :], 0)
        s = f16(W.square().mean(-1).sqrt())
        safe = torch.where(s > 0, s, 1.0)[..., None]
        states = []
        unit = torch.empty_like(W)
        for i1 in range(0, cols, self.block):
            i2 = min(cols, i1 + self.block)
            W1 = W[..., i1:i2].clone()
            Err = torch.zeros_like(W1)
            for j in range(0, i2 - i1, self.g):
                J = slice(i1 + j, i1 + j + self.g)
                st = viterbi(_tiles(W1[..., j:j + self.g] / safe, self.R, self.V), lut, self.L, self.kV)
                states.append(st.reshape(B, rows // self.R, -1))
                unit[..., J] = lut[st].reshape(B, rows, self.g)
                Q = unit[..., J] * s[..., None]
                if U is not None:  # codes are fixed per group, so the column-by-column LDLQ loop is one triangular solve
                    a, b = i1 + j, i1 + j + self.g
                    err = torch.linalg.solve_triangular(U[:, a:b, a:b], W1[..., j:j + self.g] - Q, upper=True, left=False)
                    W1[..., j + self.g:] -= err @ U[:, a:b, b:i2]
                    Err[..., j:j + self.g] = err
            if U is not None:
                W[..., i2:] -= Err @ U[:, i1:i2, i2:]
        Hf = None if H is None else H.float()
        weighted = unit if Hf is None else unit @ Hf
        num, den = (W0 * weighted).sum(-1), (unit * weighted).sum(-1)
        s = torch.where((den > 0) & (s > 0), f16(num / den.clamp_min(1e-30)), s)
        Q = unit * s[..., None]
        st = torch.stack(states, 1)  # [B, groups, tiles, T]
        first = (st[..., :1] >> torch.arange(self.L - 1, -1, -1, device=W.device)) & 1
        rest = (st[..., 1:, None] >> torch.arange(self.kV - 1, -1, -1, device=W.device)) & 1
        bits = torch.cat((first, rest.flatten(-2)), -1)
        bits = F.pad(bits, (0, -bits.shape[-1] % 8)).reshape(*bits.shape[:-1], -1, 8)
        packed = (bits << torch.arange(7, -1, -1, device=W.device)).sum(-1).to(torch.uint8)
        out = []
        for b in range(B):
            body = torch.cat((f16_bytes(s[b][:, None]).reshape(-1), packed[b].reshape(-1)))
            out.append(torch.cat((header(self.code, (self.V, self.kV, self.L, self.g, self.R // 16), W.device), body)).cpu())
        return Q, out

    @staticmethod
    def decode(raw, shape):
        rows, cols = shape
        V, kV, L, g, R = (int(v) for v in raw[1:6])
        R *= 16
        s = bytes_f16(raw[HEADER:HEADER + 2 * rows])
        T = R * g // V
        tb = -(-(L + (T - 1) * kV) // 8)
        packed = raw[HEADER + 2 * rows:].reshape(cols // g, rows // R, tb).long()
        bits = ((packed[..., None] >> torch.arange(7, -1, -1, device=raw.device)) & 1).flatten(-2)
        win = bits[..., :L + (T - 1) * kV].unfold(-1, L, kV)  # [groups, tiles, T, L]
        st = (win << torch.arange(L - 1, -1, -1, device=raw.device)).sum(-1)
        vals = trellis_lut(V, kV, L, raw.device)[st]  # [groups, tiles, T, V]
        unit = vals.reshape(cols // g, rows, g).permute(1, 0, 2).reshape(rows, cols)
        return unit * s[:, None]


# ---------------------------------------------------------------------------------------------
# Failed-baseline controls (Hessian-free).


class Control:
    def quantize(self, W, H, **_):
        payloads = [self.encode(w).cpu() for w in W]
        return torch.stack([self.decode(p.to(W.device), w.shape) for p, w in zip(payloads, W)]), payloads


class Binary256(Control):
    name, code, bpw = "binary256", 3, 1.25

    def nbytes(self, shape):
        rows, cols = shape
        return HEADER + rows * (cols // 256) * 40

    def encode(self, W):
        blocks = W.reshape(-1, 256)
        values = blocks.abs()
        first = values.argmax(1, keepdim=True)
        second = values.scatter(1, first, -1).argmax(1, keepdim=True)
        positions = torch.cat((first, second), dim=1)
        salient = values.gather(1, positions)
        kept = values.scatter(1, positions, 0).reshape(-1, 8, 32)
        counts = torch.ones_like(values).scatter_(1, positions, 0).reshape(-1, 8, 32).sum(-1)
        totals = kept.sum(-1)
        selected = totals / counts > totals.sum(1, keepdim=True) / 254
        low, high = [f16((totals * g).sum(1) / (counts * g).sum(1).clamp_min(1)) for g in (~selected, selected)]
        high = torch.where((low == 0) & (high == 0), f16(salient.mean(1) / 8), high)
        ratio = (16 * salient.mean(1) / high.maximum(low).clamp_min(1e-30)).round().clamp(0, 255)
        body = torch.cat((f16_bytes(torch.stack((low, high), 1)), pack_bits(selected, 1), positions.byte(),
                          ratio.byte()[:, None], pack_bits(blocks >= 0, 1)), 1)
        return torch.cat((header(self.code, (), body.device), body.reshape(-1)))

    @staticmethod
    def decode(raw, shape):
        raw = raw[HEADER:].reshape(-1, 40)
        low, high = bytes_f16(raw[:, :4]).unbind(1)
        selected = unpack_bits(raw[:, 4:5], 1, 8).bool()
        values = torch.where(selected, high[:, None], low[:, None]).repeat_interleave(32, 1)
        values.scatter_(1, raw[:, 5:7].long(), (high.maximum(low) * raw[:, 7] / 16)[:, None].expand(-1, 2))
        return torch.where(unpack_bits(raw[:, 8:], 1, 256).bool(), values, -values).reshape(shape)


class Scalar128(Control):
    name, code, bpw = "scalar128", 4, 2.125

    def nbytes(self, shape):
        rows, cols = shape
        return HEADER + rows * (cols // 128) * 34

    def encode(self, W):
        blocks = W.reshape(-1, 128)
        magnitude, sign = blocks.abs(), torch.where(blocks >= 0, 1, -1)
        scale = magnitude.mean(1, keepdim=True) * 0.6
        for _ in range(4):
            levels = 1 + 2 * (magnitude >= 2 * scale)
            scale = (magnitude * levels).sum(1, keepdim=True) / levels.square().sum(1, keepdim=True)
        scale = f16(scale)
        codes = ((1 + 2 * (magnitude >= 2 * scale)) * sign + 3) // 2
        body = torch.cat((f16_bytes(scale), pack_bits(codes, 2)), 1)
        return torch.cat((header(self.code, (), body.device), body.reshape(-1)))

    @staticmethod
    def decode(raw, shape):
        raw = raw[HEADER:].reshape(-1, 34)
        return ((unpack_bits(raw[:, 2:], 2, 128) * 2 - 3) * bytes_f16(raw[:, :2])).reshape(shape)


# ---------------------------------------------------------------------------------------------

class Zero(Control):
    """0 bpw: the projection is dropped (decodes to zeros), so the expert's output is zero."""
    name, code, bpw = "zero", 10, 0.0

    def nbytes(self, shape):
        return HEADER

    def encode(self, W):
        return header(self.code, (), W.device)

    @staticmethod
    def decode(raw, shape):
        return torch.zeros(shape, device=raw.device)


BY_CODE = {1: Scalar, 2: Sage, 3: Binary256, 4: Scalar128, 5: Split, 6: GGML, 7: Rotated, 8: VQ, 9: Trellis, 10: Zero}
CATALOG = [Scalar(2, 128), Scalar(2, 128, True), Scalar(2, 64, True), Scalar(3, 128), Scalar(3, 128, True), Scalar(4, 128), Sage(cap=0), Sage(cap=2), Sage(cap=4),
           Sage(cap=0, graph=False), Sage(cap=2, graph=False),
           Split(64), Split(128), Split(256), Split(128, Scalar(2, 64)), Split(256, Scalar(2, 64)),
           GGML("iq1_s"), GGML("iq1_m"), GGML("iq2_xxs"), GGML("tq1_0"),
           GGML("iq1_s", False), GGML("iq1_m", False), Binary256(), Scalar128(),
           *[VQ(8, b) for b in (8, 10, 12, 14, 16)], *[Rotated(VQ(8, b)) for b in (4, 6, 8, 10, 12, 14, 16)],
           Rotated(VQ(4, 8)), Rotated(VQ(4, 10)), Rotated(VQ(4, 12)), Rotated(VQ(8, 8, 2)), Rotated(VQ(8, 12, 1)),
           Rotated(VQ(8, 12, 2)), Rotated(VQ(8, 16, 2)),
           *[Rotated(VQ(8, b)) for b in (2, 3, 5, 7)],
           *[Rotated(Trellis(V, k, L, R=R)) for V, ks in ((2, range(1, 7)), (4, range(1, 13))) for k in ks
             for L in (12, 13, 14, 15, 16) for R in (64, 256)],
           Rotated(Scalar(2, 128, True)), Rotated(GGML("iq1_s")), Rotated(GGML("iq1_m")), Rotated(Sage(cap=2)), Zero()]
FORMATS = {f.name: f for f in CATALOG}


def quantize(fmt, W: torch.Tensor, H: torch.Tensor | None, **options):
    """Batched quantization of W [B, rows, cols] -> (dequantized W, list of B CPU payloads)."""
    return fmt.quantize(W, H, **options)


def decode(raw: torch.Tensor, shape) -> torch.Tensor:
    return BY_CODE[int(raw[0])].decode(raw, shape)


if __name__ == "__main__":
    # Self-check: payload sizes, decode == quantizer output, dead rows stay zero,
    # GSQ refinement never worsens a row.
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = (torch.randn(2, 64, 512) * torch.rand(2, 64, 1)).to(dev)
    W[0, 3] = 0  # a dead neuron
    X = (torch.randn(2, 600, 512) @ (torch.eye(512) + 0.1 * torch.randn(512, 512))).to(dev)
    H = X.transpose(1, 2) @ X / 600
    loss = lambda q: ((W - q) @ X.transpose(1, 2)).square().sum((-1, -2))
    for fmt in CATALOG:
        Q, raw = quantize(fmt, W, H)
        assert all(r.numel() == fmt.nbytes(w.shape) for r, w in zip(raw, W)), fmt.name
        torch.testing.assert_close(torch.stack([decode(r.to(dev), w.shape) for r, w in zip(raw, W)]), Q)
        assert Q[0, 3].abs().max() == 0 or isinstance(fmt, Control), fmt.name
        err = float(loss(Q).sum() / (W @ X.transpose(1, 2)).square().sum())
        print(f"{fmt.name:12s} {8 * raw[0].numel() / W[0].numel():.4f} bpw  rel output err {err:.4f}")
    gsq = Scalar(2, 128)
    Q, _ = quantize(gsq, W, H)
    R, raw = quantize(gsq, W, H, gsq_steps=50)
    assert (loss(R) <= loss(Q) * 1.0001).all()
    torch.testing.assert_close(torch.stack([decode(r.to(dev), w.shape) for r, w in zip(raw, W)]), R)
    print("gsq refine", loss(Q).tolist(), "->", loss(R).tolist())
