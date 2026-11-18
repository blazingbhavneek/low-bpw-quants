"""End-to-end KL distillation of a rotated-VQ bank's FP16 row scales; codes, codebooks and dense
weights stay fixed, so the tuned bank has the same bytes and layout.

A projection decodes as W = (C[idx] * s) R with R = H_n diag(signs) / sqrt(n), so the student
computes y = s * (C[idx] (R x)) -- rotating activations instead of weights -- and trains log s
against the decoded-weight teacher's full next-token distribution on the calibration tokens.
"""

from __future__ import annotations

import functools
import math
import random
import shutil
from pathlib import Path
from time import monotonic

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import codecs
from .bank import ExpertBank
from .runtime import LayerCache, activate, load_text_model, route_groups, walk

BODY = 2 * codecs.HEADER  # rotation header + VQ header


class Layout:
    """Byte positions of every row scale and the decode groups of a rotated-VQ v2 bank."""

    def __init__(self, bank: ExpertBank):
        if bank.version != 2:
            raise ValueError("scale tuning needs a version 2 bank")
        self.bank, E = bank, bank.experts
        self.groups, self.scale_at = {}, {}  # (layer, p) -> [(dim, bits, row_bytes, experts, starts)], [E, rows]
        for l in bank.layers:
            raw = np.asarray(bank.tensors[f"blk.{l}.routed_experts"]).view(np.uint8)
            for p, (rows, cols) in enumerate(bank.shapes):
                starts = bank.offsets[l][3 * np.arange(E) + p]
                if not ((raw[starts] == codecs.Rotated.code) & (raw[starts + 1] == 0)
                        & (raw[starts + codecs.HEADER] == codecs.VQ.code)).all():
                    raise ValueError(f"layer {l}: scale tuning needs rotated VQ (seed 0) payloads")
                dims, bits = raw[starts + codecs.HEADER + 1], raw[starts + codecs.HEADER + 2]
                groups, at = [], np.empty((E, rows), np.int64)
                for d, b in sorted(set(zip(dims.tolist(), bits.tolist()))):
                    experts = np.flatnonzero((dims == d) & (bits == b))
                    rb = codecs.VQ(d, b).row_bytes(cols)
                    groups.append((d, b, rb, experts, starts[experts]))
                    at[experts] = starts[experts, None] + BODY + rb * np.arange(rows)
                self.groups[l, p], self.scale_at[l, p] = groups, at

    def scales(self, l: int, p: int) -> np.ndarray:
        raw = np.asarray(self.bank.tensors[f"blk.{l}.routed_experts"]).view(np.uint8)
        at = self.scale_at[l, p]
        return (raw[at].astype(np.uint16) | raw[at + 1].astype(np.uint16) << 8).view(np.float16).astype(np.float32)

    def units(self, l: int, device) -> tuple[torch.Tensor, ...]:
        """Scale-free codeword stacks C[idx] in the rotated basis, BF16 [E, rows, cols] per projection."""
        raw = torch.from_numpy(np.array(self.bank.tensors[f"blk.{l}.routed_experts"]).view(np.uint8)).to(device)
        out = []
        for p, (rows, cols) in enumerate(self.bank.shapes):
            U = torch.empty(self.bank.experts, rows, cols, dtype=torch.bfloat16, device=device)
            for d, b, rb, experts, starts in self.groups[l, p]:
                C = codecs.codebook(d, b, device).bfloat16()
                for i in range(0, len(experts), 16):
                    body = torch.stack([raw[s + BODY:s + BODY + rows * rb] for s in starts[i:i + 16]])
                    idx = codecs.unpack_bits(body.reshape(-1, rows, rb)[..., 2:], b, cols // d)
                    U[torch.as_tensor(experts[i:i + 16], device=device)] = C[idx].reshape(-1, rows, cols)
            out.append(U)
        return tuple(out)

    def write(self, output: Path, scales: dict):
        """Copy the bank and overwrite its FP16 row scales in place."""
        shutil.copyfile(self.bank.path, output)
        offsets = {t.name: int(t.data_offset) for t in gguf.GGUFReader(output).tensors}
        mm = np.memmap(output, dtype=np.uint8, mode="r+")
        for (l, p), s in scales.items():
            at = offsets[f"blk.{l}.routed_experts"] + self.scale_at[l, p]
            bits = s.astype(np.float16).view(np.uint16)
            mm[at], mm[at + 1] = (bits & 0xFF).astype(np.uint8), (bits >> 8).astype(np.uint8)
        mm.flush()


def rotate(x: torch.Tensor) -> torch.Tensor:
    """x R^T for the seed-0 randomized Hadamard R of size x.shape[-1]."""
    n = x.shape[-1]
    R = codecs.rotation(n, 0, str(x.device))
    return x @ R.T


class TunedExperts(nn.Module):
    def __init__(self, layer: int, units, scales, caps, per_row: bool = False):
        super().__init__()
        self.layer, self.units, self.caps = layer, units, caps
        self.base = [torch.as_tensor(s) for s in scales]  # FP32 [E, rows] per projection
        # per_row: one log-scale per row; else one per (expert, projection) -- far better gradient SNR
        self.log_s = nn.ParameterList(nn.Parameter(torch.zeros_like(s if per_row else s[:, :1])) for s in self.base)

    def scales(self):
        return [b.to(t.device) * t.exp() for b, t in zip(self.base, self.log_s)]

    def forward(self, x, ids, route):
        Ug, Uu, Ud = self.units(self.layer)
        sg, su, sd = self.scales()
        xr = rotate(x.float()).bfloat16()  # gate and up share the seed-0 rotation
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        order, tokens, counts = route_groups(ids, Ug.shape[0])
        weight = route.reshape(-1)[order].float()
        start = 0
        for e, n in enumerate(counts):
            if not n:
                continue
            sel = tokens[start:start + n]
            g = F.linear(xr[sel], Ug[e]).float() * sg[e]
            u = F.linear(xr[sel], Uu[e]).float() * su[e]
            h = rotate(activate(g, u, *self.caps)).bfloat16()
            y = F.linear(h, Ud[e]).float() * sd[e]
            out.index_add_(0, sel, y * weight[start:start + n, None])
            start += n
        return out.to(x.dtype)


def kl_backward(model, hidden: torch.Tensor, teacher: torch.Tensor, chunk: int = 256) -> float:
    """Full-vocabulary KL(teacher || student) summed over positions; backward through `hidden`."""
    h = hidden.detach().reshape(-1, hidden.shape[-1]).requires_grad_()
    t = teacher.reshape(-1, teacher.shape[-1])
    total = 0.0
    for i in range(0, len(h), chunk):
        lt = model.lm_head(t[i:i + chunk].to(h.device)).float().log_softmax(-1)
        ls = model.lm_head(h[i:i + chunk]).float().log_softmax(-1)
        kl = (lt.exp() * (lt - ls)).sum()
        kl.backward()
        total += float(kl)
    hidden.backward(h.grad.reshape(hidden.shape))
    return total


@torch.no_grad()
def kl_eval(model, ids_list, teacher_list, device) -> float:
    total = n = 0
    for ids, teacher in zip(ids_list, teacher_list):
        h = model.model(input_ids=ids.to(device), use_cache=False).last_hidden_state[:, :-1]
        for hs, ht in zip(h.reshape(-1, h.shape[-1]).split(256), teacher[:, :-1].reshape(-1, h.shape[-1]).split(256)):
            lt = model.lm_head(ht.to(device)).float().log_softmax(-1)
            ls = model.lm_head(hs).float().log_softmax(-1)
            total += float((lt.exp() * (lt - ls)).sum())
        n += h.shape[0] * h.shape[1]
    return total / n


def tune(source, bank_path: Path, tokens: dict, output: Path, device: str, *, epochs: int, lr: float,
         batch: int, holdout: int, per_row: bool = False) -> dict:
    bank = ExpertBank(bank_path)
    bank.check(source)
    layout = Layout(bank)
    train = [r[:-holdout] for r in tokens.values()]
    val = torch.cat([r[-holdout:] for r in tokens.values()]).split(batch)
    rows = torch.cat(train)
    with source.open(device) as reader:
        model = load_text_model(source, device, LayerCache(reader.layer))
        started = monotonic()
        with torch.no_grad():  # teacher final hidden states (decoded source experts), CPU BF16
            seqs = list(rows.split(batch)) + list(val)
            hidden = walk(model, seqs, device)
            teacher = [model.model.norm(h).cpu() for h in hidden]
            del hidden
        torch.cuda.empty_cache()
        teacher_train, teacher_val = teacher[:len(seqs) - len(val)], teacher[len(seqs) - len(val):]
        print(f"teacher states {monotonic() - started:.0f}s", flush=True)
        units = LayerCache(lambda l: layout.units(l, device))
        caps = (source.gate_cap, source.up_cap)
        tuned = []
        for l, block in enumerate(model.model.layers):
            block.mlp.experts = TunedExperts(l, units, [torch.from_numpy(layout.scales(l, p)).to(device)
                                                         for p in range(3)], caps, per_row).to(device)
            tuned.append(block.mlp.experts)
        report = {"val_kl": [kl_eval(model, val, teacher_val, device)]}
        print(f"val KL before {report['val_kl'][0]:.4f}", flush=True)
        for block in model.model.layers:
            block.forward = functools.partial(checkpoint, block.forward, use_reentrant=False)
        params = [p for m in tuned for p in m.log_s]
        opt = torch.optim.Adam(params, lr=lr)
        order = list(range(len(rows)))
        for epoch in range(epochs):
            random.Random(epoch).shuffle(order)
            for step, i in enumerate(range(0, len(order), batch)):
                pick = order[i:i + batch]
                ids = rows[pick].to(device)
                t = torch.cat([teacher_train[j // batch][j % batch:j % batch + 1] for j in pick])
                hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
                kl = kl_backward(model, hidden, t[:, :-1]) / hidden[..., 0].numel()
                for p in params:
                    if p.grad is not None:
                        p.grad /= hidden[..., 0].numel()
                opt.step()
                opt.zero_grad(set_to_none=True)
                print(f"epoch {epoch} step {step} kl {kl:.4f} ({monotonic() - started:.0f}s)", flush=True)
                if (step + 1) % 20 == 0 or i + batch >= len(order):
                    report["val_kl"].append(kl_eval(model, val, teacher_val, device))
                    print(f"val KL epoch {epoch} step {step} {report['val_kl'][-1]:.4f}", flush=True)
        with torch.no_grad():
            layout.write(output, {(l, p): s.cpu().numpy() for l, m in enumerate(tuned) for p, s in enumerate(m.scales())})
    report.update({"path": str(output), "epochs": epochs, "lr": lr, "per_row": per_row})
    return report
