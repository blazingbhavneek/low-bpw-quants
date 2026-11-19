"""Routed-expert GGUF: one int8 tensor of self-describing codec payloads per layer.

Expert e of a layer stores gate, up, down payloads back to back; byte offsets per
(layer, expert, projection) and the format table are GGUF metadata. VQ codebooks are stored once
as FP16 tensors. Version 1 banks (header-less binary256 / scalar128 payloads) are still readable.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from time import monotonic

import gguf
import numpy as np
import torch

from . import codecs

ARCH = "low_bpw_moe"


def _codebooks(formats) -> dict[str, torch.Tensor]:
    """GGUF tensor name -> FP16 codebook / trellis LUT used by `formats` (banks are self-contained)."""
    found = {}
    for fmt in formats:
        while isinstance(fmt, codecs.Rotated):
            fmt = fmt.inner
        if isinstance(fmt, codecs.VQ):
            found[f"vq.{fmt.dim}x{fmt.bits}"] = codecs.codebook(fmt.dim, fmt.bits, "cpu")
        if isinstance(fmt, codecs.Trellis):
            found[f"tcq.{fmt.V}x{fmt.kV}l{fmt.L}"] = codecs.trellis_lut(fmt.V, fmt.kV, fmt.L, "cpu")
    return dict(sorted(found.items()))


def refined(fmt, passes: int):
    """The same format with `passes` VQ coordinate-descent passes (payload layout unchanged)."""
    if isinstance(fmt, codecs.Rotated):
        return dataclasses.replace(fmt, inner=refined(fmt.inner, passes))
    return dataclasses.replace(fmt, passes=passes) if isinstance(fmt, codecs.VQ) else fmt


def quantize(source, tokens: dict, allocation: Path, output: Path, device: str, *, hessian_seqs: int,
             batch: int, expert_batch: int, prior: float = 2.0, gsq_steps: int = 0, passes: int = 0,
             sequential: bool = False) -> dict:
    from .score import run

    with np.load(allocation) as data:
        choice = data["choice"]
        formats = [refined(codecs.FORMATS[str(n)], passes) for n in data["formats"]]
    names = [f.name for f in formats]
    shapes = source.expected_shapes()
    L, E = source.layers, source.experts
    sizes = np.array([[f.nbytes(s) for f in formats] for s in shapes])  # [3, K]
    nbytes = sizes[np.arange(3), choice]  # [L, E, 3]
    offsets = np.concatenate((np.zeros((L, 1), np.int64), nbytes.reshape(L, -1).cumsum(1)), 1)
    temporary = output.with_name(output.name + ".partial")
    writer = gguf.GGUFWriter(temporary, ARCH)
    writer.add_uint32(f"{ARCH}.version", 2)
    writer.add_string(f"{ARCH}.source_arch", source.arch)
    writer.add_uint32(f"{ARCH}.hidden_size", source.hidden)
    writer.add_uint32(f"{ARCH}.intermediate_size", source.ffn)
    writer.add_uint32(f"{ARCH}.experts_per_layer", E)
    writer.add_uint32(f"{ARCH}.layers", L)
    writer.add_float32(f"{ARCH}.gate_cap", source.gate_cap or 0.0)
    writer.add_float32(f"{ARCH}.up_cap", source.up_cap or 0.0)
    writer.add_array(f"{ARCH}.formats", names)
    writer.add_array(f"{ARCH}.assignments", choice.reshape(-1).tolist())
    writer.add_array(f"{ARCH}.format_bytes", sizes.reshape(-1).tolist())  # [projection, format]
    books = _codebooks(formats)
    for name, book in books.items():
        writer.add_tensor_info(name, tuple(book.shape), np.dtype(np.float16), book.numel() * 2)
    for layer in range(L):
        writer.add_tensor_info(f"blk.{layer}.routed_experts", (int(offsets[layer, -1]),), np.dtype(np.int8),
                               int(offsets[layer, -1]))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for book in books.values():
        writer.write_tensor_data(book.half().numpy())
    started = monotonic()

    def on_layer(stats, layer):
        weights = stats.load(layer)
        payloads = [[None] * 3 for _ in range(E)]
        for p, w in enumerate(weights):
            for k, fmt in enumerate(formats):
                experts = np.flatnonzero(choice[layer, :, p] == k)
                for start in range(0, len(experts), expert_batch):
                    idx = torch.as_tensor(experts[start:start + expert_batch], device=w.device)
                    h = stats.hessians(idx, prior)[0 if p < 2 else 1]
                    _, raw = codecs.quantize(fmt, w[idx], h, gsq_steps=gsq_steps, seed=layer * E + p)
                    for e, r in zip(idx.tolist(), raw):
                        payloads[e][p] = r.numpy()
        flat = np.concatenate([r for per in payloads for r in per]).view(np.int8)
        if len(flat) != offsets[layer, -1]:
            raise ValueError(f"layer {layer}: payload bytes {len(flat)} != planned {offsets[layer, -1]}")
        writer.write_tensor_data(flat)
        print(f"wrote layer {layer} ({monotonic() - started:.0f}s)", flush=True)
        if sequential:  # later layers see this layer's quantized outputs
            return tuple(torch.stack([codecs.decode(torch.from_numpy(payloads[e][p]).to(weights[p].device), shape)
                                      .bfloat16() for e in range(E)]) for p, shape in enumerate(shapes))

    try:
        run(source, tokens, output.parent, device, [], hessian_seqs=hessian_seqs, replay_seqs=0, replay_cap=0,
            batch=batch, expert_batch=expert_batch, on_layer=on_layer, sequential=sequential)
    finally:
        writer.close()
    os.replace(temporary, output)
    routed = L * E * sum(r * c for r, c in shapes)
    size = output.stat().st_size
    return {"path": str(output), "bytes": size, "routed_weights": routed, "routed_bpw": size * 8 / routed,
            "payload_bpw": float(offsets[:, -1].sum() * 8 / routed)}


class ExpertBank:
    def __init__(self, path: Path):
        self.path, reader = path, gguf.GGUFReader(path)
        field = lambda name: reader.get_field(f"{ARCH}.{name}").contents()
        if reader.get_field("general.architecture").contents() != ARCH:
            raise ValueError(f"{path} is not a {ARCH} GGUF")
        self.version = field("version")
        self.source_arch = field("source_arch")
        self.hidden, self.ffn = field("hidden_size"), field("intermediate_size")
        self.experts = field("experts_per_layer")
        self.shapes = ((self.ffn, self.hidden), (self.ffn, self.hidden), (self.hidden, self.ffn))
        self.tensors = {t.name: t.data for t in reader.tensors}
        if self.version == 1:  # failed baseline: per-expert codec 0 = binary256, 1 = scalar128
            layers = field("layers")
            choice = np.asarray(field("assignments")).reshape(len(layers), self.experts)
            controls = (codecs.Binary256(), codecs.Scalar128())
            per = np.array([[c.nbytes(s) - codecs.HEADER for s in self.shapes] for c in controls])
            self.layers = layers
            self.codes = {l: np.repeat(np.array([c.code for c in controls])[ch], 3).reshape(-1, 3)
                          for l, ch in zip(layers, choice)}
            self.offsets = {l: np.r_[0, per[ch].reshape(-1).cumsum()] for l, ch in zip(layers, choice)}
        elif self.version == 2:
            self.layers = list(range(field("layers")))
            names = list(field("formats"))
            choice = np.asarray(field("assignments")).reshape(len(self.layers), self.experts, 3)
            sizes = np.asarray(field("format_bytes")).reshape(3, len(names))
            self.offsets = {l: np.r_[0, sizes[np.arange(3), choice[l]].reshape(-1).cumsum()] for l in self.layers}
            self.formats, self.choice = names, choice
            for name, data in self.tensors.items():
                if name.startswith("vq."):
                    dim, bits = map(int, name[3:].split("x"))
                    codecs.CODEBOOKS[(dim, bits)] = torch.from_numpy(np.array(data, dtype=np.float16))
                if name.startswith("tcq."):
                    V, rest = name[4:].split("x")
                    kV, L = map(int, rest.split("l"))
                    codecs.CODEBOOKS[("tcq", int(V), kV, L)] = torch.from_numpy(
                        np.array(data, dtype=np.float16)).reshape(1 << L, int(V))
        else:
            raise ValueError(f"unsupported bank version {self.version}")
        for l in self.layers:
            if self.tensors[f"blk.{l}.routed_experts"].shape != (self.offsets[l][-1],):
                raise ValueError(f"layer {l}: tensor length does not match its offsets")

    def check(self, source):
        if (self.source_arch, self.hidden, self.ffn, self.experts) != (source.arch, source.hidden, source.ffn,
                                                                       source.experts):
            raise ValueError("bank does not match the checkpoint")

    def layer(self, layer: int, device) -> tuple[torch.Tensor, ...]:
        """Decoded FP32 (gate, up, down) stacks for one layer."""
        raw = torch.from_numpy(np.array(self.tensors[f"blk.{layer}.routed_experts"]).view(np.uint8)).to(device)
        offsets = self.offsets[layer]
        out = [torch.empty(self.experts, *s, device=device) for s in self.shapes]
        for e in range(self.experts):
            for p, shape in enumerate(self.shapes):
                payload = raw[offsets[3 * e + p]:offsets[3 * e + p + 1]]
                if self.version == 1:
                    payload = torch.cat((codecs.header(int(self.codes[layer][e, p]), (), device), payload))
                out[p][e] = codecs.decode(payload, shape)
        return tuple(out)
