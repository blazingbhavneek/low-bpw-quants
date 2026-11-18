from __future__ import annotations

import json
import re
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


PROJECTIONS = ("gate", "up", "down")
QWEN_EXPERT = re.compile(
    r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight"
)
FP4_VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_nvfp4(packed: torch.Tensor, scale: torch.Tensor, outer: torch.Tensor,
                 fp4: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
        raise ValueError(f"expected U8 NVFP4 and E4M3FN scales, got {packed.dtype}, {scale.dtype}")
    *lead, rows, cols = packed.shape
    if scale.shape != (*lead, rows, cols // 8) or cols % 8 or outer.numel() not in (1, max(1, packed[..., 0, 0].numel())):
        raise ValueError("invalid NVFP4 weight or scale shape")
    outer = outer.float().reshape(*lead, 1, 1) if outer.numel() > 1 else outer.float().reshape(())
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(*lead, rows, -1).int()
    values = fp4[codes].reshape(*lead, rows, -1, 16)
    return (values * (scale.float() * outer).unsqueeze(-1)).reshape(*lead, rows, -1)


def decode_mxfp8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8:
        raise ValueError(f"expected E4M3FN MXFP8 and UE8M0 scales, got {values.dtype}, {scale.dtype}")
    rows, cols = values.shape
    if cols % 32:
        raise ValueError("invalid MXFP8 input width")
    if scale.shape == (rows // 32, cols // 32) and rows % 32 == 0:
        scale = scale.repeat_interleave(32, dim=0)
    if scale.shape != (rows, cols // 32):
        raise ValueError(f"invalid MXFP8 scale shape {scale.shape} for {values.shape}")
    factors = torch.exp2(scale.float() - 127).masked_fill(scale == 255, float("nan"))
    return (values.float().reshape(rows, -1, 32) * factors.unsqueeze(-1)).reshape(rows, cols)


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        config = read_json(path / "config.json")
        self.arch = config["model_type"]
        self.config = config.get("text_config", config)
        self.index = read_json(path / "model.safetensors.index.json")["weight_map"]
        self.layers = self.config["num_hidden_layers"]
        self.hidden = self.config["hidden_size"]
        self.gate_cap = self.up_cap = None
        self.inkling_formats = {}
        if self.arch == "qwen3_5_moe":
            self.experts = self.config["num_experts"]
            self.ffn = self.config["moe_intermediate_size"]
            self.qwen = {(int(m[1]), int(m[2]), m[3]): key.removesuffix(".weight")
                         for key in self.index if (m := QWEN_EXPERT.fullmatch(key))}
            for layer in range(self.layers):
                for expert in range(self.experts):
                    for proj in PROJECTIONS:
                        base = self.qwen.get((layer, expert, proj))
                        if base is None or not all(base + suffix in self.index for suffix in
                                                  (".weight_scale", ".weight_scale_2")):
                            raise ValueError(f"missing NVFP4 tensors for layer {layer}, expert {expert}, {proj}")
        elif self.arch in ("inkling2", "inkling2_text"):
            self.experts = self.config["n_routed_experts"]
            self.ffn = self.config["intermediate_size"]
            self.gate_cap = self.config.get("mlp_gate_softcap")
            self.up_cap = self.config.get("mlp_up_softcap")
            quant = read_json(path / "hf_quant_config.json")
            self.inkling_formats = {name: group["compute_dtype"]
                                    for group in quant["model"]["parameter_groups"]
                                    for name in group["tensors"]}
            for layer in range(self.layers):
                for projection in ("w13", "w2"):
                    base = self.inkling_name(layer, projection)
                    fmt = self.inkling_formats.get(base)
                    if fmt not in ("nvfp4", "mxfp8"):
                        raise ValueError(f"unsupported Inkling format {fmt!r} for {base}")
                    suffixes = ("", "_scale")
                    if fmt == "nvfp4":
                        suffixes += ("_amax", "_outer_scale_denominator")
                    if not all(base + suffix in self.index for suffix in suffixes):
                        raise ValueError(f"missing Inkling {fmt} tensors for {base}")
        else:
            raise ValueError(f"unsupported model type {self.arch!r}")

    def inkling_name(self, layer: int, projection: str) -> str:
        return f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"

    def expected_shapes(self) -> tuple[tuple[int, int], ...]:
        return ((self.ffn, self.hidden), (self.ffn, self.hidden), (self.hidden, self.ffn))

    def dense_stats(self) -> dict:
        """Parameters and stored bytes of the text model outside the routed experts (vision and MTP
        heads excluded); packed U8 weights hold two FP4 values per byte, scale tensors hold none."""
        prefixes = ("model.language_model.", "lm_head.") if self.arch == "qwen3_5_moe" else ("model.llm.",)
        params = nbytes = 0
        for shard in set(self.index.values()):
            with open(self.path / shard, "rb") as f:
                header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
            for name, info in header.items():
                if name == "__metadata__" or not name.startswith(prefixes) or ".mlp.experts." in name:
                    continue
                count = 1
                for d in info["shape"]:
                    count *= d
                nbytes += info["data_offsets"][1] - info["data_offsets"][0]
                if not re.search(r"(scale|scale_2|amax|denominator)$", name):
                    params += count * (2 if info["dtype"] == "U8" else 1)
        return {"dense_params": params, "dense_bytes": nbytes}

    def missing_shards(self) -> list[str]:
        return sorted(name for name in set(self.index.values()) if not (self.path / name).is_file())

    def open(self, device: str):
        return CheckpointReader(self, device)


class CheckpointReader:
    def __init__(self, source: Checkpoint, device: str):
        self.source = source
        self.device = torch.device(device)
        self.fp4 = torch.tensor(FP4_VALUES, dtype=torch.float32, device=device)
        self.stack = ExitStack()
        self.shards = {}
        self.current = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stack.close()

    def tensor(self, name: str, index=None, device=None) -> torch.Tensor:
        shard = self.source.index[name]
        if shard not in self.shards:
            self.shards[shard] = self.stack.enter_context(
                safe_open(self.source.path / shard, framework="pt", device="cpu")
            )
        handle = self.shards[shard]
        if index is None:
            tensor = handle.get_tensor(name)
        elif isinstance(index, slice):
            tensor = handle.get_slice(name)[index]
        else:
            tensor = torch.stack([handle.get_slice(name)[i] for i in index])
        return tensor.to(self.device if device is None else device)

    def expert(self, layer: int, expert: int) -> tuple[torch.Tensor, ...]:
        return tuple(w[0] for w in self.layer(layer, [expert]))

    def layer(self, layer: int, experts: list[int] | None = None, dtype=torch.float32) -> tuple[torch.Tensor, ...]:
        """Decoded (gate, up, down) stacks in `dtype` shaped [E, ffn, hidden], [E, ffn, hidden], [E, hidden, ffn]."""
        source = self.source
        experts = list(range(source.experts)) if experts is None else experts
        if layer != self.current:
            self.stack.close()
            self.shards.clear()
            self.current = layer
        result = []
        for start in range(0, len(experts), 32):  # bounded decode workspace
            chunk = experts[start:start + 32]
            if source.arch == "qwen3_5_moe":
                parts = []
                for proj in PROJECTIONS:
                    bases = [source.qwen[layer, e, proj] for e in chunk]
                    parts.append(decode_nvfp4(*(torch.stack([self.tensor(b + suffix, device="cpu") for b in bases]).to(self.device)
                                                for suffix in (".weight", ".weight_scale", ".weight_scale_2")), self.fp4))
            else:
                stacked = []
                for proj in ("w13", "w2"):
                    name = source.inkling_name(layer, proj)
                    index = slice(chunk[0], chunk[-1] + 1) if chunk == list(range(chunk[0], chunk[-1] + 1)) else chunk
                    values, scale = self.tensor(name, index), self.tensor(name + "_scale", index)
                    if source.inkling_formats[name] == "nvfp4":
                        amax = self.tensor(name + "_amax", index)
                        denominator = self.tensor(name + "_outer_scale_denominator")
                        if denominator.numel() != 1:
                            denominator = denominator.reshape(-1)[index]
                        stacked.append(decode_nvfp4(values, scale, amax.reshape(-1) / denominator.reshape(-1), self.fp4))
                    else:
                        stacked.append(torch.stack([decode_mxfp8(v, s) for v, s in zip(values, scale)]))
                parts = [stacked[0][:, 0::2], stacked[0][:, 1::2], stacked[1]]
            result.append([part.to(dtype) for part in parts])
        weights = tuple(torch.cat(p) for p in zip(*result))
        for matrix, shape in zip(weights, source.expected_shapes(), strict=True):
            if matrix.shape[1:] != shape:
                raise ValueError(f"layer {layer}: expected expert shape {shape}, got {tuple(matrix.shape[1:])}")
        return weights
