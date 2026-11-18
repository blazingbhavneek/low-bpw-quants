"""Text-only runtime: Transformers dense modules with pluggable routed experts.

Expert weights for a layer are decoded on demand as FP32 stacks and evicted when the next
layer is requested, so a 16 GB GPU can run the full model, a gradient pass, or a layer walk.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint import Checkpoint, decode_nvfp4


def down_input(x, gate_w, up_w, gate_cap=None, up_cap=None):
    return activate(F.linear(x, gate_w), F.linear(x, up_w), gate_cap, up_cap)


def activate(gate, up, gate_cap=None, up_cap=None):
    activation = gate_cap * (gate / gate_cap).tanh() * gate.sigmoid() if gate_cap else F.silu(gate)
    if up_cap:
        up = up_cap * (up / up_cap).tanh()
    return activation * up


def expert_output(x, weights, gate_cap=None, up_cap=None):
    return F.linear(down_input(x, weights[0], weights[1], gate_cap, up_cap), weights[2])


def route_groups(ids: torch.Tensor, experts: int):
    """Token assignments sorted by expert: (flat slot order, token index per slot, count per expert)."""
    flat = ids.reshape(-1)
    order = flat.argsort(stable=True)
    return order, order // ids.shape[-1], torch.bincount(flat, minlength=experts).tolist()


def moe_forward(x, ids, route, weights, caps):
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    order, tokens, counts = route_groups(ids, weights[0].shape[0])
    scale = route.reshape(-1)[order].float()
    start = 0
    for expert, n in enumerate(counts):
        if n:
            sel = tokens[start:start + n]
            y = expert_output(x[sel].float(), [w[expert].float() for w in weights], *caps)
            out.index_add_(0, sel, y * scale[start:start + n, None])
        start += n
    return out


class LayerCache:
    """Keeps the most recently decoded layer only."""

    def __init__(self, decode: Callable[[int], tuple[torch.Tensor, ...]]):
        self.decode, self.key, self.value = decode, None, None

    def __call__(self, layer: int):
        if layer != self.key:
            self.value = None
            self.value = self.decode(layer)
            self.key = layer
        return self.value


class Experts(nn.Module):
    def __init__(self, layer: int, load, caps, tap=None, grad_tap=None):
        super().__init__()
        self.layer, self.load, self.caps, self.tap, self.grad_tap = layer, load, caps, tap, grad_tap

    def forward(self, x, ids, route):
        if torch.is_grad_enabled() and (x.requires_grad or route.requires_grad):
            return GradExperts.apply(x, ids, route, self).to(x.dtype)
        if self.tap:
            self.tap(self.layer, x, ids, route)
        return moe_forward(x, ids, route, self.load(self.layer), self.caps).to(x.dtype)


class GradExperts(torch.autograd.Function):
    """Recomputes expert outputs in backward so decoded weights are never kept for autograd."""

    @staticmethod
    def forward(ctx, x, ids, route, module):
        ctx.save_for_backward(x, ids, route)
        ctx.module = module
        if module.tap:
            module.tap(module.layer, x, ids, route)
        return moe_forward(x, ids, route, module.load(module.layer), module.caps)

    @staticmethod
    def backward(ctx, grad):
        x, ids, route = ctx.saved_tensors
        module = ctx.module
        weights = module.load(module.layer)
        grad_x = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        grad_route = torch.zeros_like(route).reshape(-1)
        order, tokens, counts = route_groups(ids, weights[0].shape[0])
        flat_route = route.reshape(-1)
        start = 0
        for expert, n in enumerate(counts):
            if not n:
                continue
            slots, sel = order[start:start + n], tokens[start:start + n]
            start += n
            with torch.enable_grad():
                xi = x[sel].detach().float().requires_grad_()
                y = expert_output(xi, [w[expert].float() for w in weights], *module.caps)
            g = grad[sel].float()
            dy = g * flat_route[slots].float()[:, None]  # dL/d(expert output)
            dx, = torch.autograd.grad(y, xi, dy)
            if module.grad_tap:
                module.grad_tap(module.layer, expert, dy)
            grad_x.index_add_(0, sel, dx)
            grad_route[slots] = (g * y.detach()).sum(-1).to(route.dtype)
        return grad_x.to(x.dtype), None, grad_route.reshape(route.shape), None


def dense_key_map(source: Checkpoint) -> dict[str, str]:
    if source.arch == "qwen3_5_moe":
        return {name.replace("model.language_model.", "model.", 1): name
                for name in source.index if not name.startswith(("model.visual.", "mtp."))}

    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    transforms = get_checkpoint_conversion_mapping("inkling2_text")
    if not transforms:
        raise ValueError("Transformers does not provide the Inkling 2 text checkpoint mapping")
    renamings = [change for change in transforms if isinstance(change, WeightRenaming)]
    mapped = {}
    for source_name in source.index:
        if not source_name.startswith("model.llm.") or ".mlp.experts." in source_name:
            continue
        name = source_name
        for renaming in renamings:
            name, _ = renaming.rename_source_key(name)
        if name in mapped:
            raise ValueError(f"duplicate Inkling 2 mapping for {name}")
        mapped[name] = source_name
    return mapped


def load_text_model(source: Checkpoint, device: str, load, *, dense_device: str | None = None,
                    tap=None, grad_tap=None):
    """Dense weights from the checkpoint; routed experts from `load(layer) -> (gate, up, down)` stacks."""
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise ValueError("install the reference extra for the text runtime") from error

    config = AutoConfig.from_pretrained(source.path, local_files_only=True)
    config = getattr(config, "text_config", config)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa", experts_implementation="eager")
    caps = (source.gate_cap, source.up_cap)
    for layer, block in enumerate(model.model.layers):
        block.mlp.experts = Experts(layer, load, caps, tap, grad_tap)
    dense_device = dense_device or device
    names = dense_key_map(source)
    with source.open(dense_device) as dense:
        tensors = [(name, param, True) for name, param in model.named_parameters()]
        tensors += [(name, buffer, False) for name, buffer in model.named_buffers()]
        for name, param, is_parameter in tensors:
            parent, _, leaf = name.rpartition(".")
            module = model.get_submodule(parent)
            source_name = names.get(name)
            if source_name is None and not is_parameter:
                setattr(module, leaf, param.to(dense_device))
                continue
            if source_name is None and name == "lm_head.weight" and config.tie_word_embeddings:
                source_name = names["model.embed_tokens.weight"]
            if source_name is None:
                raise ValueError(f"missing checkpoint parameter {name}")
            weight = dense.tensor(source_name)
            if weight.dtype == torch.uint8:
                base = source_name.removesuffix(".weight")
                weight = decode_nvfp4(weight, dense.tensor(base + ".weight_scale"),
                                      dense.tensor(base + ".weight_scale_2"), dense.fp4.to(dense_device)).bfloat16()
            elif weight.dtype == torch.float8_e4m3fn:
                scale = dense.tensor(source_name.removesuffix(".weight") + ".weight_scale")
                if scale.numel() == 1:
                    weight = weight.float() * scale.float()
                elif scale.numel() == weight.shape[0]:
                    weight = weight.float() * scale.float().reshape(-1, 1)
                else:
                    raise ValueError(f"unsupported FP8 weight scale {scale.shape} for {source_name}")
                weight = weight.bfloat16()
            if weight.shape != param.shape:
                raise ValueError(f"{source_name}: expected {param.shape}, got {weight.shape}")
            setattr(module, leaf, nn.Parameter(weight, requires_grad=False) if is_parameter else weight)
    model.model.rotary_emb.to(device)
    model.model.norm.to(device)
    return model.eval()


class _Capture(nn.Module):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def forward(self, hidden_states, *args, **kwargs):
        self.store.append((args, kwargs))
        return hidden_states


def layer_kwargs(model, embeds: torch.Tensor) -> list:
    """Arguments the text model passes to each decoder layer for inputs shaped like `embeds`."""
    layers, store = model.model.layers, []
    model.model.layers = nn.ModuleList(_Capture(store) for _ in layers)
    try:
        model.model(inputs_embeds=embeds, use_cache=False)
    finally:
        model.model.layers = layers
    return store


@torch.no_grad()
def walk(model, sequences: list[torch.Tensor], device: str, on_layer=None, before=None, redo: bool = False):
    """Run decoder layers one at a time over all sequences, keeping hidden states on `device`.

    The experts module's tap sees every routed batch of the current layer; `on_layer(layer)` runs
    after the layer has processed every sequence, before its weights are evicted. With `redo`, a
    truthy `on_layer` result recomputes the layer's outputs (its weights changed) from its inputs.
    """
    embed = model.model.embed_tokens
    hidden = [embed(ids.to(embed.weight.device)).to(device) for ids in sequences]
    shapes = {}
    for layer, block in enumerate(model.model.layers):
        block.to(device)
        inputs = list(hidden) if redo else []
        for i, h in enumerate(hidden):
            if before:
                before(layer, i)
            if h.shape not in shapes:
                shapes[h.shape] = layer_kwargs(model, h)
            args, kwargs = shapes[h.shape][layer]
            hidden[i] = block(h, *args, **kwargs)
        if on_layer and on_layer(layer) and redo:
            for i, h in enumerate(inputs):
                args, kwargs = shapes[h.shape][layer]
                hidden[i] = block(h, *args, **kwargs)
        del inputs
        block.to(model.lm_head.weight.device)
    return hidden
