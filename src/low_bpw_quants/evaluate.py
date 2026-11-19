"""WikiText-2 KL / top-1 / perplexity benchmark against the decoded-weight teacher.

32 evenly spaced, nonoverlapping 512-token windows of the raw test split; the last 256
next-token predictions of each window are scored (8,192 predictions).
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WINDOWS, LENGTH, SCORED = 32, 512, 256


def windows(tokenizer, parquet: Path) -> torch.Tensor:
    import pyarrow.parquet as pq

    text = "\n\n".join(pq.read_table(parquet).column("text").to_pylist())
    ids = torch.tensor(tokenizer(text)["input_ids"])
    starts = np.linspace(0, len(ids) - LENGTH, WINDOWS).astype(int)
    if (np.diff(starts) < LENGTH).any():
        raise ValueError("test split too short for nonoverlapping windows")
    return torch.stack([ids[s:s + LENGTH] for s in starts])


@torch.no_grad()
def final_hidden(model, tokens: torch.Tensor, device: str, batch: int = 8) -> torch.Tensor:
    """Final-norm hidden states at the scored positions, [windows, SCORED, hidden] in model dtype."""
    out = []
    for start in range(0, len(tokens), batch):
        ids = tokens[start:start + batch].to(device)
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        out.append(hidden[:, LENGTH - SCORED - 1:LENGTH - 1].cpu())
    return torch.cat(out)


@torch.no_grad()
def compare(model, teacher: torch.Tensor, student: torch.Tensor, tokens: torch.Tensor) -> dict:
    head = model.lm_head
    targets = tokens[:, LENGTH - SCORED:]
    kl = nll_t = nll_s = 0.0
    agree = 0
    for t, s, y in zip(teacher, student, targets):
        lt = head(t.to(head.weight.device)).double().log_softmax(-1)
        ls = head(s.to(head.weight.device)).double().log_softmax(-1)
        y = y.to(lt.device)
        kl += float((lt.exp() * (lt - ls)).sum())
        agree += int((lt.argmax(-1) == ls.argmax(-1)).sum())
        nll_t -= float(lt.gather(1, y[:, None]).sum())
        nll_s -= float(ls.gather(1, y[:, None]).sum())
    n = targets.numel()
    return {"predictions": n, "kl": kl / n, "top1": agree / n,
            "teacher_ppl": float(np.exp(nll_t / n)), "ppl": float(np.exp(nll_s / n))}


def mask_router(gate, dropped: torch.Tensor):
    """Top-k router that never selects `dropped` experts (bool [E]); weights renormalize over the rest."""
    def forward(self, hidden_states):
        logits = F.linear(hidden_states.reshape(-1, self.hidden_dim), self.weight)
        probs = torch.softmax(logits.float().masked_fill(dropped, float("-inf")), dim=-1)
        top, idx = torch.topk(probs, self.top_k, dim=-1)
        return logits, (top / top.sum(-1, keepdim=True)).to(logits.dtype), idx
    gate.forward = types.MethodType(forward, gate)


def run(source, bank_path: Path | None, parquet: Path, device: str, teacher_cache: Path,
        output: Path | None = None) -> dict:
    from transformers import AutoTokenizer

    from .runtime import LayerCache, load_text_model

    tokenizer = AutoTokenizer.from_pretrained(source.path, local_files_only=True)
    tokens = windows(tokenizer, parquet)
    with source.open(device) as reader:
        model = load_text_model(source, device, LayerCache(reader.layer))
        if teacher_cache.is_file():
            cached = torch.load(teacher_cache)
            if not torch.equal(cached["tokens"], tokens):
                raise ValueError(f"{teacher_cache} was computed for different windows")
            teacher = cached["hidden"]
        else:
            from .runtime import DENSE_BITS
            if DENSE_BITS:
                raise ValueError("build the teacher cache without LOW_BPW_DENSE_BITS")
            teacher = final_hidden(model, tokens, device)
            torch.save({"tokens": tokens, "hidden": teacher}, teacher_cache)
        from .runtime import DENSE_BITS
        if bank_path is None and not DENSE_BITS:
            student = teacher
        elif bank_path is None:
            student = final_hidden(model, tokens, device)  # dense-only quantization, original experts
        else:
            from .bank import ExpertBank

            bank = ExpertBank(bank_path)
            bank.check(source)
            load = LayerCache(lambda layer: bank.layer(layer, device))
            for block in model.model.layers:
                block.mlp.experts.load = load
            if os.environ.get("LOW_BPW_MASK_DROPPED") and "zero" in bank.formats:
                z = bank.formats.index("zero")  # fully dropped experts leave top-k: tokens go to the next expert
                for layer, block in enumerate(model.model.layers):
                    mask_router(block.mlp.gate, torch.from_numpy((bank.choice[layer] == z).all(-1)).to(device))
            student = final_hidden(model, tokens, device)
        metrics = compare(model, teacher, student, tokens)
    metrics["bank"] = str(bank_path) if bank_path else None
    if output:
        output.write_text(json.dumps(metrics, indent=2))
    return metrics
