"""Calibration token sets and the NatScale gradient / exposure pass."""

from __future__ import annotations

import functools
import gzip
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# NatScale's chat sources plus C4 raw text; each source is one calibration fold.
SOURCES = ("ultrachat", "magpie", "writingprompts", "c4")


def _documents(directory: Path, source: str):
    import pyarrow.parquet as pq

    if source == "c4":
        with gzip.open(directory / "c4.json.gz", "rt", encoding="utf-8") as lines:
            for line in lines:
                yield json.loads(line)["text"]
        return
    table = pq.read_table(directory / f"{source}.parquet")
    for row in table.to_pylist():
        if source == "ultrachat":
            yield [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        elif source == "magpie":
            roles = {"human": "user", "gpt": "assistant", "system": "system"}
            yield [{"role": roles[m["from"]], "content": m["value"]} for m in row["conversations"]]
        else:
            yield [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["story"]}]


def build(tokenizer, directory: Path, sequences: int, length: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Pack shuffled documents from each source into `sequences` rows of `length` tokens."""
    result = {}
    for source in SOURCES:
        docs = list(_documents(directory, source))
        random.Random(seed).shuffle(docs)
        stream, rows = [], []
        for doc in docs:
            text = doc if isinstance(doc, str) else tokenizer.apply_chat_template(doc, tokenize=False)
            stream += tokenizer(text)["input_ids"] + [tokenizer.eos_token_id]
            while len(stream) >= length and len(rows) < sequences:
                rows.append(stream[:length])
                stream = stream[length:]
            if len(rows) == sequences:
                break
        if len(rows) < sequences:
            raise ValueError(f"{source}: only {len(rows)} sequences of {length} tokens")
        result[source] = torch.tensor(rows)
    return result


def gradients(source, tokens: dict[str, torch.Tensor], per_source: int, device: str, batch: int = 4) -> dict:
    """Per-expert summed squared CE gradients at expert outputs and routed event counts (NatScale Eq. 2)."""
    from .runtime import LayerCache, load_text_model

    shape = (source.layers, source.experts)
    energy, events = np.zeros(shape), np.zeros((len(tokens), *shape), dtype=np.int64)
    fold = [0]

    def grad_tap(layer, expert, dy):  # once per routed (layer, expert) group, so events count here too
        energy[layer, expert] += float(dy.double().square().sum())
        events[fold[0], layer, expert] += len(dy)

    with source.open(device) as reader:
        load = LayerCache(lambda layer: reader.layer(layer, dtype=torch.bfloat16))  # fits 16 GB beside the dense model
        model = load_text_model(source, device, load, grad_tap=grad_tap)
        for block in model.model.layers:  # recompute layers in backward: 1024-token sequences fit 16 GB
            block.forward = functools.partial(checkpoint, block.forward, use_reentrant=False)
        embed = model.get_input_embeddings()
        for fold[0], (name, rows) in enumerate(tokens.items()):
            for i, ids in enumerate(rows[:per_source].split(batch)):  # batches share each layer decode
                ids = ids.to(device)
                inputs = embed(ids).detach().requires_grad_()
                hidden = model.model(inputs_embeds=inputs, use_cache=False).last_hidden_state
                loss = sum(F.cross_entropy(model.lm_head(h).float(), y, reduction="sum")
                           for h, y in zip(hidden[:, :-1].reshape(-1, hidden.shape[-1]).split(256),
                                           ids[:, 1:].reshape(-1).split(256)))
                loss.backward()
                print(f"gradients {name} batch {i + 1} loss/token {float(loss) / ids[:, 1:].numel():.3f}", flush=True)
    return {"energy": energy, "events": events, "hidden": source.hidden}


@torch.no_grad()
def layer_sensitivity(source, tokens: dict[str, torch.Tensor], bank_path: Path, device: str, holdout: int = 4) -> list:
    """Held-out KL to the decoded-weight model when only layer L's routed experts come from the bank
    (last `holdout` sequences of each source, one batch)."""
    from .bank import ExpertBank
    from .runtime import LayerCache, load_text_model
    from .tune import kl_eval

    bank = ExpertBank(bank_path)
    bank.check(source)
    ids = [torch.cat([rows[-holdout:] for rows in tokens.values()])]
    out = []
    with source.open(device) as reader:
        target = [None]
        load = LayerCache(lambda layer: bank.layer(layer, device) if layer == target[0] else reader.layer(layer))
        model = load_text_model(source, device, load)
        teacher = [model.model(input_ids=x.to(device), use_cache=False).last_hidden_state.cpu() for x in ids]
        for layer in range(source.layers):
            target[0], load.key = layer, None
            out.append(kl_eval(model, ids, teacher, device))
            print(f"layer {layer} KL {out[-1]:.5f}", flush=True)
    return out
