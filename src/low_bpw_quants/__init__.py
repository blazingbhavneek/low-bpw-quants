from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Mixed SAGE / GSQ routed-expert quantization with NatScale allocation")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, help):
        p = sub.add_parser(name, help=help)
        p.add_argument("checkpoint", type=Path)
        p.add_argument("--device", default=device)
        return p

    command("inspect", "show checkpoint layout and missing shards")
    p = command("tokens", "pack calibration sequences from the downloaded sources")
    p.add_argument("data", type=Path, help="directory with ultrachat/magpie/writingprompts parquet and c4.json.gz")
    p.add_argument("output", type=Path)
    p.add_argument("--sequences", type=int, default=64, help="per source")
    p.add_argument("--length", type=int, default=1024)
    p = command("gradients", "NatScale gradient sensitivity: squared CE gradients at expert outputs")
    p.add_argument("tokens", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--per-source", type=int, default=16)
    p = command("score", "per-layer Hessians, held-out replay, and joint candidate errors")
    p.add_argument("tokens", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--formats", nargs="+", default=["rvq8x8", "rvq8x10", "rvq8x12", "rvq8x14", "rvq8x16"])
    p.add_argument("--prior", type=float, default=2.0, help="Hessian shrinkage pseudo-tokens per input dim")
    p.add_argument("--hessian-seqs", type=int, default=48, help="per source")
    p.add_argument("--replay-seqs", type=int, default=16, help="per source, held out from the Hessians")
    p.add_argument("--replay-cap", type=int, default=256, help="replay vectors per expert and fold")
    p.add_argument("--batch", type=int, default=4, help="sequences per walk chunk")
    p.add_argument("--expert-batch", type=int, default=16)
    p.add_argument("--layers", type=int, nargs="+")
    p = command("allocate", "NatScale contrast-calibrated allocation under an exact byte budget")
    p.add_argument("scores", type=Path)
    p.add_argument("gradients", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--target-bpw", type=float, default=1.5, help="routed GGUF bytes * 8 / routed weights")
    p.add_argument("--contrast", type=float, default=1.0, help="target Jeffreys divergence in nats")
    p.add_argument("--uniform", help="ablation: one format for every projection")
    p.add_argument("--no-prior", action="store_true", help="ablation: drop NatScale's structural prior A_i")
    p.add_argument("--tied", action="store_true", help="ablation: one format per expert, not per projection")
    p.add_argument("--objective", choices=["natscale", "fisher"], default="natscale")
    p.add_argument("--pool", choices=["mean", "max"], default="mean", help="exposure across calibration sources")
    p.add_argument("--drop", nargs="+", default=(), help="formats the allocation may not use")
    p.add_argument("--layer-scale", type=Path, help="JSON list of per-layer multipliers on the proxy distortion")
    p = command("layer-scale", "measure each layer's KL from a bank and calibrate the proxy per layer")
    p.add_argument("tokens", type=Path)
    p.add_argument("bank", type=Path)
    p.add_argument("allocation", type=Path, help="the allocation the bank was quantized from")
    p.add_argument("scores", type=Path)
    p.add_argument("gradients", type=Path)
    p.add_argument("output", type=Path, help="JSON list for allocate --layer-scale")
    p.add_argument("--contrast", type=float, default=1.0)
    p.add_argument("--holdout", type=int, default=4, help="held-out sequences per source")
    p.add_argument("--previous", type=Path, help="earlier layer-scale JSON: take the geometric mean")
    p = command("quantize", "write the routed-expert GGUF for an allocation")
    p.add_argument("tokens", type=Path)
    p.add_argument("allocation", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--gsq-steps", type=int, default=0, help="GSQ Gumbel refinement steps for gsq2 projections")
    p.add_argument("--hessian-seqs", type=int, default=48)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--expert-batch", type=int, default=16)
    p.add_argument("--prior", type=float, default=2.0)
    p.add_argument("--passes", type=int, default=2, help="VQ coordinate-descent passes after LDLQ")
    p.add_argument("--sequential", action="store_true", help="later layers see quantized earlier layers")
    p = command("tune", "KL-distil a rotated-VQ bank's FP16 row scales end to end (codes fixed)")
    p.add_argument("tokens", type=Path)
    p.add_argument("bank", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam step on log row scales")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--holdout", type=int, default=4, help="validation sequences per source")
    p.add_argument("--per-row", action="store_true", help="one scale per row instead of per expert projection")
    p = command("evaluate", "WikiText-2 KL / top-1 / perplexity against the decoded-weight teacher")
    p.add_argument("parquet", type=Path)
    p.add_argument("--bank", type=Path)
    p.add_argument("--teacher-cache", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = parser.parse_args()

    from .checkpoint import Checkpoint

    try:
        model = Checkpoint(args.checkpoint)
        if args.command == "inspect":
            shapes = model.expected_shapes()
            report = {"architecture": model.arch, "layers": model.layers, "experts_per_layer": model.experts,
                      "projection_shapes": shapes,
                      "routed_weights": model.layers * model.experts * sum(r * c for r, c in shapes),
                      **model.dense_stats(), "missing_shards": model.missing_shards()}
        elif args.command == "tokens":
            from transformers import AutoTokenizer

            from .calibrate import build

            tokenizer = AutoTokenizer.from_pretrained(model.path, local_files_only=True)
            tokens = build(tokenizer, args.data, args.sequences, args.length)
            torch.save(tokens, args.output)
            report = {name: list(t.shape) for name, t in tokens.items()}
        elif args.command == "gradients":
            import numpy as np

            from .calibrate import gradients

            result = gradients(model, torch.load(args.tokens), args.per_source, args.device)
            np.savez(args.output, **result)
            report = {"output": str(args.output), "covered": float((result["events"].sum(0) > 0).mean())}
        elif args.command == "score":
            from . import codecs, score

            formats = [codecs.FORMATS[name] for name in args.formats]
            score.run(model, torch.load(args.tokens), args.output, args.device, formats,
                      hessian_seqs=args.hessian_seqs, replay_seqs=args.replay_seqs, replay_cap=args.replay_cap,
                      batch=args.batch, expert_batch=args.expert_batch, layers=args.layers, prior=args.prior)
            report = {"output": str(args.output)}
        elif args.command == "allocate":
            from .allocation import allocate

            report = allocate(model, args.scores, args.gradients, args.output, args.target_bpw,
                              args.contrast, args.uniform, prior=not args.no_prior, tied=args.tied,
                              objective=args.objective, pool=args.pool, drop=tuple(args.drop),
                              layer_scale=json.loads(args.layer_scale.read_text()) if args.layer_scale else None)
        elif args.command == "layer-scale":
            import numpy as np

            from .allocation import layer_proxy, objective_terms
            from .calibrate import layer_sensitivity

            measured = np.array(layer_sensitivity(model, torch.load(args.tokens), args.bank, args.device,
                                                  args.holdout))
            terms, combos, names, _ = objective_terms(model, args.scores, args.gradients, args.contrast)
            with np.load(args.allocation) as data:
                if [str(n) for n in data["formats"]] != names:
                    raise ValueError("allocation formats differ from the scores' formats")
                proxy = layer_proxy(data["choice"], terms, combos)
            scale = measured / proxy
            scale /= scale.mean()
            if args.previous:
                scale = np.sqrt(scale * np.array(json.loads(args.previous.read_text())))
                scale /= scale.mean()
            args.output.write_text(json.dumps(scale.tolist()))
            report = {"output": str(args.output), "measured_kl": measured.round(6).tolist(),
                      "scale": scale.round(4).tolist()}
        elif args.command == "quantize":
            from .bank import quantize

            report = quantize(model, torch.load(args.tokens), args.allocation, args.output, args.device,
                              gsq_steps=args.gsq_steps, hessian_seqs=args.hessian_seqs, batch=args.batch,
                              expert_batch=args.expert_batch, prior=args.prior, passes=args.passes,
                              sequential=args.sequential)
        elif args.command == "tune":
            from .tune import tune

            report = tune(model, args.bank, torch.load(args.tokens), args.output, args.device, epochs=args.epochs,
                          lr=args.lr, batch=args.batch, holdout=args.holdout, per_row=args.per_row)
        else:
            from .evaluate import run

            report = run(model, args.bank, args.parquet, args.device, args.teacher_cache, args.output)
        print(json.dumps(report, indent=2))
    except (OSError, RuntimeError, ValueError, KeyError, ImportError) as error:
        print(f"{parser.prog}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
