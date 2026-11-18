import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from low_bpw_quants import allocation, bank, codecs, score
from low_bpw_quants.checkpoint import FP4_VALUES, decode_mxfp8, decode_nvfp4


class QuantizerTest(unittest.TestCase):
    def test_native_codecs(self):
        packed = torch.tensor([[0x21, 0xA3, 0x65, 0x80, 0, 0, 0, 0]], dtype=torch.uint8)
        scales = torch.ones(1, 1).to(torch.float8_e4m3fn)
        weights = decode_nvfp4(packed, scales, torch.tensor(2.), torch.tensor(FP4_VALUES))
        self.assertEqual(weights[0, :8].tolist(), [1, 2, 3, -2, 6, 8, 0, 0])
        mx = decode_mxfp8(torch.ones(2, 64).to(torch.float8_e4m3fn),
                          torch.tensor([[127, 128], [126, 127]], dtype=torch.uint8))
        self.assertEqual(mx[:, [0, 32]].tolist(), [[1, 2], [.5, 1]])

    def test_payloads_round_trip(self):
        gen = torch.Generator().manual_seed(0)
        W = torch.randn(2, 32, 256, generator=gen)
        W[1, 5] = 0
        X = torch.randn(2, 400, 256, generator=gen)
        H = X.transpose(1, 2) @ X / 400
        for fmt in (codecs.Rotated(codecs.VQ(8, 8)), codecs.VQ(4, 8), codecs.Scalar(2, 128, True),
                    codecs.Sage(cap=2), codecs.Binary256()):
            Q, raw = codecs.quantize(fmt, W, H)
            self.assertEqual([r.numel() for r in raw], [fmt.nbytes(W.shape[1:])] * 2, fmt.name)
            torch.testing.assert_close(torch.stack([codecs.decode(r, w.shape) for r, w in zip(raw, W)]), Q)
            self.assertLess(float((Q - W).square().sum() / W.square().sum()), 1.0, fmt.name)

    def test_allocation_budget(self):
        rng = np.random.default_rng(1)
        costs = [np.sort(rng.integers(10, 100, 5)) for _ in range(50)]
        dists = [np.sort(rng.random(5))[::-1].copy() for _ in range(50)]
        budget = sum(int(c[2]) for c in costs)
        pick = allocation.greedy(costs, dists, budget)
        self.assertLessEqual(sum(int(c[p]) for c, p in zip(costs, pick)), budget)
        self.assertLessEqual(sum(d[p] for d, p in zip(dists, pick)), sum(d[2] for d in dists))

    def test_bank_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            choice = np.array([[[0, 1, 0], [1, 1, 1], [0, 0, 1]], [[1, 0, 0], [0, 0, 0], [1, 1, 1]]], dtype=np.int32)
            source, weights, report, path = make_bank(Path(tmp), ["rvq8x8", "gsq2odd"], choice)
            restored = bank.ExpertBank(path)
            for layer in range(2):
                for p, (w, q) in enumerate(zip(weights[layer], restored.layer(layer, "cpu"))):
                    err = (q - w).square().sum((1, 2)) / w.square().sum((1, 2))
                    self.assertTrue(bool((err < 0.6).all()), (layer, p, err))
            self.assertGreater(report["routed_bpw"], 1.0)

    def test_tune_layout(self):
        from low_bpw_quants import runtime, tune

        with tempfile.TemporaryDirectory() as tmp:
            choice = np.array([[[0, 1, 0], [1, 1, 1], [0, 0, 1]], [[1, 0, 0], [0, 0, 0], [1, 1, 1]]], dtype=np.int32)
            source, _, _, path = make_bank(Path(tmp), ["rvq8x8", "rvq4x8"], choice)
            restored = bank.ExpertBank(path)
            layout = tune.Layout(restored)
            gen = torch.Generator().manual_seed(3)
            x = torch.randn(10, 256, generator=gen)
            ids = torch.randint(0, 3, (10, 2), generator=gen)
            route = torch.rand(10, 2, generator=gen)
            decoded = restored.layer(1, "cpu")
            experts = tune.TunedExperts(1, lambda l: layout.units(l, "cpu"),
                                        [torch.from_numpy(layout.scales(1, p)) for p in range(3)], (None, None))
            got, want = experts(x, ids, route), runtime.moe_forward(x, ids, route, decoded, (None, None))
            self.assertLess(float((got - want).norm() / want.norm()), 0.01)  # BF16 activations
            out = Path(tmp) / "tuned.gguf"
            layout.write(out, {(l, p): 2 * layout.scales(l, p) for l in range(2) for p in range(3)})
            for q, d in zip(bank.ExpertBank(out).layer(1, "cpu"), decoded):
                torch.testing.assert_close(q, 2 * d)


def make_bank(tmp: Path, formats, choice):
    """A 2-layer, 3-expert bank quantized with identity Hessians (score.run patched out)."""
    source = SimpleNamespace(arch="qwen3_5_moe", layers=2, experts=3, hidden=256, ffn=128, gate_cap=None,
                             up_cap=None, expected_shapes=lambda: ((128, 256), (128, 256), (256, 128)))
    gen = torch.Generator().manual_seed(2)
    weights = [tuple(torch.randn(3, *s, generator=gen) for s in source.expected_shapes()) for _ in range(2)]
    stats = lambda layer: SimpleNamespace(
        load=lambda _: weights[layer],
        hessians=lambda idx, prior: (torch.eye(256).expand(3, 256, 256)[idx], torch.eye(128).expand(3, 128, 128)[idx]))

    def walk(src, tokens, out, device, formats, *, on_layer, **_):
        for layer in range(src.layers):
            on_layer(stats(layer), layer)

    original, score.run = score.run, walk
    try:
        np.savez(tmp / "alloc.npz", formats=np.array(formats), choice=choice)
        report = bank.quantize(source, {}, tmp / "alloc.npz", tmp / "bank.gguf", "cpu", hessian_seqs=1, batch=1,
                               expert_batch=2, passes=1, sequential=True)
    finally:
        score.run = original
    return source, weights, report, tmp / "bank.gguf"

if __name__ == "__main__":
    unittest.main()
