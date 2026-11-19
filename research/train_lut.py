"""Data-trained trellis LUT: alternate Viterbi / centroid on rotated, row-normalized expert weight tiles
from checkpoint layers (not the bench layers). usage: train_lut.py V kV L [layers]  -> ~/low-bpw-work/luts/"""
import sys, torch
from pathlib import Path
from low_bpw_quants import codecs
from low_bpw_quants.checkpoint import Checkpoint

V, kV, L = map(int, sys.argv[1:4])
layers = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "5,20,35").split(",")]
src = Checkpoint(Path("nvidia/Qwen3.6-35B-A3B-NVFP4"))
gen = torch.Generator().manual_seed(0)
xs = []
with src.open("cuda") as r:
    for l in layers:
        experts = torch.randperm(256, generator=gen)[:24].tolist()
        for W in r.layer(l, experts):
            W = W.cuda().float() @ codecs.rotation(W.shape[-1], 0, "cuda").T
            W = W / W.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
            B, rows, cols = W.shape
            g = W.reshape(B, rows, cols // 16, 16).permute(0, 2, 1, 3).reshape(-1, rows, 16)
            xs.append(codecs._tiles(g[:, :256 * (rows // 256)], 256, V))
x = torch.cat(xs)
x = x[torch.randperm(len(x), generator=gen)[:max(512, (1 << L) // 4)]].contiguous()
lut = codecs.trellis_lut(V, kV, L, "cuda").clone()
for it in range(10):
    st = codecs.viterbi(x, lut, L, kV).flatten()
    print(it, (x.reshape(-1, V) - lut[st]).square().mean().item(), flush=True)
    sums = torch.zeros_like(lut).index_add_(0, st, x.reshape(-1, V))
    cnt = torch.bincount(st, minlength=1 << L)[:, None]
    lut = torch.where(cnt > 0, sums / cnt.clamp_min(1), lut)
out = Path.home() / f"low-bpw-work/luts/tcq{V}x{kV}l{L}.pt"
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(lut.half().cpu(), out)
print("saved", out)
