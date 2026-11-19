"""Merge a zero-format score run into trellis scores; optionally retarget format names/bytes (e.g. l12 -> l14r256).
usage: add_zero.py CHECKPOINT SCORES ZERO_SCORES OUT [OLD NEW]
Any (gate, up, down) combo containing `zero` outputs 0, so its joint error is the zero run's M[...,0,0,0]
(= mean ||y0||^2 / hidden on the same replay vectors)."""
import sys
from pathlib import Path
import numpy as np
from low_bpw_quants import codecs
from low_bpw_quants.checkpoint import Checkpoint

src, scores, zero, out = Checkpoint(Path(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
old, new = sys.argv[5:7] if len(sys.argv) > 6 else ("", "")
shapes = src.expected_shapes()
out.mkdir(parents=True, exist_ok=True)
for l in range(src.layers):
    s, z = dict(np.load(scores / f"layer-{l:03d}.npz")), dict(np.load(zero / f"layer-{l:03d}.npz"))
    assert np.array_equal(s["replay_n"], z["replay_n"]), f"layer {l}: replay split differs"
    E, S, K = s["M"].shape[:3]
    names = [str(n).replace(old, new) if old else str(n) for n in s["formats"]] + ["zero"]
    M = np.broadcast_to(z["M"][:, :, :1, :1, :1], (E, S, K + 1, K + 1, K + 1)).copy()
    M[:, :, :K, :K, :K] = s["M"]
    nbytes = np.array([[codecs.FORMATS[n].nbytes(sh) for n in names] for sh in shapes])  # [3, K+1]
    s.update(M=M, formats=np.array(names), bits=np.append(s["bits"], 0.0),
             bytes=np.broadcast_to(nbytes[None], (E, 3, K + 1)).astype(np.int64))
    np.savez(out / f"layer-{l:03d}.npz", **s)
print(f"wrote {src.layers} layers to {out}: formats {names}")
