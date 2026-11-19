"""Bit-shift trellis quantizer (QTIP-style) prototype: MSE on iid N(0,1) vs rate-distortion bound.
State = last L bits; each step shifts in k*V new bits and emits LUT[state] (V dims)."""
import sys, torch
dev = "cuda"


def viterbi(x, lut, L, kV):
    """x [N, T, V] -> codes [N, T] (states) minimizing sum ||x_t - lut[s_t]||^2. Free start state."""
    N, T, V = x.shape
    S, P = 1 << L, 1 << kV
    cost = torch.zeros(N, S, device=dev)
    back = torch.empty(T, N, S // P, dtype=torch.uint8, device=dev)
    ln = lut.square().sum(1)
    for t in range(T):
        d = ln[None] - 2 * x[:, t] @ lut.T  # [N, S]
        if t:
            # predecessors of new state s: (s >> kV) | top << (L-kV); best over top, shared by P successors
            m, a = cost.view(N, P, S // P).min(1)
            back[t] = a.to(torch.uint8)
            cost = m.repeat_interleave(P, 1) + d
        else:
            cost = d
    s = cost.argmin(1)
    states = torch.empty(N, T, dtype=torch.long, device=dev)
    for t in range(T - 1, -1, -1):
        states[:, t] = s
        if t:
            top = back[t].gather(1, (s >> kV)[:, None])[:, 0].long()
            s = (s >> kV) | (top << (L - kV))
    return states


def run(bpw, L, V=2, T=128, N=2048, iters=6):
    kV = round(bpw * V)
    torch.manual_seed(0)
    x = torch.randn(N, T, V, device=dev)
    xt = torch.randn(N, T, V, device=dev, generator=torch.Generator(dev).manual_seed(1))
    lut = torch.randn(1 << L, V, device=dev)
    for it in range(iters + 1):
        s = viterbi(x, lut, L, kV)
        if it < iters:  # k-means-style LUT update
            sums = torch.zeros_like(lut).index_add_(0, s.flatten(), x.reshape(-1, V))
            cnt = torch.bincount(s.flatten(), minlength=1 << L)[:, None]
            lut = torch.where(cnt > 0, sums / cnt.clamp_min(1), lut)
    s = viterbi(xt, lut, L, kV)
    mse = (xt - lut[s]).square().mean().item()
    return mse, 2 ** (-2 * bpw)


if __name__ == "__main__":
    for bpw, V in [(0.5, 4), (0.75, 4), (1.0, 2), (1.5, 2), (2.0, 2)]:
        for L in (8, 12, 16):
            mse, bound = run(bpw, L, V, N=512 if L == 16 else 2048)
            print(f"bpw {bpw} V {V} L {L}: mse {mse:.4f} bound {bound:.4f} gap {10*torch.log10(torch.tensor(mse/bound)):.2f} dB", flush=True)
