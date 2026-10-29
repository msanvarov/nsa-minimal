"""Tiny end-to-end check: can an NSA model retrieve a token outside its window?

Task ("needle recall"): the sequence is filler tokens, with exactly one token
drawn from a disjoint payload range hidden at a random position *outside* the
sliding window. The model must emit that payload token at the final step.

The sliding-window branch alone cannot see the needle, so any accuracy above
chance has to come through the compression and selection branches. Runs in
under a minute on CPU.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa import NativeSparseAttention

T, WINDOW = 128, 16
FILLER = (2, 18)   # tokens 2..17 are noise
PAYLOAD = (18, 32)  # tokens 18..31 are the needle; exactly one per sequence
VOCAB = PAYLOAD[1]
N_CLASSES = PAYLOAD[1] - PAYLOAD[0]


def batch(n, device):
    x = torch.randint(*FILLER, (n, T), device=device)
    # keep the needle strictly outside the final sliding window
    pos = torch.randint(0, T - WINDOW - 1, (n,), device=device)
    payload = torch.randint(*PAYLOAD, (n,), device=device)
    x[torch.arange(n), pos] = payload
    x[:, -1] = 0  # query slot
    return x, payload - PAYLOAD[0]


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = NativeSparseAttention(
            dim, n_heads=4, n_groups=1, head_dim=16, v_dim=16,
            cmp_block=16, cmp_stride=8, slc_block=16, n_selected=4, window=WINDOW,
        )
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, h):
        h = h + self.attn(self.n1(h))
        return h + self.mlp(self.n2(h))


class Model(nn.Module):
    """Two layers on purpose: reading "the token after the cue" is an induction
    head, which a single attention layer cannot express."""

    def __init__(self, dim=64, layers=2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim)
        self.pos = nn.Parameter(torch.randn(T, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim) for _ in range(layers))
        self.nf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, N_CLASSES)

    def forward(self, x):
        h = self.emb(x) + self.pos[: x.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.nf(h)[:, -1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "cpu"
    model = Model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    for step in range(1, a.steps + 1):
        x, y = batch(a.bs, dev)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                xe, ye = batch(512, dev)
                acc = (model(xe).argmax(-1) == ye).float().mean().item()
            print(f"step {step:4d}  loss {loss.item():.3f}  recall acc {acc:.3f}")

    chance = 1 / N_CLASSES
    print(f"\nchance accuracy is {chance:.3f}; NSA must beat it to pass")
    assert acc > 5 * chance, f"no long-range recall learned: {acc:.3f}"
    print("ok  NSA retrieved the needle from outside its sliding window")


if __name__ == "__main__":
    main()
