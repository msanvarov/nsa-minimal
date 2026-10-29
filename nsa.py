"""Native Sparse Attention (NSA), arXiv:2502.11089.

Reference implementation of the three-branch sparse attention from DeepSeek's
NSA paper: coarse-grained token *compression*, fine-grained block *selection*,
and a *sliding window*, combined by a learned per-head gate.

Equation numbers in comments refer to the paper.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_attn(q, k, v, mask):
    """q: (B,G,h,T,D)  k,v: (B,G,1,S,D)  mask: (...,T,S) bool, True = visible.

    Rows with nothing visible return zeros instead of NaN.
    """
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    logits = logits.masked_fill(~mask, float("-inf"))
    live = mask.any(-1, keepdim=True)
    p = logits.softmax(-1).masked_fill(~live, 0.0)
    return p @ v, p


class BlockCompress(nn.Module):
    """phi in Eq. 7: learnable MLP with intra-block position encoding."""

    def __init__(self, dim, block):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(block, dim))
        self.proj = nn.Sequential(
            nn.Linear(block * dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, blocks):  # (B,G,n,block,D) -> (B,G,n,D)
        return self.proj((blocks + self.pos).flatten(-2))


class NativeSparseAttention(nn.Module):
    """NSA with GQA. Defaults are the paper's 27B settings (Section 4.1).

    ponytail: this is the *reference* form -- it materialises (T,S) masks and
    runs dense SDPA per branch, so it is O(T^2) and reproduces NSA's outputs,
    not its wall-clock speedup. The speedup needs the paper's blockwise Triton
    kernel; swap `masked_attn` for one if you need it.
    """

    def __init__(
        self,
        dim,
        n_heads=64,
        n_groups=4,
        head_dim=192,
        v_dim=128,
        cmp_block=32,
        cmp_stride=16,
        slc_block=64,
        n_selected=16,
        window=512,
    ):
        super().__init__()
        assert n_heads % n_groups == 0
        # Eq. 9's score remapping needs both block sizes to be stride multiples.
        assert cmp_block % cmp_stride == 0 and slc_block % cmp_stride == 0
        assert cmp_stride < cmp_block, "paper requires d < l"

        self.n_heads, self.n_groups = n_heads, n_groups
        self.hpg = n_heads // n_groups
        self.head_dim, self.v_dim = head_dim, v_dim
        self.l, self.d, self.lp = cmp_block, cmp_stride, slc_block
        self.n_selected, self.window = n_selected, window

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_groups * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_groups * v_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * v_dim, dim, bias=False)

        self.k_cmp = BlockCompress(head_dim, cmp_block)
        self.v_cmp = BlockCompress(v_dim, cmp_block)
        # Eq. 5: gate per (head, branch), from the query-side features.
        self.gate = nn.Linear(dim, n_heads * 3, bias=True)

    def _slc_from_cmp(self, n_cmp, n_slc, device):
        """Eq. 9: p_slc[j] = sum_m sum_n p_cmp[(l'/d) j + m + n], as a matrix."""
        w = torch.zeros(n_slc, n_cmp, device=device)
        for j in range(n_slc):
            for m in range(self.lp // self.d):
                for n in range(self.l // self.d):
                    i = (self.lp // self.d) * j + m + n
                    if i < n_cmp:
                        w[j, i] += 1.0
        return w

    def forward(self, x):
        B, T, _ = x.shape
        G, h, Dk, Dv = self.n_groups, self.hpg, self.head_dim, self.v_dim
        dev = x.device
        t = torch.arange(T, device=dev)

        q = self.q_proj(x).view(B, T, G, h, Dk).permute(0, 2, 3, 1, 4)
        k = self.k_proj(x).view(B, T, G, Dk).permute(0, 2, 1, 3).unsqueeze(2)
        v = self.v_proj(x).view(B, T, G, Dv).permute(0, 2, 1, 3).unsqueeze(2)

        causal = (t[:, None] >= t[None, :]).view(1, 1, 1, T, T)

        # ---- branch 1: compression (Eq. 7) --------------------------------
        n_cmp = max(0, (T - self.l) // self.d + 1)
        if n_cmp:
            idx = t[:n_cmp, None] * self.d + torch.arange(self.l, device=dev)
            kb = k.squeeze(2)[:, :, idx]  # (B,G,n_cmp,l,Dk)
            vb = v.squeeze(2)[:, :, idx]
            k_c = self.k_cmp(kb).unsqueeze(2)  # (B,G,1,n_cmp,Dk)
            v_c = self.v_cmp(vb).unsqueeze(2)
            # block i spans [i*d, i*d+l); visible to query t once it is complete
            ends = torch.arange(n_cmp, device=dev) * self.d + self.l
            m_cmp = (ends[None, :] <= t[:, None] + 1).view(1, 1, 1, T, n_cmp)
            o_cmp, p_cmp = masked_attn(q, k_c, v_c, m_cmp)
        else:
            o_cmp = q.new_zeros(B, G, h, T, Dv)
            p_cmp = q.new_zeros(B, G, h, T, 0)

        # ---- branch 2: selection (Eq. 8-12) -------------------------------
        n_slc = math.ceil(T / self.lp)
        if n_cmp:
            w = self._slc_from_cmp(n_cmp, n_slc, dev)
            p_slc = p_cmp @ w.T  # (B,G,h,T,n_slc)
            p_slc = p_slc.sum(2)  # Eq. 10: GQA-consistent, sum over heads
        else:
            p_slc = q.new_zeros(B, G, T, n_slc)

        blk = torch.arange(n_slc, device=dev)
        reachable = (blk[None, :] * self.lp) <= t[:, None]  # (T,n_slc)
        # paper keeps 1 initial + 2 local blocks unconditionally
        forced = (blk[None, :] == 0) | (blk[None, :] == t[:, None] // self.lp)
        forced |= blk[None, :] == (t[:, None] // self.lp - 1)
        forced &= reachable

        scores = p_slc.masked_fill(~reachable, float("-inf"))
        scores = scores.masked_fill(forced, float("inf"))
        keep = min(self.n_selected, n_slc)
        top = scores.topk(keep, dim=-1).indices
        chosen = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top, True)
        chosen &= reachable
        # expand block choice to tokens, then re-apply causality
        m_slc = chosen[..., blk.repeat_interleave(self.lp)[:T]]
        m_slc = m_slc.unsqueeze(2) & causal  # (B,G,1,T,T)
        o_slc, _ = masked_attn(q, k, v, m_slc)

        # ---- branch 3: sliding window -------------------------------------
        m_win = causal & (t[:, None] - t[None, :] < self.window).view(1, 1, 1, T, T)
        o_win, _ = masked_attn(q, k, v, m_win)

        # ---- gated combination (Eq. 5) ------------------------------------
        g = self.gate(x).sigmoid().view(B, T, G, h, 3).permute(0, 2, 3, 1, 4)
        o = g[..., 0:1] * o_cmp + g[..., 1:2] * o_slc + g[..., 2:3] * o_win
        o = o.permute(0, 3, 1, 2, 4).reshape(B, T, G * h * Dv)
        return self.o_proj(o)
