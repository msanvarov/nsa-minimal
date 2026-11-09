# nsa-minimal

A minimal, readable PyTorch implementation of **Native Sparse Attention (NSA)**.

> **Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention**
> Yuan et al., DeepSeek-AI — [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) (ACL 2025)

NSA replaces full attention with three branches whose outputs are mixed by a
learned per-head gate (Eq. 5):

```math
o_t^* = \sum_{c \in \{\text{cmp},\, \text{slc},\, \text{win}\}} g_t^c \cdot \operatorname{Attn}\big( q_t,\ \tilde{K}_t^c,\ \tilde{V}_t^c \big)
```

| branch | what it reads | why |
|---|---|---|
| **compression** | one learned summary vector per block of `l` keys, stride `d` | global context, cheap |
| **selection** | the `n` highest-scoring blocks of `l'` raw tokens | fine-grained precision |
| **sliding window** | the last `w` tokens | local detail, and it stops the other two branches from having to learn it |

The selection branch is the interesting part: block importance is read straight
off the compression branch's own attention scores (Eq. 8-9), so picking blocks
costs nothing extra, and scores are summed across the heads of a GQA group
(Eq. 10) so every head in a group reads the same blocks.

## Install

```bash
pip install torch
```

## Use

```python
import torch
from nsa import NativeSparseAttention

attn = NativeSparseAttention(
    dim=512, n_heads=16, n_groups=4, head_dim=64, v_dim=64,
    cmp_block=32, cmp_stride=16, slc_block=64, n_selected=16, window=512,
)
y = attn(torch.randn(2, 1024, 512))   # (2, 1024, 512)
```

Defaults match the paper's 27B configuration (`l=32, d=16, l'=64, n=16, w=512`,
64 heads in 4 GQA groups, `d_k=192`, `d_v=128`).

## Verify

```bash
python test_nsa.py   # correctness + cost scaling
python train.py      # toy long-range recall, ~1 min on CPU
```

`test_nsa.py` checks the things that are easy to get silently wrong:

- **causality** — perturbing token `t` leaves every output before `t` bit-identical
- **exactness** — open the gate on `slc` only and select every block, and NSA
  reproduces dense causal attention to `1e-5`
- **Eq. 9** — the compressed-to-selection score remap covers every block
- **cost** — the fraction of keys read shrinks as context grows (0.97 at T=256 →
  0.20 at T=2048), and doubling T costs under 2.8x, not 4x
- short sequences (`T < l`) produce finite outputs rather than NaN

`train.py` hides one needle token outside the sliding window and asks the model
to recall it. The window branch physically cannot see it, so above-chance
accuracy has to come through compression + selection. It reaches ~100% (chance
is 7%) in about 500 steps.

## Scope

This is the **reference** form of NSA, written to be read and checked against
the paper, not to be fast. It materialises per-branch `(T, S)` masks and runs
dense SDPA on each, so it is $O(T^2)$ in time and memory and reproduces NSA's
*outputs*, not its wall-clock speedup. The paper's 9-11x gains come from a
blockwise Triton kernel that exploits the GQA-consistent block selection to load
each KV block once; swap `masked_attn` in `nsa.py` for such a kernel if you need
the speed.

Also not implemented: positional encoding (RoPE belongs in your model, not here)
and KV-cache decoding.

## License

MIT
