"""Self-checks for the NSA reference implementation. Run: python test_nsa.py"""

import math

import torch
import torch.nn.functional as F

from nsa import NativeSparseAttention, masked_attn


def small(**kw):
    cfg = dict(
        dim=64, n_heads=8, n_groups=2, head_dim=16, v_dim=16,
        cmp_block=8, cmp_stride=4, slc_block=8, n_selected=4, window=6,
    )
    cfg.update(kw)
    return NativeSparseAttention(**cfg)


def dense_causal(attn, x):
    """Full causal GQA attention through the same projections, for reference."""
    B, T, _ = x.shape
    G, h, Dk, Dv = attn.n_groups, attn.hpg, attn.head_dim, attn.v_dim
    q = attn.q_proj(x).view(B, T, G, h, Dk).permute(0, 2, 3, 1, 4)
    k = attn.k_proj(x).view(B, T, G, Dk).permute(0, 2, 1, 3).unsqueeze(2)
    v = attn.v_proj(x).view(B, T, G, Dv).permute(0, 2, 1, 3).unsqueeze(2)
    t = torch.arange(T)
    m = (t[:, None] >= t[None, :]).view(1, 1, 1, T, T)
    o, _ = masked_attn(q, k, v, m)
    return attn.o_proj(o.permute(0, 3, 1, 2, 4).reshape(B, T, G * h * Dv))


def test_shape_and_grad():
    attn = small()
    x = torch.randn(2, 32, 64, requires_grad=True)
    y = attn(x)
    assert y.shape == x.shape, y.shape
    y.sum().backward()
    # every branch must receive gradient
    for name in ("k_cmp.proj.0.weight", "gate.weight", "q_proj.weight"):
        p = dict(attn.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert x.grad.abs().sum() > 0


def test_causality():
    """Perturbing token t must not change outputs at positions < t."""
    torch.manual_seed(0)
    attn = small().eval()
    x = torch.randn(1, 24, 64)
    y = attn(x)
    cut = 10
    x2 = x.clone()
    x2[:, cut:] += torch.randn_like(x2[:, cut:]) * 5
    y2 = attn(x2)
    assert torch.allclose(y[:, :cut], y2[:, :cut], atol=1e-5), (
        (y[:, :cut] - y2[:, :cut]).abs().max()
    )


def test_selection_branch_recovers_full_attention():
    """Select every block + gate open only on `slc` => exact causal attention."""
    torch.manual_seed(0)
    T = 32
    attn = small(n_selected=T, window=1).eval()
    with torch.no_grad():  # gate -> [0, 1, 0]
        attn.gate.weight.zero_()
        b = attn.gate.bias.view(attn.n_heads, 3)
        b.copy_(torch.tensor([-30.0, 30.0, -30.0]).expand_as(b))
    x = torch.randn(2, T, 64)
    got, want = attn(x), dense_causal(attn, x)
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_eq9_score_remapping():
    """Eq. 9 must route every compressed block into >=1 selection block."""
    attn = small(cmp_block=8, cmp_stride=4, slc_block=8)
    w = attn._slc_from_cmp(n_cmp=7, n_slc=4, device=torch.device("cpu"))
    assert w.shape == (4, 7)
    assert (w.sum(0) > 0).all(), "a compressed block scores into no selection block"
    # l'/d = 2 and l/d = 2 -> selection block j draws on cmp blocks 2j..2j+2
    assert w[1].nonzero().flatten().tolist() == [2, 3, 4]


def keys_read_per_head(attn, T):
    """Mean number of keys one query head actually attends to, summed over branches."""
    import nsa as nsa_mod

    total, real = [], nsa_mod.masked_attn

    def counting(q, k, v, mask):
        total.append(mask.flatten(0, -3).float().mean(0).sum().item())
        return real(q, k, v, mask)

    nsa_mod.masked_attn = counting
    try:
        attn(torch.randn(1, T, 64))
    finally:
        nsa_mod.masked_attn = real
    return sum(total)


def test_cost_grows_subquadratically():
    """The fraction of keys NSA reads must shrink as context grows.

    Measured here (n=4, l'=32, w=32, l=32, d=16):
        T= 256 -> 0.97    T= 512 -> 0.57
        T=1024 -> 0.33    T=2048 -> 0.20
    The paper's 27B budget (n=16, l'=64, w=512) at 64k tokens is ~2%.
    """
    attn = small(n_selected=4, window=32, cmp_block=32, cmp_stride=16, slc_block=32)
    fracs = [keys_read_per_head(attn, T) / (T * (T + 1) / 2) for T in (256, 512, 1024, 2048)]
    assert all(b < a for a, b in zip(fracs, fracs[1:])), fracs
    assert fracs[-1] < 0.25, fracs
    # doubling T must not double the per-query cost the way dense attention does
    reads = [keys_read_per_head(attn, T) for T in (1024, 2048)]
    assert reads[1] / reads[0] < 2.8, reads[1] / reads[0]


def test_no_nan_on_short_sequences():
    """Sequences shorter than one compression block must still work."""
    attn = small(cmp_block=8, cmp_stride=4)
    for T in (1, 3, 7, 8, 9):
        y = attn(torch.randn(1, T, 64))
        assert torch.isfinite(y).all(), T


if __name__ == "__main__":
    for fn in [
        test_shape_and_grad,
        test_causality,
        test_selection_branch_recovers_full_attention,
        test_eq9_score_remapping,
        test_cost_grows_subquadratically,
        test_no_nan_on_short_sequences,
    ]:
        fn()
        print(f"ok  {fn.__name__}")
    print("\nall checks passed")
