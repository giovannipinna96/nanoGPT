"""Gate T1.3 - RoPE correctness (phase 1).

Two properties that need no training, plus the spectrum check that catches the
truncated-spectrum bug.
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig, apply_rope, precompute_rope  # noqa: E402

BLOCK_SIZE = 1024


def rope_at(x, cos, sin, pos):
    """Apply RoPE to a single-position tensor x at absolute position `pos`."""
    return apply_rope(x, cos[pos:pos + 1], sin[pos:pos + 1])


@pytest.mark.parametrize("dim", [16, 32, 64])
def test_translation_invariance(dim):
    # <RoPE(q,i), RoPE(k,j)> must depend only on i-j
    cos, sin = precompute_rope(dim, 4 * BLOCK_SIZE)
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 1, 1, dim, generator=g, dtype=torch.float64)
    k = torch.randn(1, 1, 1, dim, generator=g, dtype=torch.float64)
    cos, sin = cos.double(), sin.double()
    s1 = (rope_at(q, cos, sin, 5) * rope_at(k, cos, sin, 3)).sum()
    s2 = (rope_at(q, cos, sin, 105) * rope_at(k, cos, sin, 103)).sum()
    s3 = (rope_at(q, cos, sin, 1005) * rope_at(k, cos, sin, 1003)).sum()
    # the tables are stored in float32 (they are consumed in bf16/fp32 anyway), so the
    # invariance holds to fp32 rounding, not exactly
    assert torch.allclose(s1, s2, atol=1e-5), (s1 - s2).abs().item()
    assert torch.allclose(s1, s3, atol=1e-5), (s1 - s3).abs().item()


@pytest.mark.parametrize("dim", [16, 32, 64])
def test_norm_is_preserved(dim):
    # RoPE is a rotation, so it preserves the L2 norm at every position
    cos, sin = precompute_rope(dim, 4 * BLOCK_SIZE)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 3, 1, dim, generator=g, dtype=torch.float64)
    cos, sin = cos.double(), sin.double()
    for pos in (0, 1, 17, 511, 1023, 4095):
        y = rope_at(x, cos, sin, pos)
        assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-10)


@pytest.mark.parametrize("dim", [16, 32, 64])
def test_spectrum_is_not_truncated(dim):
    """The slowest channel must have a period far beyond block_size.

    Computing the frequencies on head_dim and keeping only the first qk_rope/2 of them
    gives a maximum wavelength of 471 tokens instead of ~35000, so positions alias
    inside the nominal context.
    """
    # Read the spectrum off the table precompute_rope actually builds: a formula
    # recomputed here would keep passing if the implementation truncated the spectrum.
    # At position 1 every angle is its channel's frequency, below pi, and atan2 resolves
    # even the slowest one (~3e-4 rad) from the float32 sin/cos to far better than 1 token.
    cos, sin = precompute_rope(dim, 4 * BLOCK_SIZE)
    freqs = torch.atan2(sin[1].double(), cos[1].double())
    assert freqs.numel() == dim // 2
    wl_max = 2 * math.pi / freqs.min().item()
    assert wl_max > 10 * BLOCK_SIZE, f"RoPE aliasing: max wavelength {wl_max:.0f}"
    # measured: 19869 (dim 16, the config-A MLA subspace), 35333 (dim 32),
    # 47117 (dim 64, a full MHA head) -- against 471 for the truncated-spectrum bug
    expected = {16: 19869, 32: 35333, 64: 47117}[dim]
    assert abs(wl_max - expected) < 1, f"expected ~{expected}, got {wl_max:.0f}"


def test_distinct_positions_are_distinguishable_across_the_context():
    # the practical consequence of a healthy spectrum: two positions block_size apart
    # must not produce (nearly) the same rotation
    cos, sin = precompute_rope(32, 4 * BLOCK_SIZE)
    g = torch.Generator().manual_seed(2)
    q = torch.randn(1, 1, 1, 32, generator=g, dtype=torch.float64)
    k = torch.randn(1, 1, 1, 32, generator=g, dtype=torch.float64)
    cos, sin = cos.double(), sin.double()
    near = (rope_at(q, cos, sin, 0) * rope_at(k, cos, sin, 0)).sum()
    far = (rope_at(q, cos, sin, BLOCK_SIZE) * rope_at(k, cos, sin, 0)).sum()
    assert (near - far).abs() > 1e-3


def test_rope_model_has_no_wpe_and_non_persistent_buffers():
    cfg = GPTConfig(block_size=64, vocab_size=65, n_layer=2, n_head=4, n_embd=128,
                    dropout=0.0, bias=False, pos_encoding='rope', attn_impl='sdpa_mask')
    m = GPT(cfg)
    assert 'wpe' not in m.transformer, "wpe must be gone with RoPE"
    sd = m.state_dict()
    assert not any('rope_cos' in k or 'rope_sin' in k for k in sd), \
        "rope tables must be non-persistent buffers, not checkpoint weights"
    out, _ = m(torch.randint(0, 65, (2, 64)))
    assert out.shape[0] == 2


def test_learned_and_rope_models_differ_only_in_positions():
    kw = dict(block_size=64, vocab_size=65, n_layer=2, n_head=4, n_embd=128,
              dropout=0.0, bias=False, attn_impl='sdpa_mask')
    torch.manual_seed(0); learned = GPT(GPTConfig(pos_encoding='learned', **kw))
    torch.manual_seed(0); rope = GPT(GPTConfig(pos_encoding='rope', **kw))
    # wpe is 64*128 params, and get_num_params subtracts it only when present
    assert learned.get_num_params() == rope.get_num_params()
