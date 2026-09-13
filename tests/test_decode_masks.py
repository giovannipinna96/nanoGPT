"""A decode step whose keys are all visible must not build a mask.

The mask caches are keyed on (T, S, q_offset), and during decoding S or q_offset changes at
every step, so each step built -- and kept forever -- a new mask: a BlockMask on flex, a
dense (1, S) mask on sdpa_mask. With one query at the last position every cached key is
visible on a global layer, and on a local layer the ring buffer holds at most W keys, so
no mask is needed at all. These tests pin both halves: no mask is built, and the shortcut
never fires when a mask IS needed.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
from kv_cache import CacheSpec, HybridKVCache  # noqa: E402
from model import GPT, GPTConfig, attend, get_dense_mask  # noqa: E402

CELLS = {
    "mha_LLLG": dict(attn_type='mha', attn_pattern='LLLG'),
    "mla_LLLG_absorbed": dict(attn_type='mla', attn_pattern='LLLG'),
    "mla_all_local": dict(attn_type='mla', attn_pattern='L', force_last_global=False),
}


@pytest.mark.parametrize("impl", ["sdpa_mask", "flex"])
@pytest.mark.parametrize("cell", list(CELLS))
def test_decode_builds_no_mask_and_still_matches_the_full_forward(impl, cell):
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=64, vocab_size=97, n_layer=4, n_head=4, n_embd=64, dropout=0.0,
                    bias=False, pos_encoding='rope', attn_impl=impl, window_size=8, **CELLS[cell])
    model = GPT(cfg).eval()
    cache = HybridKVCache(CacheSpec.from_config(cfg, 32), 2, 'cpu', torch.float32)
    seq = torch.randint(0, 97, (2, 32))
    with torch.no_grad():
        model(seq[:, :8], cache=cache)
        M.reset_mask_cache()
        inc = [model(seq[:, t:t + 1], cache=cache)[0][:, -1] for t in range(8, 32)]
        assert M.mask_build_count() == 0, f"{M.mask_build_count()} masks built while decoding"
        for i, t in enumerate(range(8, 32)):
            full = model(seq[:, :t + 1])[0][:, -1]
            assert torch.allclose(inc[i], full, atol=1e-4), (t, (inc[i] - full).abs().max().item())
    M.reset_mask_cache()


@pytest.mark.parametrize("impl", ["sdpa_mask", "flex"])
def test_single_query_still_masked_when_keys_fall_outside_the_window(impl):
    """T=1 on a local layer with S > W (no ring buffer in between): the mask is needed."""
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, n, 16, generator=g) for n in (1, 20, 20))
    out = attend(q, k, v, is_local=True, window=4, impl=impl)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=get_dense_mask(1, 20, True, 4, 'cpu', 19))
    full = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref, atol=1e-5)
    assert (out - full).abs().max() > 1e-3, "the window was dropped"
