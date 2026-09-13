"""Bookkeeping gates for the hybrid cache (sub-step 8a).

The parity gates T4.1/T4.2, which need the model, land in sub-step 8b. What is checked
here is the part that is pure arithmetic and therefore checkable exactly: capacities,
wrap-around ordering, and the analytic size that the report's headline table uses.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_cache import CacheSpec, HybridKVCache  # noqa: E402
from model import GPTConfig  # noqa: E402


# ------------------------------------------------------------------ capacities (C4)
def test_local_layers_get_window_capacity_global_layers_do_not():
    cfg = GPTConfig(n_layer=8, n_head=8, n_embd=512, attn_type='mla',
                    attn_pattern='LLLG', window_size=256, block_size=1024)
    spec = CacheSpec.from_config(cfg, max_seq_len=16384)
    cache = HybridKVCache(spec, batch_size=1, device='cpu', dtype=torch.float32)
    caps = cache.capacities()
    assert caps == [256, 256, 256, 16384, 256, 256, 256, 16384]
    assert all(c == 256 for c, loc in zip(caps, spec.is_local) if loc)


def test_all_global_pattern_allocates_full_length_everywhere():
    cfg = GPTConfig(n_layer=4, n_head=8, n_embd=512, attn_type='mla', attn_pattern='G')
    spec = CacheSpec.from_config(cfg, max_seq_len=4096)
    assert HybridKVCache(spec, 1, 'cpu', torch.float32).capacities() == [4096] * 4


def test_from_config_is_the_only_mapping_and_honours_force_last_global():
    """CacheSpec.from_config replaced nine hand-written copies of cfg -> spec.

    The field that carries real weight is force_last_global: without it cell 7 stops being
    all-local, its last layer gets max_seq_len capacity instead of the window, and its cache
    stops being constant in T -- silently. Three of the nine copies omitted it.
    """
    kw = dict(n_layer=8, n_head=16, n_embd=512, attn_type='mla', attn_pattern='L',
              window_size=256, block_size=1024)
    cell7 = CacheSpec.from_config(GPTConfig(force_last_global=False, **kw), 16384)
    forced = CacheSpec.from_config(GPTConfig(**kw), 16384)

    assert cell7.is_local == [True] * 8
    assert [cell7.capacity(i) for i in range(8)] == [256] * 8
    assert forced.is_local[-1] is False and forced.capacity(7) == 16384
    # and the difference is the headline property of cell 7: memory constant in T.
    # Compared BELOW max_seq_len, because elements() caps each layer at its own capacity
    # and past 16384 even the global layer stops growing.
    assert cell7.elements(4096) == cell7.elements(16384)
    assert forced.elements(16384) > forced.elements(4096)


def test_from_config_takes_v_head_dim_from_the_config():
    """Belt and braces: MLA does not cache v, and mha/gqa pin v_head_dim to head_dim, so a
    copy that wrote `v_head_dim = n_embd // n_head` could not be wrong today. It would be
    the moment a latent layer started sizing anything from it."""
    sym = GPTConfig(n_embd=512, n_head=16, attn_type='mla', symmetric_head_dims=True)
    assert sym.v_head_dim != sym.n_embd // sym.n_head
    assert CacheSpec.from_config(sym, 1024).v_head_dim == sym.v_head_dim


# ------------------------------------------------------------------ contents (C3)
def test_latent_cache_has_no_place_for_reconstructed_tensors():
    """The single most damaging silent bug: caching k_nope/v gives MHA-sized memory."""
    cfg = GPTConfig(n_layer=2, n_head=8, n_embd=512, attn_type='mla', attn_pattern='G')
    cache = HybridKVCache(CacheSpec.from_config(cfg, 128), 1, 'cpu', torch.float32)
    for layer in cache.layers:
        assert set(layer.buffers) == {'c_kv', 'k_rope'}
        assert layer.buffers['c_kv'].shape[-1] == cfg.kv_lora_rank
        assert layer.buffers['k_rope'].shape[-1] == cfg.qk_rope_head_dim


def test_dense_cache_stores_k_and_v():
    cfg = GPTConfig(n_layer=2, n_head=8, n_embd=512, attn_type='mha', attn_pattern='G')
    cache = HybridKVCache(CacheSpec.from_config(cfg, 128), 1, 'cpu', torch.float32)
    for layer in cache.layers:
        assert set(layer.buffers) == {'k', 'v'}


# ------------------------------------------------------------------ wrap-around (C2)
@pytest.mark.parametrize("capacity,n_written", [(4, 4), (4, 5), (4, 9), (8, 30)])
def test_read_returns_chronological_order_after_wrap(capacity, n_written):
    spec = CacheSpec(n_layer=1, is_local=[True], window_size=capacity, max_seq_len=64,
                     attn_type='mla', kv_lora_rank=1, qk_rope_head_dim=1)
    cache = HybridKVCache(spec, batch_size=1, device='cpu', dtype=torch.float32)
    layer = cache.layers[0]
    # write tokens one at a time, marking each with its own absolute position
    for p in range(n_written):
        layer.write(p, c_kv=torch.full((1, 1, 1), float(p)),
                    k_rope=torch.full((1, 1, 1), float(p)))
    out, start = layer.read(n_written)
    got = out['c_kv'].flatten().tolist()
    n = min(n_written, capacity)
    assert start == n_written - n
    assert got == [float(p) for p in range(start, n_written)], got


@pytest.mark.parametrize("capacity,n_tokens", [(4, 7), (16, 24), (16, 40), (8, 8)])
def test_write_a_whole_chunk_across_the_wrap(capacity, n_tokens):
    """Prefill writes many tokens at once and may straddle the wrap point (C7).

    Regression test for a real bug: with a chunk longer than the buffer the wrapped
    index vector has DUPLICATES, and duplicate-index assignment has no defined winner.
    It kept the FIRST write, so a prompt longer than the window left stale keys and
    decoding diverged by ~1.0 in logit space from the very first step. A width-1 payload
    happened to behave; a realistic one did not, hence the width below.
    """
    width = 32
    spec = CacheSpec(n_layer=1, is_local=[True], window_size=capacity, max_seq_len=256,
                     attn_type='mla', kv_lora_rank=width, qk_rope_head_dim=width)
    layer = HybridKVCache(spec, 1, 'cpu', torch.float32).layers[0]
    vals = torch.arange(n_tokens, dtype=torch.float32).view(1, n_tokens, 1).repeat(1, 1, width)
    layer.write(0, c_kv=vals, k_rope=vals)
    out, start = layer.read(n_tokens)
    kept = min(n_tokens, capacity)
    assert start == n_tokens - kept
    assert out['c_kv'][0, :, 0].tolist() == [float(p) for p in range(start, n_tokens)]
    assert (out['c_kv'] == out['c_kv'][:, :, :1]).all(), "every channel must agree"


# ------------------------------------------------------------------ analytic size (T5.1)
def test_analytic_cache_matches_the_documented_ratios():
    """The nanoGPT-scale numbers of 01_meccanismi_attention.md 4.2.

    n_embd=768, n_head=12, L=12, d_c=256, d^R=32, W=256, 5 local : 1 global.
    """
    common = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, window_size=256)
    mha_g = GPTConfig(attn_type='mha', attn_pattern='G', **common)
    mla_g = GPTConfig(attn_type='mla', attn_pattern='G', **common)
    mha_s = GPTConfig(attn_type='mha', attn_pattern='LLLLLG', **common)
    mla_s = GPTConfig(attn_type='mla', attn_pattern='LLLLLG', **common)
    assert mla_g.kv_lora_rank == 256 and mla_g.qk_rope_head_dim == 32

    for T, expected in [(1024, (18.8, 37.5, 7.0)),
                        (4096, (18.8, 21.9, 4.1)),
                        (16384, (18.8, 18.0, 3.4)),
                        (65536, (18.8, 17.0, 3.2))]:
        base = CacheSpec.from_config(mha_g, T).elements(T)
        pct = tuple(round(100 * CacheSpec.from_config(c, T).elements(T) / base, 1)
                    for c in (mla_g, mha_s, mla_s))
        assert pct == expected, f"T={T}: got {pct}, documented {expected}"


def test_occupied_never_exceeds_allocated():
    cfg = GPTConfig(n_layer=4, n_head=8, n_embd=512, attn_type='mla',
                    attn_pattern='LLLG', window_size=64)
    spec = CacheSpec.from_config(cfg, 512)
    cache = HybridKVCache(spec, batch_size=2, device='cpu', dtype=torch.float32)
    for pos in (0, 10, 64, 100, 512):
        cache.pos = pos
        assert cache.occupied_elements() <= cache.n_elements()
