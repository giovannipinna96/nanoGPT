"""A GLOBAL layer must refuse to wrap its buffer.

A local layer's ring buffer is supposed to wrap: it keeps the last W tokens, which is the
window. A global layer's buffer wrapping means it silently keeps only the last max_seq_len
tokens, i.e. the model turns into a sliding-window model with no error. It happened in the
repository's own decode benchmark, which sized the cache for prefill + timed steps and then
also ran the warmup steps.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_cache import CacheSpec, HybridKVCache  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402


def make(pattern, window=4, max_seq_len=16, force_last_global=True):
    cfg = GPTConfig(block_size=64, vocab_size=97, n_layer=2, n_head=4, n_embd=64, dropout=0.0,
                    bias=False, pos_encoding='rope', attn_impl='sdpa_mask', attn_type='mha',
                    attn_pattern=pattern, window_size=window, force_last_global=force_last_global)
    spec = CacheSpec.from_config(cfg, max_seq_len)
    torch.manual_seed(0)
    return GPT(cfg).eval(), HybridKVCache(spec, 1, 'cpu', torch.float32)


def test_decode_past_global_capacity_raises():
    model, cache = make('LG', max_seq_len=16)
    seq = torch.randint(0, 97, (1, 17))
    with torch.no_grad():
        model(seq[:, :8], cache=cache)
        for t in range(8, 16):                      # fills the global buffer exactly
            model(seq[:, t:t + 1], cache=cache)
        with pytest.raises(ValueError, match="global"):
            model(seq[:, 16:17], cache=cache)


def test_prefill_longer_than_global_capacity_raises():
    model, cache = make('LG', max_seq_len=16)
    with torch.no_grad(), pytest.raises(ValueError, match="global"):
        model(torch.randint(0, 97, (1, 20)), cache=cache)


def test_local_layers_still_wrap_freely():
    """All-local: every buffer is a window, so decoding far past its size is legal."""
    model, cache = make('L', window=4, max_seq_len=16, force_last_global=False)
    seq = torch.randint(0, 97, (1, 40))
    with torch.no_grad():
        model(seq[:, :8], cache=cache)
        for t in range(8, 40):
            model(seq[:, t:t + 1], cache=cache)
    assert cache.pos == 40
