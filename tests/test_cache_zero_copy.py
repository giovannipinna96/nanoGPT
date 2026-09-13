"""Reading the cache must not copy it while the buffer has not wrapped.

The read used to gather every layer's buffer at every decode step, global layers included,
which never wrap. On the dense-cache cells that copy was most of the decode time (MHA at
T=8192, B=64: 53.3 ms/token with it, 18.5 without, identical logits), so every decode
speed-up measured against MHA was inflated. Correctness of the read is covered by the
parity gates T4.1/T4.2; this file pins the zero-copy property itself.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_cache import CacheSpec, HybridKVCache  # noqa: E402


def cache(attn_type, is_local, window=4, max_seq_len=32):
    spec = CacheSpec(n_layer=len(is_local), is_local=is_local, window_size=window,
                     max_seq_len=max_seq_len, attn_type=attn_type, n_kv_head=2, head_dim=8,
                     v_head_dim=8, kv_lora_rank=8, qk_rope_head_dim=4)
    return HybridKVCache(spec, batch_size=2, device="cpu", dtype=torch.float32)


def write_tokens(layer, first, n):
    """n tokens whose every channel holds the token's absolute position."""
    tensors = {}
    for name, buf in layer.buffers.items():
        shape = (buf.size(0), n, *buf.shape[2:])
        pos = torch.arange(first, first + n, dtype=torch.float32)
        tensors[name] = pos.view(1, n, *[1] * (len(shape) - 2)).expand(shape).clone()
    layer.write(first, **tensors)


@pytest.mark.parametrize("attn_type", ["mha", "mla"])
def test_read_before_wrap_is_a_view_of_the_buffer(attn_type):
    c = cache(attn_type, is_local=[True, False])
    for layer in c.layers:
        write_tokens(layer, 0, 3)                           # 3 <= capacity on both layers
        out, start = layer.read(3)
        assert start == 0
        for name, buf in layer.buffers.items():
            assert out[name].data_ptr() == buf.data_ptr(), f"{name} was copied"
            assert out[name].shape[1] == 3
            assert out[name][0, :, ...].reshape(3, -1)[:, 0].tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("attn_type", ["mha", "mla"])
def test_read_after_wrap_is_chronological(attn_type):
    c = cache(attn_type, is_local=[True], window=4)
    layer = c.layers[0]
    for p in range(10):                                     # wraps twice
        write_tokens(layer, p, 1)
    out, start = layer.read(10)
    assert start == 6
    for name in layer.buffers:
        assert out[name][0].reshape(4, -1)[:, 0].tolist() == [6.0, 7.0, 8.0, 9.0]
