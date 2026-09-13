"""Gates T2.1, T2.2, T2.4 - sliding window attention (phase 2).

T2.3 (real sparsity) is a wall-clock measurement and lives in analysis/sweep_window.sh,
not here: it needs a GPU and a long context to produce a readable signal.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
from model import attend, get_dense_mask, mask_build_count, reset_mask_cache  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
IMPLS = ["sdpa_mask", "flex"]


def qkv(B=2, H=4, T=64, D=32, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda: torch.randn(B, H, T, D, generator=g, dtype=torch.float32).to(device)
    return mk(), mk(), mk()


# ------------------------------------------------------------------ T2.1
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("window_extra", [0, 1, 10])
def test_degenerate_window_equals_causal(device, impl, window_extra):
    """A window >= T must reproduce full causal attention exactly.

    This single test catches roughly 90% of off-by-one errors: `q - kv < W` versus
    `<= W` shifts the window by one token, the model still trains, the loss is only
    slightly worse, and nothing ever tells you.
    """
    T = 64
    q, k, v = qkv(T=T, device=device)
    y_global = attend(q, k, v, is_local=False, impl=impl)
    y_local = attend(q, k, v, is_local=True, window=T + window_extra, impl=impl)
    assert torch.allclose(y_global, y_local, atol=1e-6), \
        (y_global - y_local).abs().max().item()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", IMPLS)
def test_window_smaller_than_T_actually_changes_the_output(device, impl):
    """The mirror image of T2.1: if a small window changes nothing, the mask is dead.

    This is the shape of threat K1: is_causal=True silently ignores attn_mask, the
    sliding window never exists, the loss looks great and the conclusion is inverted.
    """
    q, k, v = qkv(T=64, device=device)
    y_global = attend(q, k, v, is_local=False, impl=impl)
    y_local = attend(q, k, v, is_local=True, window=8, impl=impl)
    assert (y_global - y_local).abs().max().item() > 1e-3


# ------------------------------------------------------------------ T2.2
@pytest.mark.parametrize("W", [1, 4, 7, 16])
def test_visible_count_per_row(W):
    """Row i must see exactly min(i+1, W) positions, initial boundary included."""
    T = 16
    m = get_dense_mask(T, T, is_local=True, window=W, device="cpu")
    for i in range(T):
        assert m[i].sum().item() == min(i + 1, W), f"row {i}: {m[i].sum().item()}"


def test_causal_row_counts():
    m = get_dense_mask(16, 16, is_local=False, window=None, device="cpu")
    for i in range(16):
        assert m[i].sum().item() == i + 1


def test_window_convention_is_w_not_w_plus_one():
    """W means W tokens INCLUDING self, i.e. q - kv < W (docstring of _mask_mod)."""
    m = get_dense_mask(8, 8, is_local=True, window=3, device="cpu")
    assert m[5].tolist() == [False, False, False, True, True, True, False, False]


def test_decoding_offset_is_respected():
    """During decoding the query sits at an absolute position past the cached keys."""
    m = get_dense_mask(1, 10, is_local=True, window=4, device="cpu", q_offset=9)
    assert m[0].tolist() == [False] * 6 + [True] * 4


# ------------------------------------------------------------------ T2.4
def test_block_mask_is_built_once_per_shape():
    reset_mask_cache()
    q, k, v = qkv(T=64)
    for _ in range(5):
        attend(q, k, v, is_local=True, window=16, impl="flex")
    assert mask_build_count() == 1, "the block mask must be cached"
    for _ in range(5):
        attend(q, k, v, is_local=False, impl="flex")
    assert mask_build_count() == 2, "a different shape signature builds exactly once more"
    reset_mask_cache()
