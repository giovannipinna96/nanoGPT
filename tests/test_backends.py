"""Gate T1.2 - the three attention backends must agree (test_todo.md Fase 1).

`sdpa_mask` is the oracle: a dense boolean mask, correct by construction and slow.
`flex` is what every grid cell uses. If they disagree, the usual cause is a mask_mod
with `>` where it should be `>=`.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import attend, get_dense_mask  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def qkv(B=2, H=4, T=128, D=32, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda: torch.randn(B, H, T, D, generator=g, dtype=torch.float32).to(device)
    return mk(), mk(), mk()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("is_local,window", [(False, None), (True, 32), (True, 1)])
def test_flex_matches_dense_oracle(device, is_local, window):
    if device == "cpu":
        pytest.importorskip("torch.nn.attention.flex_attention")
    q, k, v = qkv(device=device)
    ref = attend(q, k, v, is_local=is_local, window=window, impl="sdpa_mask")
    out = attend(q, k, v, is_local=is_local, window=window, impl="flex")
    assert torch.allclose(ref, out, atol=1e-3), (ref - out).abs().max().item()


@pytest.mark.parametrize("device", DEVICES)
def test_dense_mask_never_leaves_a_row_empty(device):
    # a fully masked row gives softmax(-inf) = NaN and poisons the whole batch (S4)
    for is_local, window in [(False, None), (True, 1), (True, 7)]:
        m = get_dense_mask(64, 64, is_local, window, device)
        assert (m.sum(-1) > 0).all()
