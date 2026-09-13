"""Audit M-2 - the init of the MLA up-projections, and the 'matched' alternative.

The default ('bottleneck', std = r^-1/2, what every recorded run used) gives MLA's k and v
2.2x the init std of MHA's for the same unit-RMS input, so the MLA-vs-MHA loss gap of the
grid mixes the parameterisation with an init scale. mla_up_init='matched' removes that.
"""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig  # noqa: E402

BASE = dict(n_layer=8, n_head=16, n_embd=512, vocab_size=50304, bias=False, pos_encoding='rope')


def kv_std_ratio(mla_up_init):
    """std of k and v at init, MLA over MHA, for the same unit-RMS input."""
    torch.manual_seed(0)
    mha = GPT(GPTConfig(attn_type='mha', **BASE)).transformer.h[0].attn
    mla = GPT(GPTConfig(attn_type='mla', kv_lora_rank=256, mla_up_init=mla_up_init,
                        **BASE)).transformer.h[0].attn
    x = F.layer_norm(torch.randn(8, 256, 512), (512,))
    with torch.no_grad():
        _, k, v = mha.c_attn(x).split(512, dim=2)
        kv = mla.kv_up(mla.kv_norm(mla.kv_down(x)[..., :256])).view(8, 256, 16, 64)
    return (kv[..., :32].std() / k.std()).item(), (kv[..., 32:].std() / v.std()).item()


def test_default_is_the_recorded_bottleneck_init():
    assert GPTConfig().mla_up_init == 'bottleneck'
    r_k, r_v = kv_std_ratio('bottleneck')
    assert r_k > 2.0 and r_v > 2.0, (r_k, r_v)


def test_matched_init_equalises_k_and_v_with_mha():
    r_k, r_v = kv_std_ratio('matched')
    assert abs(r_k - 1) < 0.05 and abs(r_v - 1) < 0.05, (r_k, r_v)


def test_the_choice_changes_only_the_up_projections():
    """Same RNG stream: every other weight is identical, and the up-projections differ by
    exactly the ratio of the two stds."""
    models = {}
    for mode in ('bottleneck', 'matched'):
        torch.manual_seed(0)
        models[mode] = GPT(GPTConfig(attn_type='mla', kv_lora_rank=256, q_lora_rank=128,
                                     mla_up_init=mode, **dict(BASE, n_layer=2)))
    a, b = (dict(m.named_parameters()) for m in models.values())
    for name, pa in a.items():
        if name.endswith('kv_up.weight') or name.endswith('q_up.weight'):
            fan_in = pa.shape[1]
            expected = 0.02 * math.sqrt(512 / fan_in) / fan_in ** -0.5
            assert torch.allclose(b[name], pa * expected, atol=1e-6), name
        else:
            assert torch.equal(pa, b[name]), name


def test_unknown_value_is_refused():
    with pytest.raises(AssertionError):
        GPTConfig(attn_type='mla', mla_up_init='xavier')
