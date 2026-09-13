"""GQA without the per-step copy: k and v reach attend() with n_kv_head heads.

GroupedQueryAttention used to repeat_interleave k and v up to n_head right after reading the
cache, i.e. to copy the whole cached context at every decode step -- a defect
confined to the GQA cell: at T=32768, B=64 its decode ran at 0.40x MHA while doing the same
attention on a cache four times smaller. attend() now takes fewer key/value heads and hands
them to the kernel natively (SDPA and flex with enable_gqa, flash-attn's own GQA). The one
exception is sdpa_mask WITH a mask: SDPA's GQA runs only on the flash and math kernels, flash
takes no mask, so enable_gqa there would fall to math and its (B, heads, T, S) weights. That
path repeats, once per prefill, never per decode step.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_cache import CacheSpec, HybridKVCache  # noqa: E402
from model import GPT, GPTConfig, attend  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def qkv(heads=8, kv_heads=2, T=32, S=32, D=16, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda h, n: torch.randn(2, h, n, D, generator=g).to(device)
    return mk(heads, T), mk(kv_heads, S), mk(kv_heads, S)


def rep(t, n=4):
    return t.repeat_interleave(n, dim=1)


# (T, S, is_local, window): prefill with a mask, then decode steps that need none
CASES = [(32, 32, False, None), (32, 32, True, 8), (1, 32, False, None), (1, 8, True, 8)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", ["sdpa_mask", "flex"])
@pytest.mark.parametrize("T,S,is_local,window", CASES)
def test_native_gqa_matches_the_repeated_kv(device, impl, T, S, is_local, window):
    q, k, v = qkv(T=T, S=S, device=device)
    kw = dict(is_local=is_local, window=window, impl=impl)
    ref = attend(q, rep(k), rep(v), **kw)
    out = attend(q, k, v, **kw)
    assert torch.allclose(ref, out, atol=1e-5), (ref - out).abs().max().item()


@pytest.mark.parametrize("impl", ["sdpa_mask", "flex"])
def test_native_gqa_gradients_match_the_repeated_kv(impl):
    """attend() is also the training path of the GQA cell."""
    q, k, v = qkv()
    grads = []
    for native in (False, True):
        qq, kk, vv = (t.clone().requires_grad_() for t in (q, k, v))
        out = attend(qq, kk if native else rep(kk), vv if native else rep(vv),
                     is_local=True, window=8, impl=impl)
        out.square().sum().backward()
        grads.append([t.grad for t in (qq, kk, vv)])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()


def test_attend_refuses_query_heads_that_are_not_a_multiple_of_the_kv_heads():
    q, k, v = qkv(heads=8, kv_heads=3)
    with pytest.raises(AssertionError, match="multiple"):
        attend(q, k, v, impl="sdpa_mask")


def gqa(impl, pattern, window=16, device="cpu", dtype=torch.float32, **over):
    # head_dim 16: flex pads narrower heads to 16 and then needs an explicit scale, which
    # neither MHA nor GQA passes -- a limit of the flex path that predates this file
    cfg = GPTConfig(**dict(dict(block_size=128, vocab_size=97, n_layer=2, n_head=8, n_kv_head=2,
                                n_embd=128, dropout=0.0, bias=False, pos_encoding='rope',
                                attn_type='gqa', attn_pattern=pattern, attn_impl=impl,
                                window_size=window), **over))
    torch.manual_seed(0)
    return cfg, GPT(cfg).to(device, dtype).eval()


def cache_for(cfg, B, max_seq_len, device="cpu", dtype=torch.float32):
    return HybridKVCache(CacheSpec.from_config(cfg, max_seq_len), B, device, dtype)


@pytest.mark.parametrize("impl", ["sdpa_mask", "flex"])
@pytest.mark.parametrize("pattern", ["G", "LG"])
def test_a_decode_step_never_repeats_the_cached_kv(monkeypatch, impl, pattern):
    cfg, model = gqa(impl, pattern)
    cache = cache_for(cfg, 2, 64)
    seq = torch.randint(0, 97, (2, 40), generator=torch.Generator().manual_seed(0))
    calls = []
    repeat = torch.Tensor.repeat_interleave

    def spy(self, *a, **k):
        calls.append(tuple(self.shape))
        return repeat(self, *a, **k)

    with torch.no_grad():
        model(seq[:, :20], cache=cache)                    # prefill
        monkeypatch.setattr(torch.Tensor, "repeat_interleave", spy)
        for t in range(20, 40):                            # the local layer wraps (W=16)
            model(seq[:, t:t + 1], cache=cache)
    assert calls == [], f"a decode step repeated k/v: {calls[:4]}"


@CUDA
def test_decode_step_memory_does_not_grow_with_a_repeated_context():
    """At S=16384, B=8 one repeated k is 8*16*16384*32*2 bytes = 128 MiB, and the old code
    made two of them per layer per step. A whole decode step must now peak below a quarter
    of ONE of them above what is already resident."""
    cfg, model = gqa('sdpa_mask', 'G', device="cuda", dtype=torch.bfloat16, n_layer=1,
                     n_head=16, n_kv_head=4, n_embd=512, block_size=1024)
    B, S = 8, 16384
    cache = cache_for(cfg, B, S + 2, "cuda", torch.bfloat16)
    cache.advance(S)
    idx = torch.randint(0, 97, (B, 1), device="cuda")
    with torch.no_grad():
        model(idx, cache=cache)                            # warm-up: kernels, rope tables
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        model(idx, cache=cache)
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    repeated_k = B * 16 * S * 32 * 2
    assert peak < repeated_k / 4, (
        f"a decode step peaked {peak / 2**20:.1f} MiB above the resident memory; "
        f"one repeated k is {repeated_k / 2**20:.1f} MiB")


@CUDA
def test_compiled_flex_gqa_matches_the_repeated_kv_forward_and_backward():
    """The grid trains the GQA cell with compiled flex, which now receives 4 kv heads."""
    q, k, v = qkv(heads=8, kv_heads=2, T=256, S=256, D=32, device="cuda")
    compiled = torch.compile(attend)
    grads = []
    outs = []
    for native, fn in ((False, attend), (True, compiled)):
        qq, kk, vv = (t.clone().requires_grad_() for t in (q, k, v))
        out = fn(qq, kk if native else rep(kk), vv if native else rep(vv),
                 is_local=True, window=64, impl="flex" if native else "sdpa_mask")
        out.square().sum().backward()
        outs.append(out.detach())
        grads.append([t.grad for t in (qq, kk, vv)])
    assert torch.allclose(outs[0], outs[1], atol=1e-4)
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-3), (a - b).abs().max().item()
