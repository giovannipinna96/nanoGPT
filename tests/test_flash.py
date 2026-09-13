"""STEP 0 - the flash-attn 2 backend, and the symmetric head dims that make it usable.

Why this file exists. `impl='flash'` carries the sliding window inside the kernel, so it
pays neither of the two taxes of the flex path: no BlockMask to build (threats.md K2) and
no power-of-two padding of the head dim. The second one is only reachable for MLA if
d_qk == d_v, which is what GPTConfig(symmetric_head_dims=True) arranges.

Two groups of tests:

  * configuration-level, CPU, always run: the flag does what it claims and refuses the
    contradictory combinations;
  * kernel-level, CUDA + `uv sync --extra flash`, skipped otherwise: the flash path
    reproduces the dense oracle, on both a global and a local layer, for MHA, GQA and
    symmetric MLA.

    uv run --extra flash pytest tests/test_flash.py -v      # on a GPU node
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig, _pad_to_pow2, attend  # noqa: E402

HAS_FLASH = torch.cuda.is_available()
if HAS_FLASH:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        HAS_FLASH = False
needs_flash = pytest.mark.skipif(
    not HAS_FLASH, reason="needs CUDA and `uv sync --extra flash`")


# --------------------------------------------------------------- configuration level

def test_symmetric_head_dims_equalise_qk_and_v():
    cfg = GPTConfig(n_embd=512, n_head=16, attn_type='mla', symmetric_head_dims=True)
    assert cfg.v_head_dim == cfg.qk_nope_head_dim + cfg.qk_rope_head_dim


def test_default_is_unchanged():
    """The flag is opt-in: every number already recorded must stay reproducible."""
    cfg = GPTConfig(n_embd=512, n_head=16, attn_type='mla')
    assert cfg.symmetric_head_dims is False
    assert cfg.v_head_dim == 512 // 16                  # head_dim, as before
    assert cfg.qk_nope_head_dim + cfg.qk_rope_head_dim == 48 != cfg.v_head_dim


def test_symmetric_head_dims_removes_the_padding_flex_would_do():
    """The waste flex inflicts is on the SCORE width only: q and k go 48 -> 64, while
    v = 32 is already a power of two and is left alone. Flash needs no padding at all,
    but only once d_v is widened to match d_qk."""
    default = GPTConfig(n_embd=512, n_head=16, attn_type='mla')
    sym = GPTConfig(n_embd=512, n_head=16, attn_type='mla', symmetric_head_dims=True)
    d_qk = default.qk_nope_head_dim + default.qk_rope_head_dim
    assert _pad_to_pow2(d_qk) == 64 != d_qk             # flex pads q/k 48 -> 64
    assert _pad_to_pow2(default.v_head_dim) == default.v_head_dim   # v is untouched
    # flash asks for one width, a multiple of 8, at most 256 -- and gets it
    assert sym.v_head_dim == d_qk and d_qk % 8 == 0 and d_qk <= 256
    # and on the symmetric config flex would now pad BOTH sides, which is the reason
    # symmetric head dims only pay off together with impl='flash'
    assert _pad_to_pow2(sym.v_head_dim) != sym.v_head_dim


def test_symmetric_head_dims_is_mla_only():
    with pytest.raises(AssertionError, match="MLA-only"):
        GPTConfig(attn_type='mha', symmetric_head_dims=True)


def test_symmetric_head_dims_refuses_a_contradictory_v_head_dim():
    with pytest.raises(AssertionError, match="drop one of the two"):
        GPTConfig(n_embd=512, n_head=16, attn_type='mla',
                  symmetric_head_dims=True, v_head_dim=32)


def test_symmetric_head_dims_leave_the_kv_cache_untouched():
    """MLA caches c_KV and k^R, never v: the headline memory table cannot move."""
    from kv_cache import CacheSpec

    def spec(cfg):
        return CacheSpec.from_config(cfg, 4096)

    base = dict(n_embd=512, n_head=16, n_layer=8, attn_type='mla', attn_pattern='LLLG')
    a = spec(GPTConfig(**base))
    b = spec(GPTConfig(**base, symmetric_head_dims=True))
    assert a.elements(4096) == b.elements(4096)


def test_flash_refuses_asymmetric_head_dims():
    """The default MLA shape must fail loudly on flash, not silently reinterpret."""
    pytest.importorskip("flash_attn")
    q = torch.zeros(1, 2, 8, 48, dtype=torch.bfloat16)
    v = torch.zeros(1, 2, 8, 32, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="same head dim"):
        attend(q, q, v, impl="flash", scale=48 ** -0.5)


# --------------------------------------------------------------------- kernel level

def qkv(B=2, H=4, T=128, D=48, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda: torch.randn(B, H, T, D, generator=g, dtype=torch.float32).to(device)
    return mk(), mk(), mk()


@needs_flash
@pytest.mark.parametrize("is_local,window", [(False, None), (True, 32), (True, 1)])
def test_flash_matches_dense_oracle(is_local, window):
    q, k, v = qkv()
    ref = attend(q, k, v, is_local=is_local, window=window, impl="sdpa_mask")
    out = attend(q.bfloat16(), k.bfloat16(), v.bfloat16(),
                 is_local=is_local, window=window, impl="flash")
    err = (ref - out.float()).abs().max().item()
    assert err < 3e-2, err


@needs_flash
def test_flash_window_is_w_not_w_plus_one():
    """The kernel counts `left` tokens BEFORE self, this file counts W INCLUDING self.
    If the -1 were dropped, W=2 on flash would equal W=3 on the oracle and the whole
    sliding-window story would be off by one token per layer (threats.md S1)."""
    q, k, v = qkv(T=64)
    out = attend(q.bfloat16(), k.bfloat16(), v.bfloat16(),
                 is_local=True, window=2, impl="flash").float()
    same = attend(q, k, v, is_local=True, window=2, impl="sdpa_mask")
    off_by_one = attend(q, k, v, is_local=True, window=3, impl="sdpa_mask")
    assert (same - out).abs().max() < 3e-2
    assert (off_by_one - out).abs().max() > 3e-2


@needs_flash
@pytest.mark.parametrize("cell", [
    dict(attn_type='mha', attn_pattern='G'),
    dict(attn_type='mha', attn_pattern='LLLG'),
    dict(attn_type='gqa', attn_pattern='G', n_kv_head=4),
    dict(attn_type='mla', attn_pattern='LLLG', symmetric_head_dims=True),
])
def test_model_forward_on_flash_matches_the_oracle(cell):
    """End to end: same weights, same batch, two backends, one loss."""
    torch.manual_seed(1337)
    base = dict(n_layer=4, n_head=16, n_embd=512, block_size=256, window_size=64,
                vocab_size=512, dropout=0.0, bias=False, pos_encoding='rope')
    model = GPT(GPTConfig(**base, **cell, attn_impl='sdpa_mask')).cuda().eval()
    idx = torch.randint(0, 512, (2, 128), device='cuda')
    targets = torch.randint(0, 512, (2, 128), device='cuda')

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        _, ref = model(idx, targets)
        for block in model.transformer.h:
            block.attn.attn_impl = 'flash'
        _, out = model(idx, targets)
    assert abs(ref.item() - out.item()) < 5e-3, (ref.item(), out.item())


@needs_flash
def test_flash_unlocks_the_head_dim_64_configuration():
    """Configuration A (n_head=16, head_dim=32) was not a modelling choice: head_dim=64
    gives d_qk = 64 + 32 = 96, which flex pads to 128 and whose BACKWARD does not compile
    through FlexAttention on sm80 (config/grid_base.py header). On the
    flash path 96 is a perfectly legal head dim -- multiple of 8, under 256, no padding --
    so the configuration that had to be abandoned becomes reachable again.

    The test compiles and runs a backward at that width, because "does not compile" is
    exactly the claim being lifted."""
    cfg = GPTConfig(n_layer=2, n_head=8, n_embd=512, block_size=128, window_size=32,
                    vocab_size=256, dropout=0.0, bias=False, pos_encoding='rope',
                    attn_type='mla', attn_pattern='LG', symmetric_head_dims=True,
                    attn_impl='flash')
    assert cfg.qk_nope_head_dim + cfg.qk_rope_head_dim == 96 == cfg.v_head_dim
    model = torch.compile(GPT(cfg).cuda())
    idx = torch.randint(0, 256, (2, 128), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, loss = model(idx, idx)
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if 'kv_up' in n or 'kv_down' in n]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
