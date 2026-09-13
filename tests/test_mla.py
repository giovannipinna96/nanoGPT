"""Gates T3.1 and T3.2 - MLA shapes and differentiability (test_todo.md Fase 3).

T3.3 (full-rank MLA ~ MHA), T3.4 (no NaN) and T3.5 (parameter matching) need training
and belong to STEP 7.
"""
import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig, MultiHeadLatentAttention  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def cfg(**kw):
    base = dict(block_size=256, vocab_size=65, n_layer=2, n_head=4, n_embd=128,
                dropout=0.0, bias=False, attn_type='mla', pos_encoding='rope',
                attn_impl='sdpa_mask')
    base.update(kw)
    return GPTConfig(**base)


# ------------------------------------------------------------------ T3.1
@pytest.mark.parametrize(
    "q_lora_rank,is_local,B,T",
    list(itertools.product([None, 96], [False, True], [1, 4], [1, 7, 256])))
def test_shapes_on_every_branch(q_lora_rank, is_local, B, T):
    """T=1 and T=7 (not a power of two) catch reshape bugs that T=256 hides."""
    c = cfg(q_lora_rank=q_lora_rank, attn_pattern='LG' if is_local else 'G',
            window_size=8)
    m = GPT(c)
    idx = torch.randint(0, c.vocab_size, (B, T))
    logits, loss = m(idx, idx)
    assert logits.shape == (B, T, c.vocab_size)
    assert loss.ndim == 0


def test_cacheable_tensors_have_the_documented_shapes():
    """k^R must be (B, 1, T, d_rope): one shared channel, not one per head (M2).

    If it were per head the KV cache would be d_c + n_h*d^R instead of d_c + d^R,
    i.e. more than 2x the headline number of the report.
    """
    c = cfg()
    mla = MultiHeadLatentAttention(c, 0)
    x = torch.randn(2, 16, c.n_embd)
    kv = mla.kv_down(x)
    kv_lat, k_rope = kv.split([c.kv_lora_rank, c.qk_rope_head_dim], dim=-1)
    assert kv_lat.shape == (2, 16, c.kv_lora_rank)
    assert k_rope.shape == (2, 16, c.qk_rope_head_dim)
    assert k_rope.unsqueeze(1).shape == (2, 1, 16, c.qk_rope_head_dim)


def test_scale_comes_from_the_concatenated_dim():
    """threats.md M1: 1/sqrt(d_nope + d_rope), never 1/sqrt(head_dim)."""
    c = cfg(n_embd=128, n_head=4)          # head_dim = 32
    mla = MultiHeadLatentAttention(c, 0)
    assert c.qk_nope_head_dim == 32 and c.qk_rope_head_dim == 16
    assert mla.scale == pytest.approx((32 + 16) ** -0.5)
    assert mla.scale != pytest.approx(32 ** -0.5)


def test_rope_dims_are_additive_not_carved_out():
    """remediation.md #5: qk_nope is the FULL head_dim, qk_rope is extra."""
    c = cfg(n_embd=512, n_head=8)          # head_dim = 64
    assert c.qk_nope_head_dim == 64        # full, same content capacity as MHA
    assert c.qk_rope_head_dim == 32        # additive
    assert c.v_head_dim == 64
    assert c.kv_lora_rank == 256           # 4 * head_dim (remediation.md #7)


def test_joint_compression_single_latent():
    """threats.md M3: one latent feeds both k and v, not two separate latents."""
    c = cfg()
    mla = MultiHeadLatentAttention(c, 0)
    assert mla.kv_down.out_features == c.kv_lora_rank + c.qk_rope_head_dim
    assert mla.kv_up.in_features == c.kv_lora_rank
    assert mla.kv_up.out_features == c.n_head * (c.qk_nope_head_dim + c.v_head_dim)


def test_the_latent_is_scale_invariant_which_is_what_the_rmsnorm_buys():
    """threats.md M5: the RMSNorm on the compressed latent is invisible in every shape, so
    pin the one property it provides -- the latent that rebuilds k^C and v does not depend
    on the SCALE of x. Without it the low-rank bottleneck amplifies the variance and MLA
    training diverges, and nothing else in tests/ notices its removal.

    Only the latent. q and the decoupled k^R are NOT normalised (eq. 14-15, and
    q_lora_rank is None by default), so the sublayer OUTPUT does scale with x: asserting
    invariance there would be asserting something false.
    """
    torch.manual_seed(0)
    c = cfg()
    mla = MultiHeadLatentAttention(c, 0).double()
    x = torch.randn(2, 16, c.n_embd, dtype=torch.float64)

    def latent(z):
        return mla.kv_norm(mla.kv_down(z)[..., :c.kv_lora_rank])

    assert latent(x).abs().max().item() > 1e-3, "a latent of zeros would pass vacuously"
    assert (latent(x) - latent(5.0 * x)).abs().max().item() < 1e-5


def test_the_scale_reaches_the_attention_not_just_the_attribute():
    """M1 end to end. test_scale_comes_from_the_concatenated_dim checks the ATTRIBUTE; this
    checks it is the number the attention actually uses, so a refactor that stops passing
    `scale=` to attend() fails here instead of quietly training a worse MLA. The value
    substituted below is exactly the M1 mistake: 1/sqrt(head_dim) instead of 1/sqrt(d_qk).
    """
    torch.manual_seed(0)
    c = cfg(n_layer=1)
    m = GPT(c).eval()
    idx = torch.randint(0, c.vocab_size, (2, 32))
    with torch.no_grad():
        correct = m(idx)[0]
        m.transformer.h[0].attn.scale = (c.n_embd // c.n_head) ** -0.5
        wrong = m(idx)[0]
    assert (correct - wrong).abs().max().item() > 1e-4


def test_output_projection_is_marked_for_scaled_init():
    """threats.md M8: nanoGPT matches on the NAME c_proj; MLA needs an explicit flag."""
    mla = MultiHeadLatentAttention(cfg(), 0)
    assert getattr(mla.o_proj, '_is_residual_proj', False) is True
    assert mla.o_proj.in_features == cfg().n_head * cfg().v_head_dim


@pytest.mark.parametrize("device", DEVICES)
def test_flex_backend_handles_asymmetric_qk_and_v_dims(device):
    """D_qk=96 and D_v=64 differ in MLA; the grid backend must cope with that."""
    c = cfg(n_embd=512, n_head=8, attn_impl='flex', block_size=128)
    m = GPT(c).to(device)
    idx = torch.randint(0, c.vocab_size, (2, 128), device=device)
    logits, _ = m(idx, idx)
    assert logits.shape == (2, 128, c.vocab_size)


# ------------------------------------------------------------------ T3.2
def test_gradcheck_in_float64():
    torch.manual_seed(0)
    c = cfg(n_embd=32, n_head=2, kv_lora_rank=8, qk_nope_head_dim=8,
            qk_rope_head_dim=4, v_head_dim=8, block_size=8)
    mla = MultiHeadLatentAttention(c, 0).double()
    from model import precompute_rope
    cos, sin = precompute_rope(c.qk_rope_head_dim, 8)
    x = torch.randn(1, 4, c.n_embd, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda inp: mla(inp, cos[:4].double(), sin[:4].double()),
        (x,), eps=1e-6, atol=1e-4)


def test_every_parameter_receives_a_gradient():
    torch.manual_seed(0)
    c = cfg(q_lora_rank=96)
    m = GPT(c)
    idx = torch.randint(0, c.vocab_size, (2, 32))
    _, loss = m(idx, idx)
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_flex_padding_is_exact_against_the_dense_oracle():
    """The 96 -> 128 zero-padding used for FlexAttention must not change any number.

    Zero channels contribute zero to every dot product, and the scale is passed
    explicitly so it stays 1/sqrt(96) rather than being recomputed from the padded
    width (which would silently reintroduce threat M1).
    """
    from model import attend
    torch.manual_seed(0)
    B, H, T, D_qk, D_v = 2, 4, 64, 96, 64
    q = torch.randn(B, H, T, D_qk)
    k = torch.randn(B, H, T, D_qk)
    v = torch.randn(B, H, T, D_v)
    scale = D_qk ** -0.5
    for is_local, window in [(False, None), (True, 16)]:
        ref = attend(q, k, v, is_local=is_local, window=window, impl="sdpa_mask",
                     scale=scale)
        out = attend(q, k, v, is_local=is_local, window=window, impl="flex", scale=scale)
        assert out.shape == ref.shape == (B, H, T, D_v)
        assert torch.allclose(ref, out, atol=1e-4), (ref - out).abs().max().item()


def test_mla_matches_between_backends_end_to_end():
    torch.manual_seed(0)
    c = cfg(n_embd=512, n_head=8, block_size=128, attn_pattern='LLLG', window_size=32,
            attn_impl='sdpa_mask')
    m = GPT(c).eval()
    idx = torch.randint(0, c.vocab_size, (2, 128))
    with torch.no_grad():
        ref, _ = m(idx, idx)
        for blk in m.transformer.h:
            blk.attn.attn_impl = 'flex'
        out, _ = m(idx, idx)
    assert torch.allclose(ref, out, atol=1e-3), (ref - out).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile path needs CUDA")
def test_mla_under_torch_compile():
    """Regression test for the padded-output bug.

    In eager mode flex_attention returned a d_v-wide output; under torch.compile it
    returned the PADDED width, so the reshape in MLA.forward blew up only in a real
    compiled training run. Every eager test passed.
    """
    torch.manual_seed(0)
    c = cfg(n_embd=384, n_head=6, block_size=128, attn_impl='flex',
            attn_pattern='LLLG', window_size=32)
    m = GPT(c).cuda()
    mc = torch.compile(m)
    idx = torch.randint(0, c.vocab_size, (4, 128), device='cuda')
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss = mc(idx, idx)
    assert logits.shape == (4, 128, c.vocab_size)
    assert torch.isfinite(loss)
