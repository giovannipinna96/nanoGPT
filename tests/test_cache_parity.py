"""Gates T4.1, T4.2, T4.3 - the cache must be indistinguishable from a full forward.

A bug here does not crash: it produces benchmark numbers that are plausible and false.
If T4.1 fails, the first suspect (9 times out of 10) is RoPE applied with the
rolling-buffer slot index instead of the absolute position -- the tell is that
generation stays sensible for the first W tokens and then degenerates (threats.md C1).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_cache import CacheSpec, HybridKVCache  # noqa: E402
from model import GPT, GPTConfig, resolve_pattern  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# the four cells of the grid, plus the all-local reading of "hybrid"
CELLS = {
    "1_mha_full": dict(attn_type='mha', attn_pattern='G'),
    "2_mha_swa": dict(attn_type='mha', attn_pattern='LLLG'),
    "3_mla_full": dict(attn_type='mla', attn_pattern='G'),
    "4_mla_swa": dict(attn_type='mla', attn_pattern='LLLG'),
    "5_gqa_full": dict(attn_type='gqa', attn_pattern='G', n_kv_head=2),
    "6_gqa_swa": dict(attn_type='gqa', attn_pattern='LLLG', n_kv_head=2),
    # the two symmetric-head-dim variants (STEP 0b). 'reconstructed' is the one that has
    # to earn this test: it rotates the keys REBUILT from the latent, so it is the only
    # mode where the key positions are recovered rather than frozen into the cached value
    # -- exactly the situation threat C1 describes.
    "4_mla_swa_carved": dict(attn_type='mla', attn_pattern='LLLG', rope_mode='carved'),
    "4_mla_swa_recon": dict(attn_type='mla', attn_pattern='LLLG', rope_mode='reconstructed'),
}


def make(cell, device, window=16, block_size=256):
    cfg = GPTConfig(block_size=block_size, vocab_size=97, n_layer=4, n_head=4,
                    n_embd=128, dropout=0.0, bias=False, pos_encoding='rope',
                    attn_impl='sdpa_mask', window_size=window, **CELLS[cell])
    torch.manual_seed(0)
    model = GPT(cfg).to(device).eval()
    return cfg, model


def make_cache(cfg, batch_size, max_seq_len, device):
    return HybridKVCache(CacheSpec.from_config(cfg, max_seq_len), batch_size, device,
                         dtype=torch.float32)


# ------------------------------------------------------------------ T4.1
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cell", list(CELLS))
def test_incremental_equals_full_forward(device, cell):
    """20 consecutive decode steps must match a full forward, for every cell."""
    torch.manual_seed(1)
    cfg, model = make(cell, device)
    prompt_len, n_steps = 24, 20
    seq = torch.randint(0, cfg.vocab_size, (2, prompt_len + n_steps), device=device)

    cache = make_cache(cfg, 2, prompt_len + n_steps, device)
    with torch.no_grad():
        model(seq[:, :prompt_len], cache=cache)          # prefill
        for i in range(n_steps):
            t = prompt_len + i
            inc, _ = model(seq[:, t:t + 1], cache=cache)  # one decode step
            full, _ = model(seq[:, :t + 1])               # the same token, no cache
            assert torch.allclose(full[:, -1], inc[:, -1], atol=1e-4), (
                f"{cell} step {i}: {(full[:, -1] - inc[:, -1]).abs().max().item():.2e}")


# ------------------------------------------------------------------ T4.2
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cell", ["2_mha_swa", "4_mla_swa", "6_gqa_swa",
                                  "4_mla_swa_carved", "4_mla_swa_recon"])
def test_correct_after_the_rolling_buffer_wraps(device, cell):
    """Generate 3*W tokens: the buffer wraps twice and must stay correct.

    This is the test that catches a rolling buffer read in slot order (C2) and RoPE
    keyed on the slot (C1): both are invisible before the first wrap.
    """
    torch.manual_seed(2)
    window = 16
    cfg, model = make(cell, device, window=window, block_size=256)
    n = 3 * window
    seq = torch.randint(0, cfg.vocab_size, (2, 8 + n), device=device)

    cache = make_cache(cfg, 2, 8 + n, device)
    with torch.no_grad():
        model(seq[:, :8], cache=cache)
        worst = 0.0
        for i in range(n):
            t = 8 + i
            inc, _ = model(seq[:, t:t + 1], cache=cache)
            full, _ = model(seq[:, :t + 1])
            worst = max(worst, (full[:, -1] - inc[:, -1]).abs().max().item())
            assert worst < 1e-4, f"{cell} diverged at step {i} (t={t}): {worst:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_prefill_longer_than_the_window_is_still_correct(device):
    """A prompt much longer than W: local layers keep only the last W entries."""
    torch.manual_seed(3)
    cfg, model = make("4_mla_swa", device, window=16, block_size=256)
    prompt_len = 100          # >> window
    seq = torch.randint(0, cfg.vocab_size, (2, prompt_len + 5), device=device)
    cache = make_cache(cfg, 2, prompt_len + 5, device)
    with torch.no_grad():
        pre, _ = model(seq[:, :prompt_len], cache=cache)
        full, _ = model(seq[:, :prompt_len])
        assert torch.allclose(full[:, -1], pre[:, -1], atol=1e-4)
        for i in range(5):
            t = prompt_len + i
            inc, _ = model(seq[:, t:t + 1], cache=cache)
            full, _ = model(seq[:, :t + 1])
            assert torch.allclose(full[:, -1], inc[:, -1], atol=1e-4), i


# ------------------------------------------------------------------ T4.3
@pytest.mark.parametrize("cell", list(CELLS))
def test_cache_contains_only_what_it_should(cell):
    cfg, _ = make(cell, "cpu")
    cache = make_cache(cfg, 1, 512, "cpu")
    is_local = resolve_pattern(cfg.attn_pattern, cfg.n_layer)
    for i, layer in enumerate(cache.layers):
        if cfg.attn_type == 'mla':
            # 'reconstructed' has no decoupled channel: the latent IS the cache. A
            # zero-width k_rope buffer would be legal and would quietly claim a rope
            # channel that does not exist, so its absence is asserted, not tolerated.
            expected_fields = ({'c_kv'} if cfg.rope_mode == 'reconstructed'
                               else {'c_kv', 'k_rope'})
            assert set(layer.buffers) == expected_fields
            assert layer.buffers['c_kv'].shape[-1] == cfg.kv_lora_rank
            if 'k_rope' in layer.buffers:
                assert layer.buffers['k_rope'].shape[-1] == cfg.qk_rope_head_dim
            assert 'k_nope' not in layer.buffers and 'v' not in layer.buffers
        else:
            assert set(layer.buffers) == {'k', 'v'}
            # GQA must cache n_kv_head heads, not n_head: caching the repeated tensor
            # would give MHA-sized memory with GQA's capacity, the worst of both
            assert layer.buffers['k'].shape[-2] == cfg.n_kv_head
        expected = cfg.window_size if is_local[i] else 512
        assert layer.capacity == expected, f"layer {i}: {layer.capacity} != {expected}"


# ------------------------------------------------------------------ T4.4
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA memory stats")
@pytest.mark.parametrize("cell", list(CELLS))
@pytest.mark.parametrize("seq_len", [1024, 4096])
def test_measured_cache_memory_matches_the_analytic_formula(cell, seq_len):
    """The headline table of the report is analytic; this checks it against the allocator.

    max_memory_allocated on a live model would be dominated by weights and activations
    (threats.md B7/C6), so the cache is allocated in isolation between a reset and a
    measurement.
    """
    cfg = GPTConfig(block_size=1024, vocab_size=50304, n_layer=8, n_head=8, n_embd=512,
                    dropout=0.0, bias=False, pos_encoding='rope', window_size=256,
                    **CELLS[cell])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    cache = make_cache(cfg, batch_size=1, max_seq_len=seq_len, device='cuda')
    measured = torch.cuda.memory_allocated() - before
    analytic = cache.spec.bytes(seq_len, batch_size=1, bytes_per_element=4)  # fp32 here
    rel = abs(measured - analytic) / analytic
    assert rel < 0.05, (f"{cell} T={seq_len}: measured {measured} B, "
                        f"analytic {analytic} B, {100 * rel:.2f}% apart")
    del cache
    torch.cuda.empty_cache()
