"""Gate T3.7 - the absorbed MLA decode path must be the naive one, reordered.

The absorbed form folds W^UK into the query and W^UV into the output, so attention runs
directly on the cached latent and the up-projection is paid per QUERY instead of per
CACHED TOKEN. It introduces no parameters: `kv_up` is the same nn.Linear, read as n_head
blocks. Consequently there is exactly one thing to prove -- that it computes the same
function -- and one way a reordering like this fails: not at the first step, but by
accumulating, so the parity is checked over many consecutive decode steps, past the point
where a sliding window has wrapped.

fp32 throughout: in bf16 even 1e-2 would be optimistic here, and the test would be
measuring the dtype instead of the algebra.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_cache import CacheSpec, HybridKVCache  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from model import MultiHeadLatentAttention  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
WINDOW = 16


def make(rope_mode, pattern, device="cpu"):
    cfg = GPTConfig(block_size=256, vocab_size=97, n_layer=4, n_head=4, n_embd=128,
                    dropout=0.0, bias=False, pos_encoding='rope', attn_impl='sdpa_mask',
                    window_size=WINDOW, attn_type='mla', attn_pattern=pattern,
                    rope_mode=rope_mode, absorb_mode='never')
    torch.manual_seed(0)
    return cfg, GPT(cfg).to(device).eval()


def make_cache(cfg, batch_size, max_seq_len, device):
    return HybridKVCache(CacheSpec.from_config(cfg, max_seq_len), batch_size, device,
                         dtype=torch.float32)


def set_path(model, mode):
    for block in model.transformer.h:
        block.attn.absorb_mode = mode


# --------------------------------------------------------------------- the gate

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rope_mode", ['additive', 'carved'])
@pytest.mark.parametrize("pattern", ['G', 'LLLG'])
def test_absorbed_matches_naive_across_many_steps(device, rope_mode, pattern):
    """3*W decode steps, so a local layer wraps its buffer twice under both paths."""
    cfg, model = make(rope_mode, pattern, device)
    prompt_len, n_steps = 8, 3 * WINDOW
    seq = torch.randint(0, cfg.vocab_size, (2, prompt_len + n_steps), device=device)
    naive_cache = make_cache(cfg, 2, prompt_len + n_steps, device)
    absorbed_cache = make_cache(cfg, 2, prompt_len + n_steps, device)

    with torch.no_grad():
        # both prefills run the naive path: what lands in the cache is the latent, which
        # is identical either way, so from here the two caches evolve together
        set_path(model, 'never')
        model(seq[:, :prompt_len], cache=naive_cache)
        model(seq[:, :prompt_len], cache=absorbed_cache)
        worst = 0.0
        for i in range(n_steps):
            t = prompt_len + i
            set_path(model, 'never')
            naive, _ = model(seq[:, t:t + 1], cache=naive_cache)
            set_path(model, 'always')
            absorbed, _ = model(seq[:, t:t + 1], cache=absorbed_cache)
            worst = max(worst, (naive - absorbed).abs().max().item())
            assert worst < 1e-4, f"diverged at step {i} (t={t}): {worst:.2e}"


@pytest.mark.parametrize("rope_mode", ['additive', 'carved'])
def test_absorbed_path_adds_no_parameters(rope_mode):
    """The fold reads kv_up; it does not own anything."""
    a = GPT(GPTConfig(**dict(n_layer=2, n_head=4, n_embd=128, vocab_size=97,
                             attn_type='mla', rope_mode=rope_mode, absorb_mode='never')))
    b = GPT(GPTConfig(**dict(n_layer=2, n_head=4, n_embd=128, vocab_size=97,
                             attn_type='mla', rope_mode=rope_mode, absorb_mode='always')))
    assert a.get_num_params() == b.get_num_params()
    assert {n for n, _ in a.named_parameters()} == {n for n, _ in b.named_parameters()}


# ------------------------------------------------------------------- the dispatch

def test_threshold_separates_the_two_regimes():
    cfg, model = make('additive', 'G')
    attn = model.transformer.h[0].attn
    t_star = attn.absorb_threshold()
    assert 1 < t_star < cfg.block_size, t_star
    cache = make_cache(cfg, 1, 128, "cpu")
    cache.pos = 32                                   # a cache exists and holds history
    assert attn._use_absorbed(1, cache) is False     # absorb_mode='never' still wins
    attn.absorb_mode = 'auto'
    assert attn._use_absorbed(1, cache) is True                    # decoding
    assert attn._use_absorbed(int(t_star) + 1, cache) is False     # chunked prefill
    assert attn._use_absorbed(1, None) is False                    # training / no cache


def test_training_never_absorbs():
    """The absorbed form is differentiable, so nothing would crash -- it would just
    train half a grid with one path and half with the other."""
    cfg, model = make('additive', 'G')
    attn = model.transformer.h[0].attn
    attn.absorb_mode = 'always'
    cache = make_cache(cfg, 1, 128, "cpu")
    model.train()
    assert attn._use_absorbed(1, cache) is False
    with pytest.raises(AssertionError, match="inference-only"):
        attn._forward_absorbed(torch.randn(1, 1, cfg.n_embd), None, None, cache)


def test_reconstructed_cannot_absorb():
    """Variant A rotates the rebuilt k, which puts W^UK inside the rotation."""
    with pytest.raises(AssertionError, match="cannot absorb"):
        GPTConfig(attn_type='mla', rope_mode='reconstructed', absorb_mode='always')
    cfg, model = make('additive', 'G')
    attn = model.transformer.h[0].attn
    attn.rope_mode, attn.absorb_mode = 'reconstructed', 'auto'
    assert attn._use_absorbed(1, make_cache(cfg, 1, 128, "cpu")) is False


def test_mac_ratio_crosses_one_exactly_at_the_threshold():
    """T* is derived for S >> T, so the crossover is exact only in that limit -- which
    is the regime it is used in. If this ever drifts, the dispatch is choosing the more
    expensive path on one side of the boundary."""
    _, model = make('additive', 'G')
    attn = model.transformer.h[0].attn
    t_star = attn.absorb_threshold()
    assert attn.absorb_mac_ratio(8192, T=1) > 10          # decoding: absorbed wins big
    assert abs(attn.absorb_mac_ratio(65536, T=t_star) - 1) < 1e-2   # the crossover
    assert attn.absorb_mac_ratio(8192, T=4 * t_star) < 1   # prefill: naive wins
