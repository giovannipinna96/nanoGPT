"""The architecture of every cell, in one place.

GPTConfig keyword arguments for the attention axes that tell the cells apart; everything
else is a shared base supplied by the caller. bench_inference.py,
analysis/kv_cache_analytic.py and analysis/param_table.py each used to keep their own copy,
and the copies had already drifted: param_table's cell 7 lacked force_last_global=False.
tests/test_cells.py pins the trained cells against config/grid_*.py.
"""

CELLS = {
    "1_mha_full": dict(attn_type='mha', attn_pattern='G'),
    "2_mha_swa": dict(attn_type='mha', attn_pattern='LLLG'),
    "3_mla_full": dict(attn_type='mla', attn_pattern='G'),
    "4_mla_swa": dict(attn_type='mla', attn_pattern='LLLG'),
    "5_gqa_full": dict(attn_type='gqa', attn_pattern='G', n_kv_head=4),
    # force_last_global=False: with it left on, attn_pattern='L' resolves to 7 local + 1
    # global on an 8-layer model, which is a variant of cell 4 and not the all-local
    # reading this cell exists to measure.
    "7_mla_all_local": dict(attn_type='mla', attn_pattern='L', force_last_global=False),
    # STEP 0b: cell 4 with the two symmetric-head-dim variants. 'carved' keeps the decoupled
    # channel (and with it the absorbed decode form) and is the architecture of grid cell 8;
    # 'reconstructed' caches the latent alone and can never absorb. Inference-only here.
    "4_mla_swa_carved": dict(attn_type='mla', attn_pattern='LLLG', rope_mode='carved'),
    "4_mla_swa_recon": dict(attn_type='mla', attn_pattern='LLLG', rope_mode='reconstructed'),
}


def select(names):
    """A fresh, ordered copy of the named cells: callers may reorder or mutate it."""
    return {name: dict(CELLS[name]) for name in names}
