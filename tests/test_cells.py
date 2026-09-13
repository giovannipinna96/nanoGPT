"""Audit B-5 - the cell registry (cells.py) is the architecture the grid actually trained."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cells import CELLS  # noqa: E402
from model import GPTConfig, resolve_pattern  # noqa: E402

# grid config file -> registry entry with the same architecture
TRAINED = {"1_mha_full": "1_mha_full", "2_mha_swa": "2_mha_swa", "3_mla_full": "3_mla_full",
           "4_mla_swa": "4_mla_swa", "5_gqa_full": "5_gqa_full",
           "7_mla_all_local": "7_mla_all_local", "8_mla_swa_carved": "4_mla_swa_carved"}
# train.py globals the grid configs rely on when they leave a field out
TRAIN_DEFAULTS = dict(force_last_global=True, rope_mode='additive', n_kv_head=None,
                      qk_rope_head_dim=None)
ARCH = ["attn_type", "n_kv_head", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        "kv_lora_rank", "rope_mode", "window_size"]


def grid_config(cell, monkeypatch):
    monkeypatch.chdir(ROOT)
    g = dict(TRAIN_DEFAULTS)
    exec(open(f"config/grid_{cell}.py").read(), g)
    keys = list(TRAIN_DEFAULTS) + ["attn_type", "attn_pattern", "n_layer", "n_head", "n_embd",
                                   "window_size", "kv_lora_rank", "block_size", "bias",
                                   "pos_encoding"]
    return GPTConfig(**{k: g[k] for k in keys}), g


@pytest.mark.parametrize("cell", list(TRAINED))
def test_registry_matches_the_trained_grid_config(cell, monkeypatch):
    trained, g = grid_config(cell, monkeypatch)
    base = {k: g[k] for k in ["n_layer", "n_head", "n_embd", "window_size", "kv_lora_rank",
                              "block_size", "bias", "pos_encoding"]}
    reg = GPTConfig(**base, **CELLS[TRAINED[cell]])
    for k in ARCH:
        assert getattr(reg, k) == getattr(trained, k), (cell, k)
    assert (resolve_pattern(reg.attn_pattern, reg.n_layer, reg.force_last_global)
            == resolve_pattern(trained.attn_pattern, trained.n_layer, trained.force_last_global))
