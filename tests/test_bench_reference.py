"""The "x MHA" column must divide by MHA, and an OOM must cost one row, not the run.

Two defects on the same code path of bench_inference.py:

  * the reference was `ref = tps if ref is None else ref`, i.e. whichever cell produced the
    first number. 1_mha_full is the largest cell and therefore the first to run out of
    memory; when it did, its row said OOM, `ref` stayed None, and the NEXT cell silently
    became the reference while the header still said "x MHA". No published table was
    affected (every recorded latency CSV has zero OOM rows and MHA first), but the trap was
    armed for any larger T or B.
  * the cache was allocated OUTSIDE the try that catches OutOfMemoryError, so an OOM there
    took the whole benchmark down instead of printing one OOM row -- while bench_kernels
    allocated it inside its own try.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_inference as BI  # noqa: E402


# ------------------------------------------------------------------ pure helpers (CPU)
def test_reference_is_mha_whenever_mha_is_selected():
    assert BI.ref_cell() == '1_mha_full'
    assert BI.ref_label() == 'x MHA'


def test_reference_falls_back_to_the_first_cell_and_says_so(monkeypatch):
    """--cells without MHA is legal; the header must then name what it really divides by."""
    monkeypatch.setattr(BI, 'CELLS', {k: BI.CELLS[k] for k in ('3_mla_full', '4_mla_swa')})
    assert BI.ref_cell() == '3_mla_full'
    assert BI.ref_label() == 'x mla_full'


def test_ratio_column_is_a_dash_when_the_reference_has_no_number():
    assert BI.ratio_col(100.0, None).strip() == '--'
    assert BI.ratio_col(100.0, 0).strip() == '--'      # max_batch: did not fit even at B=1
    assert BI.ratio_col(150.0, 100.0).strip() == '1.50x'


# ------------------------------------------------------------------ the loop (CUDA)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the benchmark allocates on CUDA")
def test_an_oom_on_the_reference_cell_costs_one_row_not_the_run(monkeypatch, capsys):
    real = BI.make_cache
    seen = {}

    def oom_for_mha(cfg, batch_size, max_seq_len, device, dtype):
        # the reference cell cannot allocate its cache; every other cell can
        if cfg.attn_type == 'mha':
            seen['raised'] = True
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(cfg, batch_size, max_seq_len, device, dtype)

    monkeypatch.setattr(BI, 'make_cache', oom_for_mha)
    monkeypatch.setattr(BI, 'CELLS', {k: BI.CELLS[k] for k in ('1_mha_full', '3_mla_full')})
    args = SimpleNamespace(impl="sdpa_mask", absorb="never", latency_lengths=[64],
                           batch_sizes=[1], decode_steps=2, warmup=1, sdpa_kernel="dispatcher")
    BI.bench_latency(args, lambda row: None)          # must NOT raise (the allocation fix)

    out = capsys.readouterr().out
    assert seen.get('raised'), "the reference cell was never asked to allocate"
    rows = [l for l in out.splitlines() if l.startswith(('1_mha_full', '3_mla_full'))]
    assert any(l.startswith('1_mha_full') and 'OOM' in l for l in rows), rows
    mla = next(l for l in rows if l.startswith('3_mla_full'))
    assert '--' in mla, f"MLA inherited the reference role: {mla!r}"
    assert 'x' not in mla.split()[-1], f"a ratio was printed without a reference: {mla!r}"
