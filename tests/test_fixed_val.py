"""Gate T0.3 (prerequisite): the val loss the grid compares must carry no sampling noise.

nanoGPT's estimate_loss calls get_batch('val'), which draws fresh random offsets at every
evaluation. The effect this project measures is 0.01-0.05 nats and that sampling noise is
the same order, so without a fixed val set the grid would be comparing dataloaders
(considerazioni_finali.md 2.2). These tests pin the property end to end, by running the
real train.py: two independent processes must print the SAME val loss, and the old random
path must be shown to differ -- otherwise the first assertion would pass vacuously.
"""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_RE = re.compile(r"val loss ([0-9.]+)")
FP_RE = re.compile(r"val fingerprint: (-?\d+)")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, 'data', 'finewebedu', 'val.bin')),
    reason="needs data/finewebedu/val.bin")


def run(tmp_path, **over):
    args = dict(dataset='finewebedu', device='cpu', compile=False, n_layer=1, n_head=2,
                n_embd=64, block_size=64, batch_size=2, gradient_accumulation_steps=1,
                val_batches=3, eval_iters=3, eval_interval=1, max_iters=1,
                lr_decay_iters=1, warmup_iters=1, eval_only=True, out_dir=str(tmp_path))
    args.update(over)
    cmd = [sys.executable, 'train.py'] + [f'--{k}={v}' for k, v in args.items()]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    m, f = VAL_RE.search(out.stdout), FP_RE.search(out.stdout)
    assert m and f, out.stdout[-2000:]
    return float(m.group(1)), int(f.group(1)), out.stdout


def test_fixed_val_is_identical_across_architectures(tmp_path):
    """The point of the fix: MHA and MLA must validate on exactly the same tokens.

    They do not have the same parameter tensors, so building them consumes the global
    RNG differently; with nanoGPT's random val sampling that shifts which offsets each
    cell draws. The fixed set uses a generator of its own and is immune.
    """
    _, fa, log = run(tmp_path / 'a', attn_type='mha')
    _, fb, _ = run(tmp_path / 'b', attn_type='mla', kv_lora_rank=64)
    assert fa == fb, f"fixed val set differs per architecture: {fa} vs {fb}"
    assert 'fixed val set:' in log, "the val set size must be reported (2.2)"


def test_random_val_differs_across_architectures(tmp_path):
    """Control: without the fix the confound is real, not hypothetical.

    Without this test the one above could pass for the trivial reason that the two runs
    happened to draw the same offsets anyway.
    """
    _, fa, _ = run(tmp_path / 'c', val_batches=0, attn_type='mha')
    _, fb, _ = run(tmp_path / 'd', val_batches=0, attn_type='mla', kv_lora_rank=64)
    assert fa != fb, ("random sampling gave the same val tokens to both architectures; "
                      "the fixed-val test is vacuous")


def test_fixed_val_is_reproducible_across_processes(tmp_path):
    """Same val set, same weights -> bit-identical val loss in two separate processes."""
    a, fa, _ = run(tmp_path / 'e')
    b, fb, _ = run(tmp_path / 'f')
    assert (a, fa) == (b, fb), f"fixed val set gave {a}/{fa} then {b}/{fb}"


def test_val_seed_selects_a_different_set(tmp_path):
    """A different val_seed must select a different measuring stick."""
    _, fa, _ = run(tmp_path / 'g')
    _, fb, _ = run(tmp_path / 'h', val_seed=99)
    assert fa != fb, "val_seed has no effect on the val set"
