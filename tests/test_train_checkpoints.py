"""train.py and its checkpoints: new architecture knobs must reach the checkpoint.

configurator.py writes only onto train.py globals, and the checkpoint stores model_args;
a knob missing from either is silently dropped or refused, so both are checked end to end
on a tiny CPU run.
"""
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, 'data', 'finewebedu', 'val.bin')),
    reason="needs data/finewebedu/val.bin")

# warmup_iters < lr_decay_iters: equal values make train.py's get_lr divide by zero
TINY = dict(dataset='finewebedu', device='cpu', compile=False, n_layer=1, n_head=2, n_embd=64,
            block_size=64, batch_size=2, gradient_accumulation_steps=1, val_batches=2,
            eval_iters=2, eval_interval=1, max_iters=1, lr_decay_iters=2, warmup_iters=0,
            always_save_checkpoint=True, attn_type='mla', kv_lora_rank=64, pos_encoding='rope')


def train(out_dir, **over):
    args = dict(TINY, out_dir=str(out_dir))
    args.update(over)
    cmd = [sys.executable, 'train.py'] + [f'--{k}={v}' for k, v in args.items()]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return out


def test_mla_up_init_reaches_the_checkpoint(tmp_path):
    train(tmp_path, mla_up_init='matched')
    ckpt = torch.load(tmp_path / 'ckpt.pt', map_location='cpu', weights_only=False)
    assert ckpt['model_args']['mla_up_init'] == 'matched'


def test_resume_from_a_checkpoint_that_predates_the_newer_fields(tmp_path):
    """The grid checkpoints lack force_last_global, symmetric_head_dims, rope_mode and
    mla_up_init. Resuming them must fall back to the defaults, which reproduce how they
    were trained, instead of dying on a KeyError."""
    train(tmp_path)
    path = tmp_path / 'ckpt.pt'
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    for k in ('force_last_global', 'symmetric_head_dims', 'rope_mode', 'mla_up_init'):
        ckpt['model_args'].pop(k)
    torch.save(ckpt, path)
    train(tmp_path, init_from='resume', max_iters=2)
    resumed = torch.load(path, map_location='cpu', weights_only=False)
    assert resumed['iter_num'] == 2
    assert resumed['model_args']['force_last_global'] is True
    assert resumed['model_args']['symmetric_head_dims'] is False
    assert resumed['model_args']['rope_mode'] == 'additive'
    assert resumed['model_args']['mla_up_init'] == 'bottleneck'
