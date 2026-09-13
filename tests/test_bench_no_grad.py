"""Every inference benchmark must run its forwards without autograd.

bench_max_batch_decode once ran with grad enabled. Each decode step then kept every layer's
activations alive for a backward that never happened, so the resident-batch numbers of T5.3b
and STEP 0b measured those activations instead of the KV cache -- plausible and wrong, with
no error anywhere. These tests make that impossible to reintroduce silently.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_inference as BI  # noqa: E402
from model import GPT  # noqa: E402


def test_build_model_refuses_to_run_with_autograd_enabled():
    """The guard fires before anything touches the GPU, so it is checkable on CPU."""
    cfg = BI.make_cfg('4_mla_swa', 1024, 'sdpa_mask')
    with torch.enable_grad(), pytest.raises(AssertionError, match="autograd"):
        BI.build_model(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the benchmarks allocate on CUDA")
@pytest.mark.parametrize("bench", ["bench_latency", "bench_kernels", "bench_max_batch",
                                   "bench_max_batch_decode"])
@pytest.mark.parametrize("absorb", ["never", "always"])
def test_every_benchmark_forward_runs_without_autograd(monkeypatch, bench, absorb):
    seen = []
    forward = GPT.forward

    def spy(self, *a, **k):
        seen.append((torch.is_grad_enabled(), any(p.requires_grad for p in self.parameters())))
        return forward(self, *a, **k)

    monkeypatch.setattr(GPT, "forward", spy)
    monkeypatch.setattr(BI, "CELLS", {"4_mla_swa": BI.CELLS["4_mla_swa"]})
    args = SimpleNamespace(impl="sdpa_mask", absorb=absorb, latency_lengths=[64], batch_sizes=[1],
                           decode_steps=2, warmup=1, batch_lengths=[64], max_batch_probe=2,
                           max_batch_probe_decode=2, sdpa_kernel="fastest")
    getattr(BI, bench)(args, lambda row: None)
    assert seen, "the benchmark never ran a forward"
    assert not any(grad for grad, _ in seen), "a forward ran with autograd enabled"
    assert not any(req for _, req in seen), "a benchmark model has parameters requiring grad"
