"""SDPA kernel policy of the inference benchmarks: the fastest kernel for each call's shapes.

PyTorch's dispatcher walks a fixed priority list and takes the first kernel that accepts the
shapes. Cells with different shapes (MLA's score is 48 wide and its value 32, which flash
refuses) can then run different kernels, and a latency ratio between them measures the
kernels as much as the architectures. Policy 'fastest' times every kernel that accepts a
call and runs the fastest. These tests pin what that must never break: the result is the
dispatcher's, a refused kernel is never chosen, the choice is made once per shape, and
PyTorch's global flags come back exactly as they were.
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the SDPA kernels are CUDA kernels")
FLAGS = ('flash', 'mem_efficient', 'cudnn', 'math')


def flags():
    return {n: getattr(torch.backends.cuda, f"{n}_sdp_enabled")() for n in FLAGS}


@pytest.fixture(autouse=True)
def back_to_the_dispatcher():
    # the benchmarks leave their last choices recorded (they read them after the run), so a
    # test file that ran one before this would otherwise leak them in
    M.reset_sdpa_kernel_choices()
    yield
    M.set_sdpa_kernel_policy('dispatcher')
    M.reset_sdpa_kernel_choices()


def test_default_policy_is_the_dispatcher_and_records_nothing():
    q = torch.randn(1, 2, 4, 8)
    M.attend(q, q, q, impl="sdpa_mask")
    assert M.sdpa_kernel_policy_name() == 'dispatcher'
    assert M.sdpa_kernel_choices() == {}


def test_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="policy"):
        M.set_sdpa_kernel_policy('auto')


def test_fastest_leaves_cpu_tensors_to_the_dispatcher():
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, 4, 8, generator=g) for _ in range(3))
    ref = M.attend(q, k, v, impl="sdpa_mask")
    with M.sdpa_kernel_policy('fastest'), torch.no_grad():
        out = M.attend(q, k, v, impl="sdpa_mask")
    assert torch.equal(ref, out)
    assert M.sdpa_kernel_choices() == {}


def test_the_policy_context_restores_the_flags_it_found():
    before = flags()
    with M.sdpa_kernel_policy('fastest'):
        M._sdpa_enable_only(SDPBackend.MATH)
        assert flags() == {'flash': False, 'mem_efficient': False, 'cudnn': False, 'math': True}
    assert flags() == before


# name: (heads, kv heads, T, S, d_qk, d_v, local)
SHAPES = {
    "mha_decode": (16, 16, 1, 1000, 32, 32, False),
    "mla_decode_48_32": (16, 16, 1, 1000, 48, 32, False),
    "masked_prefill": (16, 16, 128, 128, 32, 32, True),
}


def tensors(heads, kv_heads, T, S, d_qk, d_v, B=4, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g).to("cuda", torch.bfloat16)
    return mk(B, heads, T, d_qk), mk(B, kv_heads, S, d_qk), mk(B, kv_heads, S, d_v)


@CUDA
@pytest.mark.parametrize("name", list(SHAPES))
def test_fastest_gives_the_dispatcher_result_with_a_kernel_that_accepts_the_shapes(name):
    heads, kv_heads, T, S, d_qk, d_v, local = SHAPES[name]
    q, k, v = tensors(heads, kv_heads, T, S, d_qk, d_v)
    kw = dict(is_local=local, window=32 if local else None, impl="sdpa_mask", scale=d_qk ** -0.5)
    before = flags()
    with torch.no_grad():
        ref = M.attend(q, k, v, **kw)
        with M.sdpa_kernel_policy('fastest'):
            out = M.attend(q, k, v, **kw)
            again = M.attend(q, k, v, **kw)
    assert flags() == before
    choices = list(M.sdpa_kernel_choices().values())
    assert len(choices) == 1, "one call signature, one choice"
    c = choices[0]
    timed = {n: t for n, t in c['ms'].items() if t is not None}
    assert c['fastest'] in timed and timed[c['fastest']] == min(timed.values())
    assert c['dispatcher'] in c['ms']
    assert torch.allclose(out.float(), ref.float(), atol=2e-2)
    assert torch.equal(out, again)
    if d_qk != d_v or local:
        # flash takes neither a value width different from the score width nor a mask
        assert c['ms']['FLASH_ATTENTION'] is None
        assert c['fastest'] != 'FLASH_ATTENTION'


@CUDA
def test_one_choice_per_shape_bucket_not_per_decode_step(monkeypatch):
    calls = []
    autotune = M._sdpa_autotune
    monkeypatch.setattr(M, "_sdpa_autotune", lambda *a, **k: calls.append(1) or autotune(*a, **k))
    with torch.no_grad(), M.sdpa_kernel_policy('fastest'):
        for S in (1000, 1001, 1010, 1024):             # one bucket: S <= 1024
            M.attend(*tensors(4, 4, 1, S, 16, 16), impl="sdpa_mask")
        assert len(calls) == 1
        M.attend(*tensors(4, 4, 1, 1025, 16, 16), impl="sdpa_mask")
        assert len(calls) == 2


@CUDA
def test_fastest_is_an_inference_policy():
    q = torch.randn(1, 2, 1, 8, device="cuda", dtype=torch.bfloat16)
    with M.sdpa_kernel_policy('fastest'), pytest.raises(AssertionError, match="no_grad"):
        M.attend(q, q, q, impl="sdpa_mask")


@CUDA
@pytest.mark.parametrize("bench", ["bench_latency", "bench_kernels"])
def test_benchmarks_record_the_kernel_of_every_call_signature(monkeypatch, bench):
    import bench_inference as BI
    monkeypatch.setattr(BI, "CELLS", {"3_mla_full": BI.CELLS["3_mla_full"]})
    args = SimpleNamespace(impl="sdpa_mask", absorb="never", latency_lengths=[64], batch_sizes=[1],
                           decode_steps=2, warmup=1, sdpa_kernel="fastest")
    rows, before = [], flags()
    getattr(BI, bench)(args, rows.append)
    assert flags() == before
    kernels = [json.loads(r['value']) for r in rows if r['metric'] == 'sdpa_kernel']
    assert kernels, "no kernel recorded"
    assert any(c['T'] == 1 for c in kernels), "the decode call is the one T5.2 times"
    if bench == "bench_latency":
        assert any(c['T'] > 1 for c in kernels), "the prefill runs under the policy too"
    for c in kernels:
        assert c['ms'][c['fastest']] is not None
        assert c['d_qk'] == 48 and c['d_v'] == 32 and c['ms']['FLASH_ATTENTION'] is None
