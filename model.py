"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import contextlib
import math
import inspect
import time
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend

try:  # FlexAttention needs torch >= 2.5 (threats.md K5)
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _HAS_FLEX = True
except ImportError:  # pragma: no cover
    _HAS_FLEX = False

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RMSNorm(nn.Module):
    """RMSNorm over the last dimension, with the norm computed in fp32.

    DeepSeek-V2 3.1.2 puts an RMSNorm on the compressed latents; without it the low-rank
    bottleneck amplifies the variance and MLA training diverges (threats.md M5). The
    reduction is done in fp32 because an RMS taken in bf16 over a low-rank bottleneck is
    numerically fragile (threats.md X3).
    """

    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        dtype = x.dtype
        # upcast the reduction to fp32 for the low-precision dtypes only; downcasting a
        # float64 tensor here would silently destroy the precision that gradcheck needs
        xf = x.float() if dtype in (torch.float16, torch.bfloat16) else x
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self.weight

def precompute_rope(dim, max_seq, theta=10000.0, device=None):
    """cos/sin tables for RoPE on a `dim`-wide subspace.

    `dim` must be the dimension RoPE is ACTUALLY applied to, never a larger dimension
    that is then sliced. mla-experiments computes the frequencies on head_dim and keeps
    the first qk_rope_head_dim/2 of them, which keeps only the HIGH-frequency half and
    drops every long-wavelength channel. The slowest surviving channel, in tokens:

        shape                        correct   sliced
        head_dim 64, d_R 32 (theirs)   35333      471
        head_dim 32, d_R 16 (ours)     19869      353

    The first row is mla-experiments' own shape and is where the often-quoted 471 comes
    from; the second is configuration A, so do not carry those two figures over to this
    repository. Either way the period collapses below the context length, positions alias
    well inside 1024 tokens, and the needle probe would blame SWA for a RoPE defect
    (remediation.md #1, audit_implementazioni.md 1.3). Gate T1.3 checks the spectrum.

    Returns (max_seq, dim//2) tensors; the caller duplicates them to `dim` because we
    use the half-split (GPT-NeoX / HuggingFace) convention, see apply_rope.
    """
    assert dim % 2 == 0, f"RoPE needs an even dim, got {dim}"
    # the angle pos*inv_freq reaches ~4000 rad at the far end of the table; computing it
    # in fp32 leaves a phase error of ~1e-4 there, which shows up as a loss of
    # translation invariance at large positions. Build in float64, store in float32.
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float64) / dim))
    f = torch.outer(torch.arange(max_seq, device=device, dtype=torch.float64), inv)
    return f.cos().float(), f.sin().float()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    """Rotate x by the positions carried in cos/sin.

    x:        (B, H, T, D)
    cos, sin: (T, D//2) already selected for the ABSOLUTE positions of those T tokens.
    Passing the positions in explicitly (rather than assuming arange(T)) is what makes
    incremental decoding correct: during decoding the position is the absolute one, not
    the index of a rolling-buffer slot (threats.md C1).

    Convention: half-split (GPT-NeoX / HuggingFace / DeepSeek), i.e. channel i pairs
    with channel i+D/2. The interleaved convention is an equally valid rotation, but the
    two MUST NOT be mixed between q and k or the model simply trains worse with no
    symptom (threats.md P1). We use half-split because it is the convention of the
    DeepSeek reference code in transformers (modeling_deepseek_v3), so the MLA module
    stays comparable to it -- but no such comparison is run in this repository. What is
    tested is the rotation itself: translation invariance, norm preservation and an
    untruncated spectrum (tests/test_rope.py, gate T1.3).
    """
    c = torch.cat((cos, cos), dim=-1).to(x.dtype)[None, None]
    s = torch.cat((sin, sin), dim=-1).to(x.dtype)[None, None]
    return x * c + rotate_half(x) * s

def resolve_pattern(pattern, n_layer, force_last_global=True):
    """Expand an attention pattern string into one flag per layer. True = LOCAL.

    The string is tiled cyclically over the layers, e.g. 'LLLG' with n_layer=8 gives
    L L L G L L L G (ratio 3:1, the Gemma-3 style interleaving of soluzione.md 1.2).

    The last layer is forced GLOBAL (remediation.md #8): it is the layer that feeds
    lm_head directly, and it is the convention of Gemma 3 and nanochat. Note this
    changes the number of global layers and therefore the analytic KV-cache formula,
    so it must be stated in the report.

    `force_last_global=False` exists for the all-local reading of "hybrid" (cell 7 of
    news.md 6.2), which is the only configuration whose KV cache is strictly constant
    in T. It is not the default: the default is the interleaved design.
    """
    assert len(pattern) > 0, "empty attn_pattern"
    assert set(pattern) <= {"L", "G"}, f"attn_pattern must use only L/G, got {pattern!r}"
    p = [c == "L" for c in (pattern * n_layer)[:n_layer]]
    if force_last_global:
        p[-1] = False
        assert not all(p), "the pattern leaves no global layer"
    return p

# Block masks are expensive to build and must NOT be rebuilt on every forward: doing so
# makes sliding-window attention SLOWER than full attention (threats.md K2). The cache
# key is the full shape signature, so a new (T, S, window) combination builds once.
_MASK_CACHE = {}
_MASK_BUILDS = [0]   # cache misses, i.e. how many masks were actually constructed

def mask_build_count():
    """Gate T2.4: how many masks were built, block-sparse and dense together. Must stay at
    one per distinct shape signature.

    Rebuilding create_block_mask on every forward is what makes sliding-window
    attention SLOWER than full attention -- a disorienting symptom that looks like the
    window being useless (threats.md K2). The dense path shares the counter because the
    same caching argument applies to it, and tests/test_decode_masks.py checks both
    backends with it.
    """
    return _MASK_BUILDS[0]

def reset_mask_cache():
    _MASK_CACHE.clear()
    _MASK_BUILDS[0] = 0

def _mask_mod(is_local, window, q_offset):
    """Window convention: `q - kv < W`, i.e. exactly W visible tokens INCLUDING itself.

    Mistral 2 phrases it as "between i-W and i", inclusive on both ends, which is W+1
    tokens; FlexAttention and flash-attn converge on W. Neither is wrong, but mixing two
    conventions between the reference path and the fast path makes the equivalence tests
    fail for what looks like a numerical bug (threats.md S1). We use W everywhere and
    verify it by counting visible entries per row (gate T2.2).

    `q_offset` is the absolute position of the first query, which is non-zero during
    incremental decoding; the diagonal stays visible (q >= kv, not >) so no row can be
    fully masked (threats.md S4).
    """
    if not is_local:
        def mod(b, h, q_idx, kv_idx):
            return q_idx + q_offset >= kv_idx
    else:
        def mod(b, h, q_idx, kv_idx):
            q_abs = q_idx + q_offset
            return (q_abs >= kv_idx) & (q_abs - kv_idx < window)
    return mod

def get_block_mask(T, S, is_local, window, device, q_offset=0):
    """Block-sparse mask for FlexAttention, cached on (T, S, is_local, W, offset, device)."""
    key = ("flex", T, S, is_local, window if is_local else -1, q_offset, str(device))
    bm = _MASK_CACHE.get(key)
    if bm is None:
        bm = create_block_mask(_mask_mod(is_local, window, q_offset), None, None, T, S,
                               device=device)
        _MASK_CACHE[key] = bm
        _MASK_BUILDS[0] += 1
    return bm

def get_dense_mask(T, S, is_local, window, device, q_offset=0):
    """Dense boolean mask (T, S), True = visible. Correctness oracle only.

    A dense mask is CORRECT but saves nothing: the full T x S score matrix is still
    allocated and every dot product is still computed (threats.md S2). It is the
    reference path used by the tests, never by the grid runs.
    """
    key = ("dense", T, S, is_local, window if is_local else -1, q_offset, str(device))
    m = _MASK_CACHE.get(key)
    if m is None:
        q = torch.arange(T, device=device).unsqueeze(1) + q_offset
        kv = torch.arange(S, device=device).unsqueeze(0)
        m = q >= kv
        if is_local:
            m = m & ((q - kv) < window)
        _MASK_CACHE[key] = m
        _MASK_BUILDS[0] += 1
    return m

def _pad_to_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return max(p, 16)

# ------------------------------------------------------------------- SDPA kernel policy
# PyTorch's dispatcher walks a FIXED priority list (flash, efficient, math, ...) and takes
# the first kernel that accepts the shapes; it never asks which one is fastest. The
# inference benchmarks compare cells whose shapes differ -- MLA's score is 48 wide and its
# value 32, which flash refuses -- so under the dispatcher two cells can run two different
# kernels, and the latency ratio between them measures the kernels as much as the
# architectures. Policy 'fastest' times every kernel that accepts a call's shapes, once per
# shape, and runs the fastest from then on (bench_inference.py --sdpa-kernel). The default,
# 'dispatcher', leaves PyTorch alone: training and the tests never reach the timing below.
_SDPA_KERNELS = {SDPBackend.FLASH_ATTENTION: 'flash', SDPBackend.EFFICIENT_ATTENTION: 'mem_efficient',
                 SDPBackend.CUDNN_ATTENTION: 'cudnn', SDPBackend.MATH: 'math'}
_SDPA_NAMES = {int(getattr(SDPBackend, n)): n for n in dir(SDPBackend) if n.isupper()}
_SDPA = {'policy': 'dispatcher', 'saved': None, 'active': None}
_SDPA_CHOICES = {}   # call signature -> the choice made for it, see _sdpa_autotune

def set_sdpa_kernel_policy(policy):
    """'dispatcher' (default): PyTorch's priority list. 'fastest': for every call, the
    fastest kernel that accepts its shapes, chosen by timing them (CUDA, inference only)."""
    if policy not in ('dispatcher', 'fastest'):
        raise ValueError(f"unknown SDPA kernel policy {policy!r}: 'dispatcher' or 'fastest'")
    if policy == 'fastest' and _SDPA['saved'] is None:
        _SDPA['saved'] = {n: getattr(torch.backends.cuda, f"{n}_sdp_enabled")()
                          for n in _SDPA_KERNELS.values()}
        _SDPA['active'] = None
    if policy == 'dispatcher' and _SDPA['saved'] is not None:
        _sdpa_enable_only(None)          # the flags exactly as they were found
        _SDPA['saved'] = None
    _SDPA['policy'] = policy

def sdpa_kernel_policy_name():
    return _SDPA['policy']

@contextlib.contextmanager
def sdpa_kernel_policy(policy):
    """`with sdpa_kernel_policy('fastest'):` -- back to 'dispatcher', flags restored, on exit."""
    set_sdpa_kernel_policy(policy)
    try:
        yield
    finally:
        set_sdpa_kernel_policy('dispatcher')

def sdpa_kernel_choices():
    """{signature: choice} made since the last reset; every value is JSON-serialisable."""
    return {key: {f: val for f, val in c.items() if not f.startswith('_')}
            for key, c in _SDPA_CHOICES.items()}

def reset_sdpa_kernel_choices():
    _SDPA_CHOICES.clear()

def _sdpa_enable_only(kernel):
    """Enable `kernel` alone in PyTorch's global SDPA flags; None restores the saved flags.

    The flags are touched only when the kernel changes, so consecutive calls with the same
    choice pay nothing -- the torch.nn.attention.sdpa_kernel context manager costs ~15 us
    per call, which on a decode step of 8 layers would be a measurable share of it."""
    code = None if kernel is None else int(kernel)
    if _SDPA['active'] == code:
        return
    if kernel is None:
        state = _SDPA['saved']
    else:
        state = {n: int(b) == code for b, n in _SDPA_KERNELS.items()}
    for n, on in state.items():
        getattr(torch.backends.cuda, f"enable_{n}_sdp")(on)
    _SDPA['active'] = code

def _sdpa_signature(q, k, v, mask, enable_gqa):
    """What decides which kernels accept a call and how fast they are. S enters as the next
    power of two, so a decode loop, whose S grows by one per step, is timed once per bucket
    and not once per step."""
    S = k.size(-2)
    sig = dict(B=q.size(0), heads=q.size(1), kv_heads=k.size(1), T=q.size(-2),
               S_max=1 << (S - 1).bit_length(), d_qk=q.size(-1), d_v=v.size(-1),
               dtype=str(q.dtype).split('.')[-1], mask=mask is not None, gqa=enable_gqa,
               layout=[[t.is_contiguous(), 0 in t.stride()] for t in (q, k, v)])
    key = tuple((f, tuple(map(tuple, val)) if f == 'layout' else val) for f, val in sig.items())
    return key, sig

def _sdpa_fastest(q, k, v, mask, scale, enable_gqa):
    assert not torch.is_grad_enabled(), (
        "SDPA kernel policy 'fastest' times forward calls only: it is an inference policy, "
        "run under torch.no_grad()")
    key, sig = _sdpa_signature(q, k, v, mask, enable_gqa)
    choice = _SDPA_CHOICES.get(key)
    if choice is None:
        choice = _SDPA_CHOICES[key] = _sdpa_autotune(q, k, v, mask, scale, enable_gqa, sig)
    _sdpa_enable_only(choice['_kernel'])
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False,
                                          scale=scale, enable_gqa=enable_gqa)

def _sdpa_autotune(q, k, v, mask, scale, enable_gqa, sig):
    """Time every kernel that accepts this call; return the choice.

    A kernel is a candidate only if it runs AND reproduces the dispatcher's output: a fast
    wrong kernel must never win. math is not tried when its float32 (B, heads, T, S) weights
    could not fit, since the attempt would OOM in the middle of a benchmark. Each candidate
    runs once as a warm-up and is then timed over ~50 ms, at least one call."""
    def call():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False,
                                              scale=scale, enable_gqa=enable_gqa)

    _sdpa_enable_only(None)
    dispatcher = _SDPA_NAMES.get(int(torch._fused_sdp_choice(
        q, k, v, mask, 0.0, False, scale=scale, enable_gqa=enable_gqa)), 'UNKNOWN')
    ref = call()
    finite = bool(torch.isfinite(ref).all())
    tol = 1e-2 * max(ref.abs().max().item() if finite else 0.0, 1.0)
    math_bytes = 3 * 4 * q.size(0) * q.size(1) * q.size(-2) * k.size(-2)
    free_bytes = torch.cuda.mem_get_info(q.device)[0]
    ms, status, best = {}, {}, None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')        # every refusing kernel warns about why
        for kernel in _SDPA_KERNELS:
            name = _SDPA_NAMES[int(kernel)]
            ms[name] = None
            if int(kernel) == int(SDPBackend.MATH) and math_bytes > free_bytes // 2:
                status[name] = 'not tried: would not fit'
                continue
            _sdpa_enable_only(kernel)
            try:
                out = call()
                torch.cuda.synchronize()
            except RuntimeError:               # refuses these shapes, or out of memory
                status[name] = 'refused'
                torch.cuda.empty_cache()
                continue
            err = (out - ref).abs().max().item()
            del out
            if finite and not err <= tol:
                status[name] = f'wrong: max abs error {err:.3g}'
                continue
            t0 = time.perf_counter()
            call()
            torch.cuda.synchronize()
            reps = max(1, min(50, int(0.05 / max(time.perf_counter() - t0, 1e-6))))
            t0 = time.perf_counter()
            for _ in range(reps):
                call()
            torch.cuda.synchronize()
            ms[name] = (time.perf_counter() - t0) / reps * 1e3
            status[name] = 'ok'
            if best is None or ms[name] < ms[_SDPA_NAMES[int(best)]]:
                best = kernel
    del ref
    return dict(sig, fastest=None if best is None else _SDPA_NAMES[int(best)],
                dispatcher=dispatcher, ms=ms, status=status, _kernel=best)

def attend(q, k, v, *, is_local=False, window=None, dropout_p=0.0, impl="sdpa_mask",
           scale=None, q_offset=None):
    """Single attention entry point shared by every attention variant.

    q: (B, nh, T, D_qk)   k: (B, nkv, S, D_qk)   v: (B, nkv, S, D_v)  ->  (B, nh, T, D_v)

    nkv divides nh. GQA passes its key/value heads as they are cached, and every backend
    shares them across the query heads without copying the context once per query head.

    The local/global choice enters ONLY through `is_local`/`window`, which is the whole
    point of the design: MLA changes how k and v are parameterised, SWA changes which
    kv positions are visible, and the two never meet in the code. That is the
    constructive proof of orthogonality the report claims (soluzione.md 4.3).

    Backends:
      'sdpa_mask' - dense boolean mask + SDPA. Always available, correct, NOT sparse. The
                    backend of the inference benchmarks (bench_inference.py --impl), of the
                    Fase H probe (probe_longctx.py --impl) and, always, of the absorbed
                    decode path (audit M-3).
      'flex'      - FlexAttention block-sparse. The backend of every grid TRAINING run, so
                    the kernel is a constant across cells (threats.md K3).
      'flash'     - flash-attn 2 with window_size; an optional extra (`uv sync --extra
                    flash`), see pyproject.toml.

    `is_causal=True` is never used: SDPA silently IGNORES attn_mask when is_causal is
    set, which would make the sliding window disappear while the loss still looks great
    (threats.md K1).
    """
    S = k.size(-2)
    T = q.size(-2)
    n_rep = q.size(1) // k.size(1)
    assert k.size(1) == v.size(1) and q.size(1) == n_rep * k.size(1), (
        f"the query heads ({q.size(1)}) must be a multiple of the key/value heads "
        f"({k.size(1)}/{v.size(1)})")
    if q_offset is None:
        q_offset = S - T  # queries are the last T positions of the key sequence
    if is_local:
        assert window is not None and window > 0, "is_local=True requires a window size"
    # One query at the last position with every key visible: a decode step on a global
    # layer, or on a local layer whose ring buffer holds at most W keys. No mask is needed,
    # and building one here per step is what grew the mask caches without bound while
    # decoding, since their key changes with S and q_offset at every step (audit B-1).
    all_visible = T == 1 and q_offset >= S - 1 and (not is_local or q_offset < window)

    if impl == "flex":
        assert _HAS_FLEX, "FlexAttention requires torch >= 2.5"
        assert dropout_p == 0.0, "FlexAttention has no dropout; the grid runs use dropout=0"
        block_mask = (None if all_visible
                      else get_block_mask(T, S, is_local, window, q.device, q_offset))
        # FlexAttention (torch 2.6) rejects head dims that are not powers of two, and
        # MLA's score dimension is qk_nope + qk_rope = 32 + 16 = 48 on configuration A
        # (head_dim 32, remediation.md #5). Zero-padding q and k to 64 adds exactly zero
        # to every dot product, so the result is mathematically identical -- verified
        # against the dense oracle in tests/test_mla.py. The scale is passed explicitly,
        # so it stays 1/sqrt(48) and is NOT recomputed from the padded width (threats.md
        # M1). Cost: the attention matmuls run at 64 instead of 48, ~33% more attention
        # FLOPs on the MLA cells. Only flex pays it, i.e. training throughput (T5.4); the
        # decode latency of T5.2 runs on sdpa_mask and has no padding (audit M-3). The
        # KV-cache numbers are analytic and unaffected.
        d_qk, d_v = q.size(-1), v.size(-1)
        pad_qk, pad_v = _pad_to_pow2(d_qk), _pad_to_pow2(d_v)
        # Padding reintroduces threat M1 through the back door: with q/k padded from 48
        # to 64, the DEFAULT scale of flex_attention would become 1/sqrt(64) instead of
        # the correct 1/sqrt(48). Zero channels do not change the scores, but the scale
        # does -- and the model would train perfectly well while MLA came out
        # systematically worse. An explicit scale is therefore mandatory whenever we pad.
        assert pad_qk == d_qk or scale is not None, (
            f"attend() is padding the head dim {d_qk} -> {pad_qk}; pass the scale "
            f"explicitly (1/sqrt({d_qk})) or the padded width would silently set it "
            "(threats.md M1)")
        if pad_qk != d_qk:
            q = F.pad(q, (0, pad_qk - d_qk))
            k = F.pad(k, (0, pad_qk - d_qk))
        if pad_v != d_v:
            v = F.pad(v, (0, pad_v - d_v))
        # bool(): under torch.compile `n_rep > 1` is a symbolic boolean, which the flex
        # lowering cannot branch on ("cannot determine truth value of Relational"). The
        # head counts never change, so specialising on them costs nothing.
        y = flex_attention(q, k, v, block_mask=block_mask, scale=scale,
                           enable_gqa=bool(n_rep > 1))
        # Always slice back to d_v. When the query dim is padded and v is not, eager
        # flex_attention returns a d_v-wide output but the torch.compile path returns a
        # PADDED-width one; slicing is a no-op in the first case and the fix in the
        # second. Found the hard way: the eager tests passed and the compiled training
        # run died on a reshape.
        return y[..., :d_v]

    if impl == "flash":
        # flash-attn 2 carries the window INSIDE the kernel (`window_size=(left, right)`):
        # no BlockMask to build, no mask cache to police, and no power-of-two padding. It
        # is the reason this branch exists. Install with `uv sync --extra flash`; the exact
        # wheel and the ABI trap behind it are documented in pyproject.toml.
        from flash_attn import flash_attn_func  # optional dependency, CUDA-only kernel

        # (1) ONE head dim for q, k and v. Plain MHA/GQA satisfy this for free; MLA only
        #     with GPTConfig(symmetric_head_dims=True). Without the check the caller would
        #     get a raw kernel error at best, and at worst a silently reinterpreted tensor.
        d_qk, d_v = q.size(-1), v.size(-1)
        assert d_qk == d_v, (
            f"flash-attn needs q/k/v with the same head dim, got d_qk={d_qk}, d_v={d_v}. "
            "For MLA use GPTConfig(symmetric_head_dims=True), or stay on impl='flex'.")
        assert d_qk % 8 == 0 and d_qk <= 256, (
            f"flash-attn head dim must be a multiple of 8 and at most 256, got {d_qk}")
        # (2) the kernel is fp16/bf16 only and fails deep inside the extension otherwise
        assert q.dtype in (torch.float16, torch.bfloat16), (
            f"flash-attn is fp16/bf16 only, got {q.dtype}: run under autocast or cast first")
        # (3) causal alignment. When T != S flash-attn anchors the causal mask to the
        #     BOTTOM-RIGHT, i.e. it assumes the queries are the LAST T positions of the
        #     keys. That is the same convention as the default q_offset here (S - T), and
        #     the rolling cache does return exactly that -- but a caller passing anything
        #     else would silently get a different mask, which is threat K1 wearing another
        #     hat. Asserted rather than trusted.
        assert q_offset == S - T, (
            f"flash-attn assumes bottom-right causal alignment (q_offset == S - T == "
            f"{S - T}), got q_offset={q_offset}; use impl='sdpa_mask' or 'flex' instead")
        # (4) window convention: this file counts W visible tokens INCLUDING self, the
        #     kernel counts `left` tokens BEFORE self (threats.md S1), hence W - 1.
        w = (window - 1, 0) if is_local else (-1, -1)
        y = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                            dropout_p=dropout_p, softmax_scale=scale, causal=True,
                            window_size=w)
        return y.transpose(1, 2)

    if impl == "sdpa_mask":
        mask = None if all_visible else get_dense_mask(T, S, is_local, window, q.device, q_offset)
        if n_rep > 1 and mask is not None:
            # SDPA's own GQA runs only on the flash and math kernels (torch 2.6), and flash
            # takes no mask: enable_gqa here would land on math and its (B, nh, T, S)
            # weights. Repeating keeps the memory-efficient kernel, and it happens once per
            # prefill -- a decode step needs no mask and takes the native path below.
            k, v = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)
        gqa = bool(n_rep > 1) and mask is None
        if _SDPA['policy'] == 'fastest' and q.is_cuda:
            assert dropout_p == 0.0, "the 'fastest' SDPA kernel policy is inference-only"
            return _sdpa_fastest(q, k, v, mask, scale, enable_gqa=gqa)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p,
                                              is_causal=False, scale=scale, enable_gqa=gqa)

    raise ValueError(f"unknown attn_impl {impl!r}")

class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj._is_residual_proj = True   # scaled init marker (threats.md M8)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # per-layer local/global role, decided once by the pattern
        self.layer_idx = layer_idx
        self.is_local = resolve_pattern(config.attn_pattern, config.n_layer, config.force_last_global)[layer_idx]
        self.window = config.window_size
        self.attn_impl = config.attn_impl
        self.use_rope = config.pos_encoding == 'rope'

    def forward(self, x, cos=None, sin=None, cache=None, rope=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.use_rope:
            # plain MHA rotates the whole head; only MLA splits the head into a content
            # part and a smaller RoPE part (eq. 16-17 of DeepSeek-V2). cos/sin are
            # already selected for the ABSOLUTE positions of these T tokens, so the
            # rotation is correct during decoding too (threats.md C1).
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        q_offset = None
        if cache is not None:
            # k and v are cached POST-RoPE, so the position is frozen inside the value
            # and the rolling buffer never has to reconstruct it (remediation.md #10).
            k, v, q_offset = _cache_step(cache, self.layer_idx, T,
                                         k=k.transpose(1, 2), v=v.transpose(1, 2))
            k, v = k.transpose(1, 2), v.transpose(1, 2)

        # causal self-attention, dispatched through the shared backend. The original
        # nanoGPT line used is_causal=True, which silently ignores attn_mask and makes a
        # sliding window inexpressible (threats.md K1) -- it is replaced, not extended.
        y = attend(q, k, v, is_local=self.is_local, window=self.window,
                   dropout_p=self.dropout if self.training else 0.0,
                   impl=self.attn_impl, q_offset=q_offset)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention: the control cell the DeepSeek paper actually argues against.

    DeepSeek-V2 does not motivate MLA against MHA, it motivates it against GQA, which is
    the industry standard for shrinking the KV cache. Without this cell the obvious
    question -- "GQA cuts the cache by as much in three lines of code, why would I pay
    for MLA?" -- has no answer (considerazioni_finali.md 2.1).

    It is also the iso-cache comparison: with n_embd=512, n_head=16, head_dim=32, the
    grid's n_kv_head=4 and kv_lora_rank=256 (config/grid_5_gqa_full.py),

        GQA-4 : 2 * n_kv_head * head_dim = 2 * 4 * 32 = 256 elements/token/layer
        MLA   : d_c + d^R_h             = 256 + 16    = 272 elements/token/layer

    so GQA-4 has slightly LESS cache than MLA. At equal (indeed favourable) memory,
    which one wins on loss? Either answer is a result.
    """

    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_head % config.n_kv_head == 0, "n_head must be a multiple of n_kv_head"
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.kv_proj = nn.Linear(config.n_embd, 2 * self.n_kv_head * self.head_dim,
                                 bias=config.bias)
        self.c_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=config.bias)
        self.c_proj._is_residual_proj = True
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.is_local = resolve_pattern(config.attn_pattern, config.n_layer, config.force_last_global)[layer_idx]
        self.window = config.window_size
        self.attn_impl = config.attn_impl
        self.use_rope = config.pos_encoding == 'rope'

    def forward(self, x, cos=None, sin=None, cache=None, rope=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        q_offset = None
        if cache is not None:
            # only n_kv_head heads are cached, so the cache really is n_kv/n_head of the
            # MHA one
            k, v, q_offset = _cache_step(cache, self.layer_idx, T,
                                         k=k.transpose(1, 2), v=v.transpose(1, 2))
            k, v = k.transpose(1, 2), v.transpose(1, 2)

        # k and v go to attend() with their n_kv_head heads. Repeating them here, as this
        # module once did, copied the whole cached context at every decode step: GQA decoded
        # at 0.40x MHA (T=32768, B=64) on a cache four times smaller. The backends share
        # the heads natively instead (tests/test_gqa_native.py).
        y = attend(q, k, v, is_local=self.is_local, window=self.window,
                   dropout_p=self.dropout if self.training else 0.0,
                   impl=self.attn_impl, q_offset=q_offset)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.c_proj(y))

def _cache_step(cache, layer_idx, T, **tensors):
    """Write the new tokens, read the valid window back, return the mask offset.

    Tensors come in as (B, T, ...) with the token axis second, which is the layout the
    rolling buffer stores.

    Two regimes, deliberately kept separate:

    * PREFILL from an empty cache (T > 1, pos == 0): the chunk attends to ITSELF, with
      the band mask over the chunk, and the buffer simply keeps the last `capacity`
      entries. This is correct for any prompt length, including prompts far longer than
      the window.
    * DECODE (T == 1), and short continuations that still fit the window: write, then
      read the whole valid window, and let the mask see absolute positions through
      q_offset.

    Chunked prefill that both starts past position 0 AND overflows a local window is the
    remaining case (threat C7); it is asserted against rather than silently mishandled.
    """
    layer = cache.layers[layer_idx]
    if cache.pos == 0 and T > 1:
        layer.write(0, **tensors)
        return (*tensors.values(), 0)
    assert T == 1 or cache.pos + T <= layer.capacity, (
        f"chunked prefill across the window boundary is not implemented "
        f"(pos={cache.pos}, T={T}, capacity={layer.capacity})")
    layer.write(cache.pos, **tensors)
    out, start = layer.read(cache.pos + T)
    return (*[out[name] for name in tensors], cache.pos - start)

class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention, DeepSeek-V2 section 2.1 (eq. 9-19).

    The idea in one line: instead of caching k and v, cache a single small latent c_KV
    per token and rebuild k and v from it. The KV cache per token per layer drops from
    2*n_h*d_h to d_c + d^R_h.

    Shapes (soluzione.md 4.1). Only the two marked entries are ever cached:

        x        (B, T, d)
        c_KV     (B, T, d_c)                      <- CACHED
        k^R      (B, 1, T, d^R_h)                 <- CACHED, shared across heads
        q^C/q^R  (B, n_h, T, d_nope) / (..., d^R_h)
        k^C, v   (B, n_h, T, d_nope) / (..., d_v)    rebuilt from c_KV, never cached

    Five things that are easy to get wrong and produce no symptom (threats.md M1-M5):

    1. the softmax scale is 1/sqrt(qk_nope + qk_rope), NOT 1/sqrt(head_dim). The
       concatenated query has dimension d_nope + d^R_h, so using sqrt(d_h) inflates the
       logits by ~22%, lowers the attention entropy and makes MLA look systematically
       worse than MHA for a reason that has nothing to do with compression.
    2. k^R is SHARED across heads -- eq. (17) has no head index on it. It is generated
       once with shape (B, 1, T, d^R) and expanded AFTER RoPE. Generating it per head
       works, trains fine, and silently makes the reported KV cache more than 2x too
       large, which is the headline number of the report.
    3. k^R is derived from x, not from c_KV (eq. 15). Deriving it from the latent is
       symmetric and elegant and wrong: the latent is optimised to be reconstructible,
       not to carry position, and it would tie the RoPE width to the compression rank.
    4. RoPE is never applied to the "nope" halves. Rotating k^C would re-couple W^UK to
       a position-dependent matrix, which is precisely the problem that the decoupled
       RoPE of section 2.1.3 exists to avoid.
    5. RMSNorm on the compressed latent, computed in fp32. Without it the low-rank
       bottleneck amplifies the variance and training diverges.

    The local/global role is not visible anywhere in this class: it is a single flag
    handed to attend(). MLA reparameterises k and v, SWA restricts which positions are
    visible, and the two never interact. That is the orthogonality claim, made in code.
    """

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.d_nope = config.qk_nope_head_dim
        self.d_rope = config.qk_rope_head_dim
        self.d_v = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.rope_mode = config.rope_mode
        self.absorb_mode = config.absorb_mode
        d_qk = self.d_nope + self.d_rope

        # --- query path, eq. (12)-(14) ---
        # DeepSeek-V2-Lite does not compress the queries: query compression saves
        # activation memory in training but nothing in the KV cache, and at this scale
        # it costs capacity for no benefit. Default is None; T3.6 ablates it.
        if config.q_lora_rank is None:
            self.q_proj = nn.Linear(config.n_embd, self.n_head * d_qk, bias=False)
        else:
            self.q_down = nn.Linear(config.n_embd, config.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(config.q_lora_rank)
            self.q_up = nn.Linear(config.q_lora_rank, self.n_head * d_qk, bias=False)
            self.q_up._is_bottleneck_up_proj = True   # calibrated init (threats.md M7)

        # --- key/value path, eq. (9)-(11) + (15) ---
        # W^DKV and W^KR fused into one Linear, then split: both act on x, so this is
        # mathematically identical to two matrices and costs one matmul instead of two.
        self.kv_down = nn.Linear(config.n_embd, config.kv_lora_rank + self.d_rope,
                                 bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank)
        # W^UK and W^UV fused: ONE joint latent feeds both (this is the "joint" in
        # low-rank joint compression; two separate latents would double the cache and
        # would not be MLA at all, threats.md M3)
        self.kv_up = nn.Linear(config.kv_lora_rank,
                               self.n_head * (self.d_nope + self.d_v), bias=False)
        self.kv_up._is_bottleneck_up_proj = True   # calibrated init (threats.md M7)

        # --- output projection, eq. (19) ---
        # n_head * v_head_dim is NOT necessarily n_embd once v_head_dim is free (M9)
        self.o_proj = nn.Linear(self.n_head * self.d_v, config.n_embd, bias=False)
        self.o_proj._is_residual_proj = True   # scaled init marker, see STEP 7 / M8

        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        self.scale = d_qk ** -0.5              # eq. (18), threats.md M1
        self.is_local = resolve_pattern(config.attn_pattern, config.n_layer, config.force_last_global)[layer_idx]
        self.window = config.window_size
        self.attn_impl = config.attn_impl
        self.use_rope = config.pos_encoding == 'rope'

    def absorb_threshold(self):
        """T*: below this many queries the absorbed decode form is cheaper.

        Per decode call with S cached keys, T queries and n_h heads, counting MACs:

            naive     S*n_h*r*(d_nope+d_v)          rebuild k and v for ALL S keys
                    + T*S*n_h*(d_qk+d_v)            attention at the head width
            absorbed  T*n_h*r*(d_nope+d_v)          fold W^UK into q, W^UV out of y
                    + T*S*n_h*(2r+d_rope)           attention in the LATENT width

        The first line is the whole story: the naive form pays the up-projection per
        CACHED TOKEN, the absorbed one per QUERY. Setting the two equal for S >> T,

            T* = r*(d_nope+d_v) / (2r + d_rope - d_qk - d_v)

        and if the denominator is <= 0 the absorbed form is cheaper at every length.
        Note what this says about the regimes: decoding (T=1) sits far below T* and wants
        the absorbed form, prefill and training (T = full sequence) sit far above it and
        want the naive one. Same weights, same arithmetic, two different orders.
        """
        num = self.kv_lora_rank * (self.d_nope + self.d_v)
        den = 2 * self.kv_lora_rank + self.d_rope - (self.d_nope + self.d_rope) - self.d_v
        return float('inf') if den <= 0 else num / den

    def absorb_mac_ratio(self, context_len, T=1):
        """How many times fewer multiply-accumulates the absorbed path does, per layer.

        Same two MAC counts as absorb_threshold above, which derives them; this reports
        their RATIO instead of the T at which they cross. Kernel-independent arithmetic,
        the same currency as the KV-cache table of T5.1 (and unlike the latency of T5.2,
        which on this hardware is a lower bound).

        Note the ratio does NOT grow with S: both paths have a term linear in S, so at
        T=1 it converges to r*(d_nope+d_v) / (2r+d_rope), which is 31x at configuration A
        and is already 30x at S=1024. The absorbed form saves a large constant factor,
        not an unbounded one.
        """
        S, nh, r = context_len, self.n_head, self.kv_lora_rank
        d_qk, up = self.d_nope + self.d_rope, self.d_nope + self.d_v
        naive = S * r * nh * up + T * S * nh * (d_qk + self.d_v)
        absorbed = T * nh * r * up + T * S * nh * (2 * r + self.d_rope)
        return naive / absorbed

    def _use_absorbed(self, T, cache):
        if self.absorb_mode == 'never' or self.rope_mode == 'reconstructed':
            return False
        if cache is None or self.training:
            # without a cache S == T: the up-projection is amortised over every query and
            # the naive form is strictly cheaper. In training it would also double the
            # attention work and make the backward heavier, so it is refused outright
            # rather than left to the threshold.
            return False
        return self.absorb_mode == 'always' or T < self.absorb_threshold()

    def forward(self, x, cos=None, sin=None, cache=None, rope=None):
        if self.rope_mode == 'reconstructed':
            return self._forward_reconstructed(x, cos, sin, cache, rope)
        if self._use_absorbed(x.size(1), cache):
            return self._forward_absorbed(x, cos, sin, cache)
        return self._forward_decoupled(x, cos, sin, cache)

    def _forward_absorbed(self, x, cos, sin, cache):
        """The same function as _forward_decoupled, evaluated in a different order.

        score = q^C . (W^UK c) = (W^UK^T q^C) . c        and       y = W^UV (sum_s a_s c_s)

        so W^UK can be folded into the QUERY (T vectors) instead of applied to every
        cached key (S vectors), and W^UV applied to the OUTPUT instead of to every cached
        value. Attention then runs directly on the latent: k = [c, k^R], v = c, one KV
        head shared by all query heads. No new parameters, no precomputed matrices, no
        retraining -- kv_up is the same nn.Linear, read as n_head blocks.

        Three things that are silently wrong if done differently:

        M1  the scale stays 1/sqrt(d_qk). The absorbed score is the SAME NUMBER as the
            naive one, so it must be normalised the same way; r + d_rope is an
            implementation width, not the dimension of the attention. Letting SDPA pick
            its default here would rescale every logit and quietly change the model.
        M6  the fold is PER HEAD. kv_up.weight is (n_head*(d_nope+d_v), r) and must be
            read as n_head separate blocks; one big matmul would mix the heads and give a
            different, perfectly plausible-looking model.
        --  the value width (r) differs from the score width (r + d_rope), which flash
            refuses and flex would pad to the next power of two (272 -> 512 at the grid
            config, i.e. the fix would cost more than the bug). The absorbed path
            therefore runs on the dense-mask backend, which is free here: with T = 1 the
            mask is a single row.
        """
        assert not self.training, "the absorbed path is inference-only"
        B, T, _ = x.size()
        nh, r = self.n_head, self.kv_lora_rank

        # 1) queries: identical to the naive path up to the split
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_up(self.q_norm(self.q_down(x)))
        q = q.view(B, T, nh, self.d_nope + self.d_rope).transpose(1, 2)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)
        if self.use_rope:
            q_rope = apply_rope(q_rope, cos, sin)

        # 2) THE FOLD. kv_up read as n_head blocks of (d_nope + d_v, r)
        w = self.kv_up.weight.view(nh, self.d_nope + self.d_v, r)
        w_uk, w_uv = w.split([self.d_nope, self.d_v], dim=1)
        q_lat = torch.einsum('bhtd,hdr->bhtr', q_nope, w_uk)

        # 3) the cache: the latent is never expanded
        kv = self.kv_down(x)
        kv_lat, k_rope = kv.split([r, self.d_rope], dim=-1)
        kv_lat = self.kv_norm(kv_lat)
        k_rope = k_rope.unsqueeze(1)
        if self.use_rope:
            k_rope = apply_rope(k_rope, cos, sin)
        kv_lat, k_rope_c, q_offset = _cache_step(
            cache, self.layer_idx, T, c_kv=kv_lat, k_rope=k_rope.squeeze(1))
        k_rope = k_rope_c.unsqueeze(1)

        # 4) attention in the latent space. k and v are the SAME tensor up to k^R.
        #    expand(), not repeat(): a view, so the one KV head costs nothing per head.
        q_abs = torch.cat([q_lat, q_rope], dim=-1)                    # (B, nh, T, r+d_rope)
        k_abs = torch.cat([kv_lat.unsqueeze(1), k_rope], dim=-1).expand(-1, nh, -1, -1)
        v_abs = kv_lat.unsqueeze(1).expand(-1, nh, -1, -1)            # (B, nh, S, r)
        y_lat = attend(q_abs, k_abs, v_abs, is_local=self.is_local, window=self.window,
                       dropout_p=0.0, impl='sdpa_mask', scale=self.scale,
                       q_offset=q_offset)

        # 5) unfold W^UV on the OUTPUT: T vectors, not S
        y = torch.einsum('bhtr,hvr->bhtv', y_lat, w_uv)
        y = y.transpose(1, 2).contiguous().view(B, T, nh * self.d_v)
        return self.resid_dropout(self.o_proj(y))

    def _forward_reconstructed(self, x, cos, sin, cache, rope):
        """Variant A: no decoupled channel. Rotate the k rebuilt from the latent.

        The cache holds c_KV and nothing else -- r elements per token, the smallest of
        the three modes -- and q/k/v are all head_dim wide, so no kernel has anything to
        complain about and the content width stays FULL (unlike 'carved').

        The one thing that must not go wrong: the rebuilt keys are rotated at their OWN
        absolute positions, which start BEFORE the queries whenever a cache is in play
        and, after a wrap, are not the buffer slot indices either (threat C1). The first
        key position is recovered from the offset the cache returns, and the cos/sin for
        that range are sliced by the model's own table -- never rebuilt here, so the
        length-extrapolation behaviour of T7.3 is identical for every mode.

        Why this forecloses the absorbed form: W^UK ends up sandwiched inside a
        position-dependent rotation, R_s (W^UK c_s), so it can no longer be folded into
        W^Q once and for all. That is precisely the coupling the decoupled channel of
        DeepSeek-V2 2.1.3 exists to avoid, and giving it up is the price of variant A.
        """
        B, T, _ = x.size()
        nh = self.n_head

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_up(self.q_norm(self.q_down(x)))
        q = q.view(B, T, nh, self.d_nope).transpose(1, 2)

        kv_lat = self.kv_norm(self.kv_down(x))       # (B, T, r) -- the ONLY cached tensor

        q_offset, k_start = None, 0
        if cache is not None:
            kv_lat, q_offset = _cache_step(cache, self.layer_idx, T, c_kv=kv_lat)
            k_start = cache.pos - q_offset           # absolute position of the first key

        S = kv_lat.size(1)
        kv_up = self.kv_up(kv_lat).view(B, S, nh, self.d_nope + self.d_v).transpose(1, 2)
        k, v = kv_up.split([self.d_nope, self.d_v], dim=-1)

        if self.use_rope:
            q = apply_rope(q, cos, sin)
            if cache is None:
                cos_k, sin_k = cos, sin              # keys are the queries
            else:
                assert rope is not None, (
                    "rope_mode='reconstructed' needs the model's rope slicer to rotate "
                    "the rebuilt keys at their own positions")
                cos_k, sin_k = rope(k_start, S)
            k = apply_rope(k, cos_k, sin_k)

        y = attend(q, k, v, is_local=self.is_local, window=self.window,
                   dropout_p=self.dropout if self.training else 0.0,
                   impl=self.attn_impl, scale=self.scale, q_offset=q_offset)
        y = y.transpose(1, 2).contiguous().view(B, T, nh * self.d_v)
        return self.resid_dropout(self.o_proj(y))

    def _forward_decoupled(self, x, cos=None, sin=None, cache=None):
        """The decoupled-RoPE path of DeepSeek-V2, eq. (14)-(19). The default, and the
        path every recorded grid run trained on.

        Naive order, in four steps: build q, build the two cacheable tensors (c_KV and the
        shared k^R) and rotate them at their absolute positions, rebuild k^C and v from the
        latent, concatenate and attend. _forward_absorbed is this same function evaluated
        in a different order and is used only for decoding; training always comes through
        here, whatever absorb_mode says.

        Serves modes 'additive' and 'carved', which differ ONLY in how the widths are
        derived in GPTConfig.__post_init__ -- not in a single line below. At head_dim=32:

            mode      d_nope  d_rope   d_qk = nope+rope   d_v   cache/token
            additive      32      16                 48    32   d_c + 16
            carved        24       8                 32    32   d_c +  8     (cell 8)
            carved        16      16                 32    32   d_c + 16     (cell 9)

        'additive' ADDS the RoPE channel on top of the head width, so d_qk (48) exceeds
        d_v (32): that asymmetry is what flex pads to 64 and what flash refuses. 'carved'
        takes the channel OUT of the head width instead, so d_qk == d_v == head_dim and
        nothing is padded -- at the cost of a narrower content subspace. Cell 9 carves
        d_rope=16 rather than the default 8 precisely so its cache matches cell 4's and
        the contrast isolates the carve alone (config/grid_9_mla_swa_carved16.py).

        The scale follows d_qk and therefore differs between the two (M1, class docstring):
        it is set once in __init__, never here.
        """
        B, T, _ = x.size()

        # 1) queries
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_up(self.q_norm(self.q_down(x)))
        q = q.view(B, T, self.n_head, self.d_nope + self.d_rope).transpose(1, 2)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)

        # 2) the two tensors that are cacheable: c_KV and k^R
        kv = self.kv_down(x)
        kv_lat, k_rope = kv.split([self.kv_lora_rank, self.d_rope], dim=-1)
        kv_lat = self.kv_norm(kv_lat)
        k_rope = k_rope.unsqueeze(1)                      # (B, 1, T, d_rope), shared

        if self.use_rope:
            q_rope = apply_rope(q_rope, cos, sin)          # eq. (14)
            k_rope = apply_rope(k_rope, cos, sin)          # eq. (15), ONCE for all heads

        # 2b) THE ONLY TWO TENSORS THAT ARE EVER CACHED. Note what is NOT here:
        # k_nope and v are rebuilt below from c_KV and never stored. Caching them would
        # give perfect quality with MHA-sized memory and quietly cancel the project
        # (threats.md C3).
        q_offset = None
        if cache is not None:
            kv_lat, k_rope_c, q_offset = _cache_step(
                cache, self.layer_idx, T,
                c_kv=kv_lat, k_rope=k_rope.squeeze(1))
            k_rope = k_rope_c.unsqueeze(1)

        # 3) rebuild k^C and v from the latent
        S = kv_lat.size(1)
        kv_up = self.kv_up(kv_lat).view(B, S, self.n_head,
                                        self.d_nope + self.d_v).transpose(1, 2)
        k_nope, v = kv_up.split([self.d_nope, self.d_v], dim=-1)

        # 4) concatenate and attend. is_local is the ONLY place SWA enters.
        q_full = torch.cat([q_nope, q_rope], dim=-1)                       # eq. (16)
        k_full = torch.cat([k_nope, k_rope.expand(-1, self.n_head, -1, -1)],
                           dim=-1)                                         # eq. (17)
        y = attend(q_full, k_full, v, is_local=self.is_local, window=self.window,
                   dropout_p=self.dropout if self.training else 0.0,
                   impl=self.attn_impl, scale=self.scale,
                   q_offset=q_offset)                                      # eq. (18)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_v)
        return self.resid_dropout(self.o_proj(y))                          # eq. (19)

def build_attention(config, layer_idx):
    """Pick the attention parameterisation. The mask pattern is orthogonal to this."""
    if config.attn_type == 'mha':
        return CausalSelfAttention(config, layer_idx)
    if config.attn_type == 'mla':
        return MultiHeadLatentAttention(config, layer_idx)
    if config.attn_type == 'gqa':
        return GroupedQueryAttention(config, layer_idx)
    raise ValueError(f"unknown attn_type {config.attn_type!r}")

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj._is_residual_proj = True   # scaled init marker (threats.md M8)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = build_attention(config, layer_idx)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cos=None, sin=None, cache=None, rope=None):
        x = x + self.attn(self.ln_1(x), cos, sin, cache, rope)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # --- attention parameterisation (how k and v are represented) ---
    attn_type: str = 'mha'          # 'mha' | 'gqa' | 'mla'
    n_kv_head: int = None           # GQA only; None -> n_head (i.e. plain MHA)

    # --- attention mask (which kv positions are visible) ---
    attn_pattern: str = 'G'         # tiled over layers, e.g. 'LLLG'; last layer forced global
    window_size: int = 256          # W visible tokens INCLUDING self (see _mask_mod)
    force_last_global: bool = True  # remediation.md #8. Set False ONLY for the all-local
                                    # reading of "hybrid" (grid cell 7, news.md 6): with it
                                    # left on, attn_pattern='L' silently resolves to 7
                                    # local + 1 global and the cell measures the wrong thing

    # --- MLA hyperparameters (DeepSeek-V2 2.1), filled from head_dim in __post_init__ ---
    kv_lora_rank: int = None        # d_c        default 4 * head_dim   (remediation.md #7)
    q_lora_rank: int = None         # d'_c       None = no query compression (DeepSeek-V2-Lite)
    qk_nope_head_dim: int = None    # d_h        default head_dim, FULL   (remediation.md #5)
    qk_rope_head_dim: int = None    # d^R_h      default head_dim // 2, ADDITIVE
    v_head_dim: int = None          # d_v        default head_dim
    symmetric_head_dims: bool = False   # MLA only: force d_v = d_qk, see __post_init__
    absorb_mode: str = 'auto'       # MLA decode path: 'auto' | 'never' | 'always'.
                                    # 'auto' picks the absorbed form below T*, see
                                    # MultiHeadLatentAttention.absorb_threshold()
    rope_mode: str = 'additive'     # MLA only, where the RoPE channel comes from:
                                    # 'additive'      d_qk = head_dim + d_rope (DeepSeek)
                                    # 'carved'        d_rope taken OUT of head_dim
                                    # 'reconstructed' no channel; rotate the rebuilt k
    mla_up_init: str = 'bottleneck' # MLA only, init of the up-projections kv_up/q_up:
                                    # 'bottleneck'  std = in_features**-0.5, what every
                                    #               recorded run used
                                    # 'matched'     std = 0.02*sqrt(n_embd/in_features),
                                    #               k/v init variance equal to MHA (M-2)

    # --- positions ---
    pos_encoding: str = 'learned'   # 'learned' (nanoGPT wpe) | 'rope'
    rope_theta: float = 10000.0

    # --- backend ---
    attn_impl: str = 'sdpa_mask'    # 'sdpa_mask' (oracle) | 'flex' (grid runs) | 'flash'

    def __post_init__(self):
        head_dim = self.n_embd // self.n_head
        if self.kv_lora_rank is None:
            self.kv_lora_rank = 4 * head_dim
        assert self.rope_mode in ('additive', 'carved', 'reconstructed'), self.rope_mode
        assert self.absorb_mode in ('auto', 'never', 'always'), self.absorb_mode
        assert self.mla_up_init in ('bottleneck', 'matched'), self.mla_up_init

        # Where the RoPE channel comes from. This one choice fixes d_qk, and d_qk is the
        # only thing the kernels care about.
        #
        # 'additive' -- DeepSeek-V2 2.1.3, the default, and what every recorded number
        #     was measured with. The decoupled channel is ADDED to a full-width content
        #     part: d_qk = head_dim + d_rope. Content capacity is preserved, so a loss
        #     gap is attributable to the compression alone (remediation.md #5), which is
        #     why the grid uses it. The price is that d_qk != d_v and d_qk is not a power
        #     of two: flex pads the score side (~33% of its attention FLOPs), flash
        #     refuses the call, and at head_dim=64 the padded backward does not compile
        #     on sm80 -- which is what forced configuration A.
        #
        # 'carved' -- take the channel OUT of the head instead: d_nope = head_dim -
        #     d_rope, hence d_qk = d_v = head_dim. Every kernel constraint above
        #     disappears at once (no padding on flex, native on flash, any power-of-two
        #     head_dim compiles) and the decoupled channel survives, so the absorbed
        #     decode form stays reachable. The price is d_rope/head_dim less content
        #     width per head, and a loss gap no longer attributable to compression alone.
        #
        # 'reconstructed' -- no decoupled channel at all. RoPE is applied to the k
        #     rebuilt from the latent, at its own absolute position, exactly as MHA does.
        #     d_qk = d_v = head_dim with FULL content width, and the cache drops to r
        #     elements per token, the smallest of the three. The price is irreversible:
        #     rotating a reconstructed k re-couples W^UK to the position, so the absorbed
        #     form becomes impossible. Choose it when memory matters more than decode.
        if self.rope_mode == 'reconstructed':
            assert self.qk_rope_head_dim in (None, 0), (
                "rope_mode='reconstructed' has no decoupled channel; leave "
                "qk_rope_head_dim unset")
            self.qk_rope_head_dim = 0
            if self.qk_nope_head_dim is None:
                self.qk_nope_head_dim = head_dim
        elif self.rope_mode == 'carved':
            if self.qk_rope_head_dim is None:
                self.qk_rope_head_dim = max(8, head_dim // 4)
            assert self.qk_rope_head_dim % 2 == 0, (
                "the RoPE channel is rotated in pairs and must be even, got "
                f"{self.qk_rope_head_dim}")
            assert 0 < self.qk_rope_head_dim < head_dim, (
                f"a carved channel must fit inside the head: 0 < "
                f"{self.qk_rope_head_dim} < {head_dim}")
            carved = head_dim - self.qk_rope_head_dim
            assert self.qk_nope_head_dim in (None, carved), (
                f"rope_mode='carved' derives qk_nope_head_dim = head_dim - "
                f"qk_rope_head_dim = {carved}; drop the explicit value")
            self.qk_nope_head_dim = carved
        else:
            if self.qk_nope_head_dim is None:
                self.qk_nope_head_dim = head_dim
            if self.qk_rope_head_dim is None:
                self.qk_rope_head_dim = head_dim // 2
        if self.rope_mode == 'reconstructed' and self.absorb_mode == 'always':
            raise AssertionError(
                "rope_mode='reconstructed' cannot absorb: rotating the rebuilt k puts "
                "W^UK inside a position-dependent rotation, so it cannot be folded into "
                "W^Q. Use rope_mode='carved' if you want the absorbed decode path.")
        if self.rope_mode in ('carved', 'reconstructed'):
            # both variants exist to make the three widths equal, so d_v is derived and
            # an explicit contradicting value is a typo, not an option
            assert self.v_head_dim in (None, head_dim), (
                f"rope_mode={self.rope_mode!r} pins v_head_dim to head_dim={head_dim} "
                f"(that is its whole purpose), but v_head_dim={self.v_head_dim} was given")
            assert not self.symmetric_head_dims, (
                "symmetric_head_dims widens d_v up to d_qk; rope_mode="
                f"{self.rope_mode!r} already has d_qk == d_v == head_dim by construction "
                "and widening on top would only add parameters. Today's measurement says "
                "widening costs ~15% throughput, carving costs none: keep the carve.")
            self.v_head_dim = head_dim
            assert self.qk_nope_head_dim + self.qk_rope_head_dim == head_dim
        # flash-attn wants ONE head dim for q, k and v; FlexAttention wants a power of two.
        # By default MLA satisfies neither: d_qk = qk_nope + qk_rope = 48 and d_v = 32 at
        # this width, so the flex path pads the SCORE side 48 -> 64 and throws away ~33%
        # of the attention FLOPs (v = 32 is already a power of two and is left alone, see
        # attend()). Choosing d_v = d_qk makes the three widths identical and a multiple
        # of 8, which is exactly what the flash kernel accepts: with impl='flash' NOTHING
        # is padded. Note the flag is only worth setting TOGETHER with impl='flash': on
        # flex a symmetric 48/48 would pad both sides and be strictly worse.
        #
        # It is a parameterisation change, not a free lunch: kv_up and o_proj grow with
        # v_head_dim, so a symmetric cell is no longer parameter-matched against the grid
        # runs. Hence OPT-IN and off by default -- every number already recorded stays
        # reproducible. The KV cache is untouched, because MLA caches c_KV and k^R and
        # never v: the headline memory table is bit-identical with and without this flag.
        if self.symmetric_head_dims:
            assert self.attn_type == 'mla', (
                "symmetric_head_dims is an MLA-only knob: mha/gqa already have "
                "d_qk == d_v == head_dim")
            d_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
            assert self.v_head_dim in (None, d_qk), (
                f"symmetric_head_dims=True forces v_head_dim = qk_nope + qk_rope = {d_qk}, "
                f"but v_head_dim={self.v_head_dim} was also given; drop one of the two")
            self.v_head_dim = d_qk
        if self.v_head_dim is None:
            self.v_head_dim = head_dim
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # NOTE: keep wpe in its original position in this dict. self.apply(_init_weights)
        # walks the modules in registration order, so moving it changes the order in
        # which the RNG is consumed and therefore every initial weight -- which breaks
        # the T1.1 identity for reasons that have nothing to do with positions.
        modules = dict(wte=nn.Embedding(config.vocab_size, config.n_embd))
        if config.pos_encoding == 'learned':
            # learned absolute positions cannot generate past block_size and would have
            # to be allocated for the full generation length, which erodes the very
            # memory saving SWA exists for (remediation.md #9)
            modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        modules.update(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )
        self.transformer = nn.ModuleDict(modules)
        if config.pos_encoding == 'rope':
            # RoPE lives on the dimension it is applied to: the full head for MHA/GQA,
            # the dedicated qk_rope subspace for MLA (never head_dim then sliced, #1).
            head_dim = config.n_embd // config.n_head
            # 'reconstructed' MLA rotates the FULL rebuilt head, so it needs the same
            # table as MHA; the other two modes rotate only the decoupled channel.
            rope_dim = (config.qk_rope_head_dim
                        if config.attn_type == 'mla' and config.qk_rope_head_dim > 0
                        else head_dim)
            # 4x headroom so that length-extrapolation evaluation (T7.3) at 2x block_size
            # needs no rebuild; the table is a non-persistent buffer, so the checkpoint
            # does not grow
            cos, sin = precompute_rope(rope_dim, 4 * config.block_size, config.rope_theta)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)

        # Special scaled init on the residual projections, per the GPT-2 paper. nanoGPT
        # selects them by NAME (pn.endswith('c_proj.weight')), which is the single most
        # dangerous trap of this codebase for our purpose (threats.md M8): the MLA
        # output projection is called o_proj, so it would silently NOT receive the
        # 1/sqrt(2L) scaling. The MHA branch would then have the variance of its
        # residual stream controlled with depth and the MLA branch would not -- MLA
        # converges worse, with no symptom, and the experiment measures the init
        # instead of the architecture. We mark the modules explicitly and assert the
        # count, so any new attention variant either gets the same treatment or fails
        # loudly at construction time.
        #
        # NOTE on the count: the original filter matches BOTH attn.c_proj and
        # mlp.c_proj, i.e. two modules per layer. plan.md/remediation.md #6 write the
        # assert as n_layer; that would exclude the MLP and change the baseline, so the
        # faithful count is 2 * n_layer.
        n_scaled = 0
        for m in self.modules():
            if getattr(m, '_is_residual_proj', False):
                torch.nn.init.normal_(m.weight, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))
                n_scaled += 1
        assert n_scaled == 2 * config.n_layer, (
            f"scaled init reached {n_scaled} modules instead of {2 * config.n_layer}: "
            "some residual projection is unmarked and would train with a different "
            "variance than the others (threats.md M8)")

        # Bottleneck-calibrated init for the MLA up-projections (threats.md M7). Their
        # fan_in is the latent rank d_c, not n_embd, so the fixed std=0.02 of nanoGPT
        # gives them a much smaller output variance than a dense layer of the same
        # width, and the signal crosses two projections instead of one. DeepSeek 3.1.2
        # says it uses additional scaling factors at the width bottlenecks for exactly
        # this reason. remediation.md #6 suggests std = fan_in ** -0.5.
        #
        # That fix overshoots, and mla_up_init='matched' is the alternative (audit M-2). The
        # input of an up-projection is an RMSNorm'd latent, i.e. unit RMS, so std = r^-1/2
        # gives k and v UNIT variance: 2.2x the std MHA gets from std=0.02 on an
        # n_embd-wide unit-RMS input. std = 0.02*sqrt(n_embd/r) makes the two equal, so the
        # MLA-vs-MHA loss gap stops including an init scale. 'bottleneck' stays the default
        # because every recorded run used it. Both draw the same normal_ call, so the RNG
        # stream -- and every other initial weight -- is identical under either choice.
        for m in self.modules():
            if getattr(m, '_is_bottleneck_up_proj', False):
                std = (0.02 * math.sqrt(config.n_embd / m.in_features)
                       if config.mla_up_init == 'matched' else m.in_features ** -0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and 'wpe' in self.transformer:
            # with RoPE there is no wpe to subtract; the count then means something
            # slightly different, which matters for the cross-cell parameter matching
            # of T3.5 (threats.md N7)
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope_slice(self, p0, t):
        """cos/sin for absolute positions [p0, p0+t), growing the table when needed.

        The table is built with headroom, but a rolling buffer means the model can be
        evaluated arbitrarily far past block_size -- that is the whole point of SWA, and
        the inference benchmark goes to 128k. Silently returning a short slice would
        raise an opaque shape error deep inside attention; growing is cheap (the table
        is a non-persistent buffer, so it never reaches the checkpoint) and keeps the
        length-extrapolation evaluation of T7.3 honest.
        """
        need = p0 + t
        if need > self.rope_cos.size(0):
            cos, sin = precompute_rope(self.rope_cos.size(1) * 2, 2 * need,
                                       self.config.rope_theta)
            self.register_buffer('rope_cos', cos.to(self.rope_cos.device), persistent=False)
            self.register_buffer('rope_sin', sin.to(self.rope_sin.device), persistent=False)
        return self.rope_cos[p0:need], self.rope_sin[p0:need]

    def forward(self, idx, targets=None, cache=None):
        device = idx.device
        b, t = idx.size()
        # Two reasons the length limit does not always apply. With a cache the model may
        # be evaluated far past block_size -- that is the point of a rolling buffer. And
        # with RoPE there is no learned position table to run off the end of: _rope_slice
        # grows the frequencies on demand, which is what makes the length-extrapolation
        # gate T7.3 (evaluate at 2x block_size) possible at all. Only learned absolute
        # positions are genuinely bounded, because wpe has exactly block_size rows.
        p0 = 0 if cache is None else cache.pos
        if cache is None and self.config.pos_encoding == 'learned':
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        cos = sin = None
        if self.config.pos_encoding == 'rope':
            # ABSOLUTE positions p0..p0+t, never the rolling-buffer slot index (C1)
            cos, sin = self._rope_slice(p0, t)
            x = self.transformer.drop(tok_emb)
        else:
            pos = torch.arange(p0, p0 + t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        rope = self._rope_slice if self.config.pos_encoding == 'rope' else None
        for block in self.transformer.h:
            x = block(x, cos, sin, cache, rope)
        if cache is not None:
            cache.advance(t)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        # GUARD: this surgery assumes learned position embeddings and the dense causal
        # buffer of vanilla nanoGPT; neither survives RoPE or MLA (mappa_modifiche 1.2).
        assert self.config.pos_encoding == 'learned', "crop_block_size assumes wpe"
        assert self.config.attn_type == 'mha', "crop_block_size assumes vanilla MHA"
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # GUARD: the OpenAI checkpoints are plain MHA with learned positions; the state
        # dict assert below cannot match an MLA/GQA model (threats.md N4).
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        assert config.attn_type == 'mha' and config.pos_encoding == 'learned', \
            "from_pretrained only supports the vanilla MHA + learned-position model"
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_flops(self, seq_len=None):
        """Training FLOPs per token, window-aware and parameterisation-aware (gate T5.5).

        nanoGPT uses `6*N + 12*L*H*Q*T` (PaLM appendix B). Both halves need fixing here:

        * the attention term assumes every query attends to all T positions. On a LOCAL
          layer it attends to min(T, W), so with W=256 at T=4096 the original formula
          overstates that layer's attention cost by 16x. Left uncorrected it reports an
          MFU that quietly drops as the window shrinks -- plausible numbers, wrong
          (threats.md N6). This is the window-aware accounting of nanochat.
        * `12*H*Q*T` assumes the score and the value share one head dimension. MLA has
          d_qk = qk_nope + qk_rope for the score and v_head_dim for the value, and they
          differ. The general form is 6*H*(d_qk + d_v)*T_eff per layer per token, which
          reduces to 12*H*Q*T when d_qk = d_v = Q.

        The 6*N term already covers every projection, MLA's extra ones included, since N
        counts parameters. Note that the number below is the ANALYTIC cost: on the MLA
        cells the kernel actually runs at the padded width (48 -> 64), so the measured
        wall clock reflects ~33% more attention work than this accounts for.
        """
        cfg = self.config
        T = seq_len or cfg.block_size
        head_dim = cfg.n_embd // cfg.n_head
        if cfg.attn_type == 'mla':
            d_qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
            d_v = cfg.v_head_dim
        else:
            d_qk = d_v = head_dim
        flops_attn = 0
        for is_local in resolve_pattern(cfg.attn_pattern, cfg.n_layer, cfg.force_last_global):
            t_eff = min(T, cfg.window_size) if is_local else T
            flops_attn += 6 * cfg.n_head * (d_qk + d_v) * t_eff
        return 6 * self.get_num_params() + flops_attn

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS

        Added after the fork -- the summary line above is upstream nanoGPT's and still
        holds; what changed underneath it is where flops_per_token comes from.

        Upstream computed it inline as `6*N + 12*L*H*Q*T`, the PaLM formula the comment
        below points at. It now comes from GPT.estimate_flops, which corrects both halves
        of that expression for this fork: attention is counted over min(T, window) on a
        local layer instead of over all T, and the score and value widths are allowed to
        differ (d_qk != d_v on MLA). On a global MHA layer with d_qk == d_v the two agree
        exactly, so this is not a change of units and MFU stays comparable to upstream's;
        on the SWA and MLA cells it is the difference between a plausible number and a
        correct one. See estimate_flops for the derivation and for what it deliberately
        does NOT model (the padded kernel width).
        """
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        T = self.config.block_size
        flops_per_token = self.estimate_flops(T)
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
