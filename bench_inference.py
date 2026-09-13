"""Fase F - inference efficiency, measured BEFORE any training (gates T5.1-T5.3).

An untrained model has exactly the same KV cache as a trained one, so the most
convincing figure of the report costs no training time at all. If the GPU budget ran out
here, half the deliverable would already be in hand (plan.md, Fase F).

Two regimes, deliberately kept in separate tables (threats.md B9):

  MEMORY  - exact arithmetic, kernel-independent. Measured against the allocator to
            confirm the formula, then reported analytically.
  LATENCY - kernel-dependent. Every table here runs on --impl sdpa_mask (dense mask + SDPA,
            uncompiled), so the flex padding of MLA's 48-wide score to 64 does NOT apply
            to these numbers: it costs training throughput only (T5.4, audit M-3). What
            keeps MLA's latency a lower bound is the kernel: the specialised MLA kernels
            (FlashMLA, the vLLM integrations) need SM90 while this is an A100 (sm80). The
            naive decode (--absorb never) also rebuilds k and v for every cached token;
            the absorbed form (--absorb always) is measured separately (STEP 0b). Neither
            factor touches the memory.

    uv run python bench_inference.py --tests cache,latency      # on a GPU node
"""
import argparse
import csv
import gc
import json
import os
import time

import torch

from cells import select
from kv_cache import CacheSpec, HybridKVCache
from model import (GPT, GPTConfig, reset_sdpa_kernel_choices, sdpa_kernel_choices,
                   sdpa_kernel_policy)

# Ordered: the first cell is the reference every "x MHA" column divides by. The STEP 0b
# variants of cell 4 come last -- the comparison L1 asks for: MLA's cache saving is exact,
# but it only becomes throughput once the up-projection stops running per cached token.
CELLS = select(["1_mha_full", "2_mha_swa", "5_gqa_full", "3_mla_full", "4_mla_swa",
                "7_mla_all_local", "4_mla_swa_carved", "4_mla_swa_recon"])
BASE = dict(n_layer=8, n_head=16, n_embd=512, vocab_size=50304, dropout=0.0, bias=False,
            pos_encoding='rope', window_size=256, kv_lora_rank=256, block_size=1024)


def make_cfg(cell, block_size, impl, absorb='auto'):
    return GPTConfig(**dict(BASE, block_size=block_size, attn_impl=impl),
                     **dict(CELLS[cell], **({'absorb_mode': absorb}
                                            if CELLS[cell]['attn_type'] == 'mla'
                                            and CELLS[cell].get('rope_mode')
                                            != 'reconstructed' else {})))


def make_cache(cfg, batch_size, max_seq_len, device, dtype):
    return HybridKVCache(CacheSpec.from_config(cfg, max_seq_len), batch_size, device, dtype)


def build_model(cfg):
    """The model every benchmark below runs: bf16 on the GPU, eval mode, and NO autograd.

    Everything in this file measures inference. A forward with grad enabled keeps every
    layer's activations alive for a backward that never runs, and the memory numbers then
    measure those activations instead of the KV cache: T5.3b was once recorded that way,
    with 39 GB (MHA) to 69 GB (MLA) of retained activations next to the cache at T=8192
    (audit A-1). Two independent guards: callers run under torch.no_grad(), and the
    parameters do not require grad, so a caller that forgets the decorator records nothing.
    """
    assert not torch.is_grad_enabled(), (
        "benchmark model built with autograd enabled: run the caller under torch.no_grad()")
    return GPT(cfg).cuda().eval().to(torch.bfloat16).requires_grad_(False)


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def ref_cell():
    """The cell every "x MHA" column divides by, or the first selected one if MHA is not
    among them (--cells). Bound BY NAME on purpose: binding it to whichever cell happens to
    produce the first number means that an OOM on MHA -- the largest cell, so the first to
    run out -- silently makes the next cell the reference while the header still says MHA."""
    return '1_mha_full' if '1_mha_full' in CELLS else next(iter(CELLS))


def ref_label():
    """Header of the ratio column, e.g. 'x MHA'. Names the cell actually divided by."""
    base = ref_cell()
    return 'x MHA' if base == '1_mha_full' else f"x {base.split('_', 1)[1]}"


def ratio_col(value, ref, width=6, unit='x'):
    """`value / ref` for the ratio column, or an em dash when the reference has no number
    (it went OOM, or has not run yet). Same convention as analysis/pareto_plot.fmt."""
    if not ref:
        return f"{'--':>{width}s} "
    return f"{value / ref:{width}.2f}{unit}"


def write_kernels(test, cell, T, B, writer):
    """One row per SDPA call signature of the run just finished: the kernel it ran, the one
    PyTorch's dispatcher would have picked, and every kernel's time (JSON in `value`)."""
    for c in sdpa_kernel_choices().values():
        writer(dict(test=test, cell=cell, seq_len=T, batch_size=B, metric='sdpa_kernel',
                    value=json.dumps(c, sort_keys=True)))


def kernel_summary():
    """The decode kernels of the run just finished, e.g. 'S<=256 FLASH_ATTENTION', with the
    dispatcher's pick next to any call where it differs."""
    parts = []
    for c in sorted(sdpa_kernel_choices().values(), key=lambda c: c['S_max']):
        if c['T'] == 1:
            parts.append(f"S<={c['S_max']} {c['fastest']}" + (
                "" if c['fastest'] == c['dispatcher'] else f" (dispatcher {c['dispatcher']})"))
    return ', '.join(parts)


# --------------------------------------------------------------------------- T5.1
@torch.no_grad()
def bench_cache(args, writer):
    """Measured cache memory against the analytic formula, per cell and per length."""
    print("\n=== T5.1  KV cache: measured vs analytic ===")
    base = ref_cell()
    print(f"{'cell':17s} {'T':>8s} {'measured MB':>12s} {'analytic MB':>12s} "
          f"{'diff':>7s} {'% of ' + ref_label()[2:]:>9s}")
    for T in args.cache_lengths:
        ref = None
        for cell in CELLS:
            cfg = make_cfg(cell, min(T, 1024), args.impl, args.absorb)
            free()
            before = torch.cuda.memory_allocated()
            cache = make_cache(cfg, args.batch_size, T, 'cuda', torch.bfloat16)
            measured = (torch.cuda.memory_allocated() - before) / 1024 ** 2
            analytic = cache.spec.bytes(T, args.batch_size, 2) / 1024 ** 2
            if cell == base:
                ref = analytic
            pct = '      --' if not ref else f"{100 * analytic / ref:8.1f}"
            print(f"{cell:17s} {T:8d} {measured:12.2f} {analytic:12.2f} "
                  f"{100 * abs(measured - analytic) / analytic:6.2f}% "
                  f"{pct}%")
            writer(dict(test='T5.1', cell=cell, seq_len=T, batch_size=args.batch_size,
                        metric='cache_mb_measured', value=round(measured, 3)))
            writer(dict(test='T5.1', cell=cell, seq_len=T, batch_size=args.batch_size,
                        metric='cache_mb_analytic', value=round(analytic, 3)))
            del cache
    free()


# --------------------------------------------------------------------------- T5.2
@torch.no_grad()
def bench_latency(args, writer):
    """Decode latency and throughput. compile=False on purpose: a compiled model
    recompiles on every new sequence length during decoding, and we would be measuring
    the compiler instead of the model (threats.md K4).

    Kernel: --sdpa-kernel, 'fastest' by default. Every SDPA call runs the fastest kernel
    that accepts its shapes, chosen during the prefill and the warm-up, and the choice is
    written next to the timing (metric sdpa_kernel): a ratio between two cells cannot hide
    a kernel difference."""
    with sdpa_kernel_policy(args.sdpa_kernel):
        _latency(args, writer)


def _latency(args, writer):
    print("\n=== T5.2  decode latency and throughput ===")
    base = ref_cell()
    print(f"{'cell':17s} {'ctx':>7s} {'B':>4s} {'ms/token':>10s} {'tok/s':>10s} "
          f"{ref_label():>7s}  decode kernels")
    for T in args.latency_lengths:
        for B in args.batch_sizes:
            ref = None
            for cell in CELLS:
                cfg = make_cfg(cell, 1024, args.impl, args.absorb)
                free()
                torch.manual_seed(0)
                model = build_model(cfg)
                reset_sdpa_kernel_choices()
                cache = None
                try:
                    # Allocated inside the try: at B=64, T=32k the cache is tens of GB and is
                    # the most likely thing here to run out of memory. Outside, an OOM on it
                    # took the whole benchmark down instead of printing one OOM row --
                    # bench_kernels already allocated it inside its own try.
                    # Sized for T + decode_steps only, the global layers wrapped during the
                    # last `warmup` timed steps (audit M-5).
                    cache = make_cache(cfg, B, T + args.warmup + args.decode_steps, 'cuda',
                                       torch.bfloat16)
                    idx = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')
                    model(idx, cache=cache)                       # prefill
                    nxt = torch.randint(0, cfg.vocab_size, (B, 1), device='cuda')
                    for _ in range(args.warmup):                  # warmup (B2)
                        model(nxt, cache=cache)
                    torch.cuda.synchronize()                      # (B1)
                    t0 = time.perf_counter()
                    for _ in range(args.decode_steps):
                        model(nxt, cache=cache)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    ms = dt / args.decode_steps * 1000
                    tps = B * args.decode_steps / dt
                    if cell == base:
                        ref = tps
                    print(f"{cell:17s} {T:7d} {B:4d} {ms:10.3f} {tps:10.1f} "
                          f"{ratio_col(tps, ref)}  {kernel_summary()}")
                    writer(dict(test='T5.2', cell=cell, seq_len=T, batch_size=B,
                                metric='ms_per_token', value=round(ms, 4)))
                    writer(dict(test='T5.2', cell=cell, seq_len=T, batch_size=B,
                                metric='tokens_per_s', value=round(tps, 2)))
                    write_kernels('T5.2', cell, T, B, writer)
                except torch.cuda.OutOfMemoryError:
                    print(f"{cell:17s} {T:7d} {B:4d}        OOM")
                    writer(dict(test='T5.2', cell=cell, seq_len=T, batch_size=B,
                                metric='ms_per_token', value=''))
                del model, cache
                free()


@torch.no_grad()
def bench_kernels(args, writer):
    """Which SDPA kernel every decode call of T5.2 runs under 'fastest', and whether PyTorch's
    dispatcher picks the same one -- without timing the decode.

    Every latency recorded before --sdpa-kernel existed ran on the dispatcher. Where the
    dispatcher already picks the fastest kernel for every decode call of a cell, that cell's
    numbers are what 'fastest' would measure; where it does not, they need re-measuring. The
    cache is brought to position T without a prefill (as in T5.3b) and filled with random
    entries, then two decode steps run: the calls, shapes and tensor layouts of the timed
    loop of T5.2, at a fraction of its cost.
    """
    print("\n=== T5.2k  SDPA kernel of every decode call: fastest vs dispatcher ===")
    print(f"{'cell':17s} {'ctx':>7s} {'B':>4s}  decode kernels")
    with sdpa_kernel_policy('fastest'):
        for T in args.latency_lengths:
            for B in args.batch_sizes:
                for cell in CELLS:
                    cfg = make_cfg(cell, 1024, args.impl, args.absorb)
                    free()
                    torch.manual_seed(0)
                    model = build_model(cfg)
                    reset_sdpa_kernel_choices()
                    cache = None
                    try:
                        cache = make_cache(cfg, B, T + 2, 'cuda', torch.bfloat16)
                        for layer in cache.layers:
                            for buf in layer.buffers.values():
                                buf.normal_()
                        cache.advance(T)
                        idx = torch.randint(0, cfg.vocab_size, (B, 1), device='cuda')
                        for _ in range(2):
                            model(idx, cache=cache)
                        same = all(c['fastest'] == c['dispatcher']
                                   for c in sdpa_kernel_choices().values())
                        print(f"{cell:17s} {T:7d} {B:4d}  {kernel_summary()}"
                              + ("" if same else "   <- the dispatcher picks another kernel"))
                        write_kernels('T5.2k', cell, T, B, writer)
                    except torch.cuda.OutOfMemoryError:
                        print(f"{cell:17s} {T:7d} {B:4d}  OOM")
                    del model, cache
                    free()


# --------------------------------------------------------------------------- T5.3
@torch.no_grad()
def bench_max_batch(args, writer):
    """Largest batch that fits at a fixed context, by bisection.

    In production throughput is limited by the batch and the batch is limited by the KV
    cache, so this is the number that translates "7% of the cache" into "N times more
    requests served per GPU".

    Kernel: PyTorch's dispatcher, whatever --sdpa-kernel says, here and in T5.3b. Which
    kernel is fastest does not change what fits, and timing candidate kernels inside the
    bisection would add their transient allocations to the very number being measured.
    """
    print("\n=== T5.3  maximum batch at fixed context ===")
    base = ref_cell()
    print(f"{'cell':17s} {'ctx':>7s} {'max batch':>10s} {ref_label():>7s}")
    for T in args.batch_lengths:
        ref = None
        for cell in CELLS:
            cfg = make_cfg(cell, 1024, args.impl, args.absorb)

            def fits(B):
                free()
                try:
                    torch.manual_seed(0)
                    model = build_model(cfg)
                    cache = make_cache(cfg, B, T + 1, 'cuda', torch.bfloat16)
                    idx = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')
                    model(idx, cache=cache)
                    model(idx[:, :1], cache=cache)
                    ok = True
                except torch.cuda.OutOfMemoryError:
                    ok = False
                free()
                return ok

            lo, hi = 1, args.max_batch_probe
            if not fits(lo):
                best = 0
            else:
                while lo < hi:                      # bisection on the largest feasible B
                    mid = (lo + hi + 1) // 2
                    if fits(mid):
                        lo = mid
                    else:
                        hi = mid - 1
                best = lo
            if cell == base:
                ref = best
            # ref == 0 means the reference cell did not fit even at B=1: max(ref, 1) used to
            # turn the column into an absolute count that still read "x MHA".
            print(f"{cell:17s} {T:7d} {best:10d} {ratio_col(best, ref)}")
            writer(dict(test='T5.3', cell=cell, seq_len=T, batch_size=best,
                        metric='max_batch', value=best))


# ------------------------------------------------------------------- T5.3 (decode)
@torch.no_grad()
def bench_max_batch_decode(args, writer):
    """Max batch in the STEADY-STATE serving regime: cache full at T, one decode step.

    Why this exists. The variant above prefills the whole prompt in a single forward,
    and at large B that transient dominates: at the batch it reports, the MLA+SWA cache
    holds 5.5 GB of the 80 available, so the other 74 GB are prefill activations. The
    number it produces (2.10x) therefore measures the prefill, not the cache, and the
    5-20x of test_todo.md T5.3 -- which is explicitly about "il batch e limitato dalla
    KV cache" -- is not what the probe was constraining.

    Here the cache is brought to position T WITHOUT running a prefill: the buffers are
    already allocated at full capacity by HybridKVCache, so advancing `pos` puts the
    model in exactly the memory state it would be in after serving T tokens. Then one
    decode step runs. This is the regime a serving stack is actually in, and it is where
    the cache is the binding constraint.

    Both numbers are reported. They answer different questions and the report needs both:
    the prefill-limited one bounds a batch you admit all at once, the decode-limited one
    bounds the batch you can keep resident.
    """
    print("\n=== T5.3b  maximum batch at fixed context, decode-limited ===")
    base = ref_cell()
    print(f"{'cell':17s} {'ctx':>7s} {'max batch':>10s} {ref_label():>7s} "
          f"{'cache GB':>9s}")
    for T in args.batch_lengths:
        ref = None
        for cell in CELLS:
            cfg = make_cfg(cell, 1024, args.impl, args.absorb)

            def fits(B):
                free()
                try:
                    torch.manual_seed(0)
                    model = build_model(cfg)
                    cache = make_cache(cfg, B, T + 1, 'cuda', torch.bfloat16)
                    cache.advance(T)         # full at T, no prefill transient
                    idx = torch.randint(0, cfg.vocab_size, (B, 1), device='cuda')
                    model(idx, cache=cache)
                    ok = True
                except torch.cuda.OutOfMemoryError:
                    ok = False
                free()
                return ok

            lo, hi = 1, args.max_batch_probe_decode
            if not fits(lo):
                best = 0
            else:
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if fits(mid):
                        lo = mid
                    else:
                        hi = mid - 1
                best = lo
            if cell == base:
                ref = best
            spec_bytes = make_cache(cfg, 1, T + 1, 'cpu', torch.bfloat16).n_bytes()
            gb = spec_bytes * best / 1024 ** 3
            print(f"{cell:17s} {T:7d} {best:10d} {ratio_col(best, ref)} {gb:9.2f}")
            writer(dict(test='T5.3b', cell=cell, seq_len=T, batch_size=best,
                        metric='max_batch_decode', value=best))
            writer(dict(test='T5.3b', cell=cell, seq_len=T, batch_size=best,
                        metric='cache_gb_at_max_batch', value=round(gb, 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="cache,latency,max_batch")
    ap.add_argument("--impl", default="sdpa_mask",
                    help="decoding runs uncompiled, where the dense path is the honest "
                         "reference; flex needs compilation to be fast (K5)")
    ap.add_argument("--cache-lengths", type=int, nargs="+",
                    default=[1024, 4096, 16384, 65536, 131072])
    ap.add_argument("--latency-lengths", type=int, nargs="+", default=[1024, 8192, 32768])
    ap.add_argument("--batch-lengths", type=int, nargs="+", default=[8192])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--batch-size", type=int, default=1, help="for the cache table")
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-batch-probe", type=int, default=4096)
    ap.add_argument("--max-batch-probe-decode", type=int, default=32768)
    ap.add_argument("--cells", default="", help="comma-separated subset of CELLS; empty = all")
    # default "never": every recorded T5.2/T5.3 number was measured on the naive decode path,
    # so a bare re-run must reproduce them. GPTConfig keeps "auto"; absorbed numbers are
    # measured by asking for them (STEP 0b, results/STEP0b_*).
    ap.add_argument("--absorb", default="never", choices=["auto", "never", "always"],
                    help="MLA decode path: 'auto' follows T*, the other two force it. "
                         "Forcing both and diffing is how the absorbed form is measured "
                         "rather than asserted (gate T3.7 proves they agree).")
    ap.add_argument("--sdpa-kernel", default="fastest", choices=["fastest", "dispatcher"],
                    help="latency: 'fastest' runs, for every SDPA call, the fastest kernel that "
                         "accepts its shapes; 'dispatcher' keeps PyTorch's priority list, which "
                         "every latency recorded before this option ran on. The max-batch "
                         "tests always use the dispatcher. --tests kernels compares the two "
                         "without timing the decode.")
    ap.add_argument("--out", default="results/T5_inference.csv")
    args = ap.parse_args()

    if args.cells:
        keep = args.cells.split(",")
        unknown = [c for c in keep if c not in CELLS]
        assert not unknown, f"unknown cells {unknown}; known: {list(CELLS)}"
        for name in list(CELLS):
            if name not in keep:
                del CELLS[name]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fh = open(args.out, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=['test', 'cell', 'seq_len', 'batch_size',
                                       'metric', 'value'])
    w.writeheader()

    def writer(row):
        w.writerow(row)
        fh.flush()

    print(f"device: {torch.cuda.get_device_name(0)}  impl={args.impl}  dtype=bfloat16  "
          f"sdpa_kernel={args.sdpa_kernel}")
    tests = args.tests.split(",")
    if "cache" in tests:
        bench_cache(args, writer)
    if "kernels" in tests:
        bench_kernels(args, writer)
    if "latency" in tests:
        bench_latency(args, writer)
    if "max_batch" in tests:
        bench_max_batch(args, writer)
    if "max_batch_decode" in tests:
        bench_max_batch_decode(args, writer)
    fh.close()
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
