# env.md - environment pin (gate T0.2)

Generated on `babbage.units.it` at 2026-09-11T13:18:44+02:00

## Hardware

| | |
|---|---|
| Cluster | Demetra (UniTS), SLURM partition `Main` |
| Node | babbage.units.it |
| GPU | NVIDIA A100-PCIE-40GB |
| Compute capability | sm80 |
| GPU memory | 39.5 GiB |
| Driver | 580.173.02 |
| CPU | Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz |

The table above describes the node this file was generated on. The campaign ran on two
A100 variants with the same GPU architecture (sm80), NVIDIA driver and Python / PyTorch / CUDA
stack; the hosts differ in GPU memory, CPU and OS minor release (per-job log headers):

| node | partition | GPU | host CPU, OS | used for |
|---|---|---|---|---|
| lovelace-01, lovelace-02 | `lovelace` | A100 80GB PCIe | AMD EPYC 7542, Rocky Linux 9.4 | 4 of the 12 grid runs (jobs 97727, 97756), all 6 follow-up runs (97929, 97930, 97964, 97965), every efficiency benchmark (the published Fase F and the `results/v2/` re-measurement, 98713-98715), every validation and test job of the campaign, the carved Fase H probe (97957) |
| babbage | `Main` | A100-PCIE-40GB | Intel Xeon Gold 6140, Rocky Linux 9.3 | 8 of the 12 grid runs (jobs 97747, 97755), the campaign Fase H probe (97832, 97833), the third-seed probe (98055), the GPU test suite after the audit fixes (98677) |

## Software

| | |
|---|---|
| OS | Rocky Linux release 9.3 (Blue Onyx) |
| Python | 3.12.11 |
| torch | 2.6.0+cu124 |
| torch CUDA | 12.4 |
| cuDNN | 90100 |
| numpy | 1.26.4 |
| transformers | 5.17.0 |
| uv | uv 0.9.10 |
| lockfile | `uv.lock` (committed) |

## Numerics and determinism

| | |
|---|---|
| TF32 in training | `train.py` sets `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` to **True**: every training run used TF32 |
| TF32 elsewhere (PyTorch defaults) | `matmul.allow_tf32` = False, `cudnn.allow_tf32` = True; `bench_inference.py` and `probe_longctx.py` do not change them |
| training dtype | bfloat16 (autocast), fp32 master weights |
| bf16 supported | True |
| seeds | 1337 (seed A) / 2024 (seed B); third seed 3141 for cells 4 and 9 only (STEP 0c); fixed val set built with seed 1234 |

> FlexAttention and FlashAttention have non-deterministic backward passes (atomic
> accumulation), so bit-exact reproducibility is **not** promised
> (`considerazioni_finali.md` 2.5). `sigma_seed` (T0.3) is measured with the same
> backend used for the grid, so it absorbs this source of noise.

## Attention backends

- FlexAttention (`torch.nn.attention.flex_attention`): **available** - used by every grid
  cell for training (`attn_impl='flex'`), so the training kernel is constant across cells
  (threat K3).
- flash-attn: available (2.7.4.post1) - optional extra (`uv sync --extra flash`), used by the STEP 0 tests and benchmarks, never by a grid cell
- SDPA with a dense boolean mask, or none when every key is visible
  (`attn_impl='sdpa_mask'`): always available. It is the oracle of `tests/` AND the backend
  of every inference number: `bench_inference.py` (Fase F and `results/v2/`), `probe_longctx.py`
  (Fase H) and the absorbed MLA decode path, which uses it whatever `attn_impl` says. The
  `results/v2/` latency runs put each call on the fastest SDPA kernel that accepts its shapes
  (`--sdpa-kernel fastest`, choices recorded in the CSVs).

## Storage layout

The home filesystem `/u` has **no free inodes** (21.24M of 21.25M used, cluster-wide),
so unpacking wheels there fails with ENOSPC even though bytes are available. Everything
that creates many small files lives on `/share` instead (see `env.sh`):

- `HYBRID_ATTN_SCRATCH` = `/share/malelab/gpinna/hybrid-attn`
- `UV_CACHE_DIR` = `/share/malelab/gpinna/hybrid-attn/uv-cache`
- `UV_PROJECT_ENVIRONMENT` = `/share/malelab/gpinna/hybrid-attn/venv`
- `HF_HOME` = `/share/malelab/gpinna/hybrid-attn/hf`
- `HYBRID_ATTN_DATA` = `/share/malelab/gpinna/hybrid-attn/data`
- `HYBRID_ATTN_OUT` = `/share/malelab/gpinna/hybrid-attn/out`
