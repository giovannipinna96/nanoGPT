#!/bin/bash
# run_all.sh -- one command per result (test_todo.md T8.3).
#
#   bash run_all.sh <target> [args]           run here (most targets need an A100)
#
# Every command is the one the campaign ran, and every result it writes goes under results/.
#
#   env                  T0.2        env.md
#   data                 T0.4        FineWeb-Edu sample-10BT, 2 shards; shakespeare_char
#   tests                            every unit-test gate (CUDA tests skip without a GPU)
#   baseline             T0.1        unmodified nanoGPT, shakespeare_char (1.4697 +/- 0.02)
#   validation           T1.1 T1.3 T2.3 T2.5 T3.3   the small correctness runs
#   bench                Fase F      T5.1-T5.5, on UNTRAINED models (before any training)
#   cell <cell> <seed>   T6.1        one grid run, e.g.  bash run_all.sh cell 4_mla_swa 1337
#   grid                 T6.1        the six campaign cells x seeds 1337/2024 (12 runs, ~1 h each)
#   faseh                Fase H      E8, T7.1-T7.3
#   analysis             T0.3 T3.5 T6.2-T6.4 T7 T8.1 T8.2   tables and figures (CPU only)
#   step0c                           carved cells 8 and 9 x 2 seeds, their analysis, their Fase H
#   seed3                            third seed (3141) for cells 4 and 9, its Fase H, 3-seed reading
#   remeasure            T5.2-T5.3b T8.1 T8.2   benchmarks after the audit fixes -> results/v2/
#   all                              everything above except `cell`
#
# Paths come from env.sh (HYBRID_ATTN_OUT etc.), which is specific to Demetra: /u has no
# free inodes, so the venv, the data and the checkpoints live on /share. Edit env.sh first
# anywhere else. The flash tests need `uv sync --extra flash` (see pyproject.toml).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source env.sh
OUT="${HYBRID_ATTN_OUT:-out}"
SEEDS=(1337 2024)
CAMPAIGN=(1_mha_full 2_mha_swa 3_mla_full 4_mla_swa 5_gqa_full 7_mla_all_local)
CARVED=(8_mla_swa_carved 9_mla_swa_carved16)
ALL8="$(IFS=,; echo "${CAMPAIGN[*]},${CARVED[*]}")"

gpu() { echo "+ $1"; bash -c "$1"; }   # needs an A100
cpu() { echo "+ $1"; bash -c "$1"; }   # CPU only

t_env()        { gpu "uv run python scripts/env_report.py > env.md"; }
t_data()       { cpu "uv run python data/finewebedu/prepare.py --shards 2"
                 cpu "uv run python data/shakespeare_char/prepare.py"; }
t_tests()      { gpu "uv run pytest tests/ -q"; }
t_baseline()   { gpu "uv run python train.py config/train_shakespeare_char.py --out_dir=$OUT/shakespeare-char-baseline"; }

t_validation() {
  local S="config/train_shakespeare_char.py --dropout=0.0"
  # T1.1: the refactor must not move the unmodified model (init checksum and 20 losses)
  gpu "uv run python scripts/regression_check.py --compare results/T1.1_reference.json"
  # T1.3: RoPE trains, and flex and the dense oracle agree
  gpu "uv run python train.py $S --pos_encoding=rope --attn_impl=sdpa_mask --max_iters=500 --lr_decay_iters=5000 --eval_interval=250 --out_dir=$OUT/t13-rope-sdpa_mask"
  gpu "uv run python train.py $S --pos_encoding=rope --attn_impl=flex --max_iters=500 --lr_decay_iters=5000 --eval_interval=250 --out_dir=$OUT/t13-rope-flex"
  # T2.3: real sparsity -- ms/step must fall as the window shrinks
  gpu "bash analysis/sweep_window.sh"
  gpu "bash analysis/sweep_window.sh results/T2.3_window_sweep_T8192.csv 8192 4 LLLG 8192 1024 256"
  gpu "bash analysis/sweep_window.sh results/T2.3_window_sweep_allloc.csv 4096 8 LLLLLLLG 4096 1024 256"
  # T2.5: SWA trains within +0.10 of the global baseline
  gpu "uv run python train.py $S --max_iters=2000 --lr_decay_iters=2000 --eval_interval=250 --attn_impl=flex --pos_encoding=rope --attn_type=mha --attn_pattern=G --block_size=256 --out_dir=$OUT/t25-ref"
  gpu "uv run python train.py $S --max_iters=2000 --lr_decay_iters=2000 --eval_interval=250 --attn_impl=flex --pos_encoding=rope --attn_type=mha --attn_pattern=LLLG --window_size=64 --block_size=256 --out_dir=$OUT/t25-swa"
  # T3.3: full-rank MLA (no compression, no decoupled RoPE) lands within 3% of MHA
  gpu "uv run python train.py $S --max_iters=2000 --lr_decay_iters=2000 --eval_interval=250 --attn_impl=flex --attn_type=mha --pos_encoding=learned --out_dir=$OUT/t33-mha"
  gpu "uv run python train.py $S --max_iters=2000 --lr_decay_iters=2000 --eval_interval=250 --attn_impl=flex --attn_type=mla --pos_encoding=learned --kv_lora_rank=384 --qk_rope_head_dim=0 --out_dir=$OUT/t33-mla"
}

t_bench() {
  # --absorb never: Fase F was measured before matrix absorption existed (naive decode)
  # kv_cache_analytic.py now also lists cells 8 and 9: the 30 recorded rows come back
  # unchanged, followed by 10 carved rows
  cpu "uv run python analysis/kv_cache_analytic.py --out results/T5.1_kv_cache_analytic.csv"
  gpu "uv run python bench_inference.py --tests latency,max_batch --absorb never --out results/T5_inference.csv"
  gpu "bash analysis/step9c.sh"                                  # T5.1 measured, T5.4, T5.5
  gpu "uv run python bench_inference.py --tests max_batch_decode --absorb never --max-batch-probe-decode 65536 --out results/T5.3b_max_batch_decode.csv"
  gpu "uv run python bench_inference.py --cells 7_mla_all_local --tests cache,latency,max_batch_decode --absorb never --max-batch-probe-decode 65536 --out results/T5_cell7_alllocal.csv"
}

t_cell() { gpu "bash analysis/run_cell.sh $1 $2"; }                # idempotent: skips a DONE run
t_grid() { for c in "${CAMPAIGN[@]}"; do for s in "${SEEDS[@]}"; do t_cell "$c" "$s"; done; done; }
t_faseh() { gpu "bash analysis/run_faseh.sh"; }

t_analysis() {
  # grid_analysis pools sigma over the cells it is given: the default is the campaign's six,
  # which is what reproduces 2*sigma = 0.0138. Never point it at results/ with --cells.
  # The efficiency tables and the Pareto plot are built from results/v2/ (the remeasure
  # target), the numbers the report uses.
  cpu "uv run python analysis/param_table.py"
  cpu "uv run python analysis/grid_analysis.py --root $OUT/grid --out results/"
  cpu "uv run python analysis/loss_curves.py"
  cpu "uv run python analysis/faseh_analysis.py --out results"
  cpu "uv run python analysis/pareto_plot.py --efficiency v2 --decode naive --out results/v2/ && uv run python analysis/pareto_plot.py --efficiency v2 --decode absorbed --out results/v2/"
}

t_step0c() {
  # the reading of every number below is fixed in results/STEP0c_carving_preregistration.md
  for c in "${CARVED[@]}"; do for s in "${SEEDS[@]}"; do t_cell "$c" "$s"; done; done
  cpu "mkdir -p results/step0c && uv run python analysis/grid_analysis.py --root $OUT/grid --cells $ALL8 --out results/step0c"
  gpu "bash analysis/run_faseh_carved.sh"
}

t_seed3() {
  # addendum 2 of results/STEP0c_carving_preregistration.md
  t_cell 4_mla_swa 3141
  t_cell 9_mla_swa_carved16 3141
  gpu "bash analysis/run_faseh_seed3.sh"
}

t_remeasure() {
  # T5.2, T5.3 and T5.3b of all eight benchmark cells after the audit fixes (A-1, A-2, M-5,
  # B-1 and the GQA copy), with the fastest SDPA kernel for every call shape
  # (bench_inference.py --sdpa-kernel, default 'fastest'), then T8.1 and T8.2 from them.
  #
  # The absorbed column is measured with --absorb auto, NOT always, and is labelled
  # "decode absorbed (prefill naive)". auto prefills in the naive form and decodes in the
  # absorbed one (T=1 is far below T*), which is how a server runs MLA. The latent cache
  # left by the prefill is the same either way, so the timed decode is the one 'always'
  # would time. 'always' also forces the absorbed form on the prefill, whose per-head
  # widths of 272/256 instead of 48/32 cost ~3.9x the memory per sequence at T=32768 and
  # OOM at B=64 on 80 GB while the absorbed decode itself would fit (measured in the audit).
  local L="--latency-lengths 1024 8192 32768 --batch-sizes 1 64"
  gpu "uv run python bench_inference.py --tests latency --absorb never $L --out results/v2/T5.2_latency_never.csv && uv run python bench_inference.py --tests latency --absorb auto $L --out results/v2/T5.2_latency_auto.csv"
  gpu "uv run python bench_inference.py --tests max_batch --absorb never --batch-lengths 8192 --out results/v2/T5.3_max_batch_prefill.csv"
  gpu "uv run python bench_inference.py --tests max_batch_decode --absorb never --batch-lengths 8192 --max-batch-probe-decode 262144 --out results/v2/T5.3b_max_batch_decode_never.csv && uv run python bench_inference.py --tests max_batch_decode --absorb auto --batch-lengths 8192 --max-batch-probe-decode 262144 --out results/v2/T5.3b_max_batch_decode_auto.csv"
  cpu "uv run python analysis/pareto_plot.py --efficiency v2 --decode naive --out results/v2/ && uv run python analysis/pareto_plot.py --efficiency v2 --decode absorbed --out results/v2/"
}

TARGET="${1:-}"; shift || true
case "$TARGET" in
  env|data|tests|baseline|validation|bench|grid|faseh|analysis|step0c|seed3|remeasure) "t_$TARGET" ;;
  cell)  [ $# -eq 2 ] || { echo "usage: bash run_all.sh cell <cell> <seed>"; exit 2; }; t_cell "$1" "$2" ;;
  all)   for t in env data tests baseline validation bench grid faseh analysis step0c seed3 remeasure; do
           TARGET=$t; "t_$t"; done ;;
  *)     awk 'NR > 1 && !/^#/ {exit} NR > 1' "$0"; exit 2 ;;
esac
