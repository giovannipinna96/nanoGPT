#!/bin/bash
# env.sh — environment pinning for this project (gate T0.2).
#
# Why this file exists: the cluster home filesystem (/u, NetApp /homes export) has
# NO FREE INODES (21.24M/21.25M used, ~10k free filesystem-wide). Byte-wise there is
# space, but unpacking a torch wheel (~10k small files) fails with ENOSPC. Everything
# that creates many small files (uv cache, virtualenv, HF datasets cache) therefore
# lives on /share, which has ~178M free inodes.
#
# Usage:  source env.sh   (before uv run / uv sync, on every node)

export HYBRID_ATTN_SCRATCH=/share/malelab/gpinna/hybrid-attn
export UV_CACHE_DIR="$HYBRID_ATTN_SCRATCH/uv-cache"
export UV_PROJECT_ENVIRONMENT="$HYBRID_ATTN_SCRATCH/venv"
export HF_HOME="$HYBRID_ATTN_SCRATCH/hf"
export HF_DATASETS_CACHE="$HYBRID_ATTN_SCRATCH/hf/datasets"
# training data (.bin memmaps) and checkpoints also live off /u
export HYBRID_ATTN_DATA="$HYBRID_ATTN_SCRATCH/data"
export HYBRID_ATTN_OUT="$HYBRID_ATTN_SCRATCH/out"
