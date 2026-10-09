#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
ENV="${RAG_ROOT}/envs/posttraining"
RUN="${RAG_ROOT}/runs/posttraining/full-ft-v1/dev-finalists"
CKPT="${RAG_ROOT}/checkpoints/posttraining/m3/dev-finalists"
mkdir -p "${RAG_ROOT}/logs"
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$ENV/bin/torchrun" --standalone --nproc_per_node=4 -m posttraining.training.m3_full train \
    --mode final --seed "$seed" --epochs 3 --save-epochs 2 3 --expected-accumulation 2 \
    --output "$RUN/seed-$seed" --checkpoint "$CKPT/seed-$seed" \
    >"${RAG_ROOT}/logs/m3-final-seed-${seed}.log" 2>&1
done
