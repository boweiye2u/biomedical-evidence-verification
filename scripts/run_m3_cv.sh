#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
ENV="${RAG_ROOT}/envs/posttraining"
CONFIG="configs/posttraining/train-m3-full-ft-v1.json"
for fold in 0 1 2 3 4; do
  output="${RAG_ROOT}/runs/posttraining/full-ft-v1/cv/fold-${fold}"
  checkpoint="${RAG_ROOT}/checkpoints/posttraining/m3/cv/fold-${fold}"
  log="${RAG_ROOT}/logs/m3-cv-fold-${fold}.log"
  CUDA_VISIBLE_DEVICES=0,1,2,3 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$ENV/bin/torchrun" --standalone --nproc_per_node=4 -m posttraining.training.m3_full train \
      --mode cv --config "$CONFIG" --output "$output" --checkpoint "$checkpoint" \
      --seed 20261019 --epochs 3 --fold "$fold" --save-epochs 2 3 --expected-accumulation 2 >"$log" 2>&1
done
