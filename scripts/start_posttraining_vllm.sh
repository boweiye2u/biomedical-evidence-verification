#!/usr/bin/env bash
set -euo pipefail
ENV="/home/boweiye2/rag/envs/serving";MODEL="/home/boweiye2/rag/checkpoints/posttraining/m2/dev-finalists/mix-2-epoch-3/seed-0/merged"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
exec "$ENV/bin/vllm" serve "$MODEL" --served-model-name posttraining-m2-3b --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.72 --max-num-seqs 16 --generation-config vllm --host 127.0.0.1 --port 8001 --disable-log-requests
