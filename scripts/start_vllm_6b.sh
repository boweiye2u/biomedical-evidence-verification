#!/usr/bin/env bash
set -euo pipefail
SERVING_ENV="${RAG_ROOT:-$HOME/rag}/envs/serving"
export PATH="$SERVING_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
exec "$SERVING_ENV/bin/vllm" serve "${RAG_ROOT:-$HOME/rag}/models/Qwen2.5-7B-Instruct" \
  --served-model-name qwen2.5-7b-instruct --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.72 --max-num-seqs 16 --generation-config vllm \
  --host 127.0.0.1 --port 8001 --disable-log-requests
