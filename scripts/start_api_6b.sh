#!/usr/bin/env bash
set -euo pipefail
SERVING_ENV="${RAG_ROOT:-$HOME/rag}/envs/serving"
export PATH="$SERVING_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$SERVING_ENV/bin/uvicorn" serving.app:app --host 127.0.0.1 --port 8000
