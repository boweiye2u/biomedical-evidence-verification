#!/usr/bin/env bash
set -euo pipefail
ENV="/home/boweiye2/rag/envs/serving";export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 POSTTRAINING_SERVING_CONFIG="configs/posttraining/serving-tuned-3b-v1.json"
exec "$ENV/bin/uvicorn" serving.posttraining_app:app --host 127.0.0.1 --port 8000
