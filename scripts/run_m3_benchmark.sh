#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
POST_ENV="${RAG_ROOT}/envs/posttraining"
SERVE_ENV="${RAG_ROOT}/envs/serving"
CONFIG="configs/posttraining/m3-benchmark-v1.json"
MODEL="$($POST_ENV/bin/python -c 'import json;print(json.load(open("configs/posttraining/m3-benchmark-v1.json"))["systems"]["M3"]["model_path"])')"
LOG="${RAG_ROOT}/logs/m3-benchmark-vllm.log"
server_pid=""
cleanup(){ if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then kill -INT "$server_pid"; wait "$server_pid" || true; fi; }
trap cleanup EXIT
"$POST_ENV/bin/python" -m posttraining.evaluation.m3_benchmark preflight --config "$CONFIG"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$SERVE_ENV/bin/vllm" serve "$MODEL" --served-model-name posttraining-benchmark-m3 --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.72 --max-num-seqs 16 \
  --generation-config vllm --host 127.0.0.1 --port 8001 --disable-log-requests >"$LOG" 2>&1 &
server_pid=$!
ready=0
for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then cat "$LOG" >&2; exit 1; fi
  sleep 1
done
[[ "$ready" == 1 ]] || { echo "vLLM readiness timeout" >&2; exit 1; }
"$POST_ENV/bin/python" -m posttraining.evaluation.m3_benchmark generate --config "$CONFIG"
kill -INT "$server_pid"; wait "$server_pid" || true; server_pid=""
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$POST_ENV/bin/python" -m posttraining.evaluation.m3_benchmark calibrate --config "$CONFIG" --batch-size 1
"$POST_ENV/bin/python" -m posttraining.evaluation.m3_benchmark finalize --config "$CONFIG"
