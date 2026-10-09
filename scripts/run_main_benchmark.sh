#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
POST_ENV="${RAG_ROOT}/envs/posttraining"
SERVE_ENV="${RAG_ROOT}/envs/serving"
RUN_ROOT="${RAG_ROOT}/runs/posttraining/benchmark-eval-v1"
CONFIG="configs/posttraining/final-eval-v1.json"
mkdir -p "${RAG_ROOT}/logs"
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -INT "${server_pid}" || true
    wait "${server_pid}" || true
  fi
}
trap cleanup EXIT
model_path() {
  case "$1" in
    B0|B1) echo "${RAG_ROOT}/models/Qwen2.5-7B-Instruct" ;;
    B2|B3) echo "${RAG_ROOT}/models/Qwen2.5-3B-Instruct" ;;
    M1) echo "${RAG_ROOT}/checkpoints/posttraining/m1/dev-finalists/seed-0/merged" ;;
    M2) echo "${RAG_ROOT}/checkpoints/posttraining/m2/dev-finalists/mix-2-epoch-3/seed-0/merged" ;;
  esac
}
for system in B0 B1 B2 B3 M1 M2; do
  served="posttraining-benchmark-${system,,}"
  log="${RAG_ROOT}/logs/posttraining-benchmark-${system}.vllm.log"
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    "${SERVE_ENV}/bin/vllm" serve "$(model_path "$system")" \
    --served-model-name "$served" --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization 0.72 --max-num-seqs 16 --generation-config vllm \
    --host 127.0.0.1 --port 8001 --disable-log-requests >"$log" 2>&1 &
  server_pid=$!
  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then cat "$log" >&2; exit 1; fi
    sleep 1
  done
  [[ "$ready" == 1 ]] || { echo "vLLM readiness timeout for ${system}" >&2; exit 1; }
  "${POST_ENV}/bin/python" -m posttraining.evaluation.benchmark generate \
    --config "$CONFIG" --system "$system" --base-url http://127.0.0.1:8001
  kill -INT "$server_pid"; wait "$server_pid" || true; server_pid=""
done
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${POST_ENV}/bin/python" -m posttraining.evaluation.benchmark calibrate --config "$CONFIG" --system B0 --system B1
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${POST_ENV}/bin/python" -m posttraining.evaluation.benchmark calibrate --config "$CONFIG" --system B2 --system B3
for system in M1 M2; do
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${POST_ENV}/bin/python" -m posttraining.evaluation.benchmark calibrate --config "$CONFIG" --system "$system"
done
"${POST_ENV}/bin/python" -m posttraining.evaluation.benchmark finalize --config "$CONFIG"
