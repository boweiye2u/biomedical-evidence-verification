#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
POST_ENV="${RAG_ROOT}/envs/posttraining"
SERVE_ENV="${RAG_ROOT}/envs/serving"
RUN_ROOT="${RAG_ROOT}/runs/posttraining/lora-mixed-v1"
CHECKPOINT_ROOT="${RAG_ROOT}/checkpoints/posttraining/m2"
mkdir -p "${RAG_ROOT}/logs"
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then kill -INT "$server_pid"; wait "$server_pid" || true; fi
}
trap cleanup EXIT
for epoch in 2 3; do
  for seed in 0 1 2; do
    name="m2-mix-2-epoch-${epoch}-seed-${seed}"
    merged="${CHECKPOINT_ROOT}/dev-finalists/mix-2-epoch-${epoch}/seed-${seed}/merged"
    output="${RUN_ROOT}/dev-finalists/mix-2-epoch-${epoch}/seed-${seed}/dev-evaluation"
    log="${RAG_ROOT}/logs/${name}-vllm.log"
    [[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
      "$SERVE_ENV/bin/vllm" serve "$merged" --served-model-name "$name" --dtype bfloat16 \
      --max-model-len 16384 --gpu-memory-utilization 0.72 --max-num-seqs 16 \
      --generation-config vllm --host 127.0.0.1 --port 8001 --disable-log-requests >"$log" 2>&1 &
    server_pid=$!
    ready=0
    for _ in $(seq 1 180); do
      if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then ready=1; break; fi
      if ! kill -0 "$server_pid" 2>/dev/null; then cat "$log" >&2; exit 1; fi
      sleep 1
    done
    [[ "$ready" == 1 ]] || { echo "vLLM readiness timeout" >&2; exit 1; }
    "$POST_ENV/bin/python" -m posttraining.evaluation.baselines \
      --config "configs/posttraining/${name}-dev-v1.json" \
      --mapping "${RAG_ROOT}/runs/milestone5a-verification-v1/verification-mapping.json" \
      --rankings "${RAG_ROOT}/runs/scifact_dev_v1/bge-rankings.json" \
      --corpus "${RAG_ROOT}/data/scifact/original/corpus.jsonl" \
      --tokenizer "$merged" --output-dir "$output" --base-url http://127.0.0.1:8001
    kill -INT "$server_pid"; wait "$server_pid" || true; server_pid=""
  done
done
