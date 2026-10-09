#!/usr/bin/env bash
set -u -o pipefail
source "$(dirname "$0")/env.sh"
ENV="${RAG_ROOT}/envs/posttraining"
ROOT="${RAG_ROOT}/runs/posttraining/fsdp-scaling-v1"
CONFIG="configs/posttraining/train-m3-full-ft-v1.json"
[[ ! -e "$ROOT" ]] || { echo "Refusing existing scaling root: $ROOT" >&2; exit 1; }
mkdir -p "$ROOT" "${RAG_ROOT}/logs"
for n in 1 2 4; do
  case "$n" in 1) visible=0; accum=8;; 2) visible=0,1; accum=4;; 4) visible=0,1,2,3; accum=2;; esac
  output="$ROOT/gpu-${n}"
  log="${RAG_ROOT}/logs/m3-fsdp-scaling-${n}gpu.log"
  CUDA_VISIBLE_DEVICES="$visible" CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$ENV/bin/torchrun" --standalone --nproc_per_node="$n" -m posttraining.training.m3_full train \
      --mode scaling --config "$CONFIG" --output "$output" --seed 20261023 --epochs 3 \
      --max-steps 12 --measure-after-step 2 --expected-accumulation "$accum" >"$log" 2>&1
  code=$?
  if [[ "$code" -ne 0 ]]; then
    "$ENV/bin/python" -c "import json;from pathlib import Path;p=Path('$output');p.mkdir(parents=True,exist_ok=True);(p/'failure.json').write_text(json.dumps({'status':'failed','gpu_count':$n,'exit_code':$code,'log':'$log','preserved':True},indent=2)+'\\n')"
  fi
done
