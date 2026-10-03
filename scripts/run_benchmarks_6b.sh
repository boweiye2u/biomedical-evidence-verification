#!/usr/bin/env bash
set -euo pipefail
SERVING_PY="${RAG_ROOT:-$HOME/rag}/envs/serving/bin/python"
"$SERVING_PY" -m scripts.benchmark_serving_6b --phase generator
"$SERVING_PY" -m scripts.benchmark_serving_6b --phase end-to-end
