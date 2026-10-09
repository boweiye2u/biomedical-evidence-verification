# Milestone 7 — M2 serving integration

The frozen M2 Mix-2/epoch-3/seed-0 merged checkpoint is integrated into a separate FastAPI + vLLM service. Scientific training, retrieval, prompts, parsers, and benchmark results were not changed or rerun.

## Frozen identity and architecture

```text
claim → BGE base-en-v1.5 → exact FAISS top-1 → full abstract
      → frozen M2 Qwen2.5-3B → decision + rationale_sentences
```

M2 weights are `c76fc29521b759e0b5f302116ec1c0a3d55d8b694846ee3b6e6dd0967a3cd2d2` and `66d2737fde38a876994f3f14ee27f7930fdcaf986a1839a00086d50d2298e828`. Base/tokenizer revision is `aa8e72537993ba99e69dfaafa59ed015b17504d1`. The serving config SHA-256 is `2c2ffa9ceaff8478e1e063e3b42afa784b3278507546fc523d04972b5c0df1c6`.

Retrieval remains BGE revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, query-prefix + CLS pooling, FP32 L2-normalized embeddings, the existing 5,183-document `IndexFlatIP`, and top-1 full-abstract evidence. The service rejects base-3B and M3 checkpoint paths.

Generation remains BF16, greedy temperature 0, repetition penalty 1.05, maximum 192 new tokens, model length 16,384, and seed 20261009. Invalid generations are returned as invalid and logged; no repair is applied.

## Functional regression

All 10 frozen smoke claims retrieved the identical document and produced valid v2 JSON with valid decisions and rationale indices. Exact raw output and decision matched **8/10**; rationale lists matched **9/10**. The two decision differences occurred with identical weights, prompts, documents, decoding settings, and vLLM version and are consistent with continuous-batching/request-scheduling numerical differences. No tuning followed this check.

## End-to-end concurrency curve

Twenty warmups preceded 100 measured requests at each concurrency on one L40S.

| Concurrency | P50 ms | P95 ms | Requests/s | Retrieval mean ms | Prompt mean ms | Generation mean ms | Peak GPU MiB | Failure | Invalid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 166.9 | 204.4 | 5.76 | 6.9 | 2.7 | 156.4 | 34914 | 0.0% | 0.0% |
| 2 | 189.5 | 250.1 | 9.84 | 19.8 | 2.4 | 167.3 | 34914 | 0.0% | 0.0% |
| 4 | 220.3 | 273.5 | 17.01 | 19.9 | 2.2 | 177.6 | 34914 | 0.0% | 0.0% |
| 8 | 248.0 | 290.5 | 30.68 | 13.7 | 1.4 | 184.3 | 34914 | 0.0% | 0.0% |
| 16 | 269.7 | 426.1 | 51.33 | 13.7 | 1.2 | 218.0 | 34914 | 0.0% | 0.0% |

At concurrency 8, the mean input/output lengths were 584.6/13.66 tokens, with 419.1 generated tokens/s. Internal vLLM queueing is included in generation time; client latency minus service total captures HTTP/client overhead but is not a direct queue measurement.

## Frozen quality and serving comparison

| System | Accuracy | Macro F1 | Training GPU-h | Training | P50 ms | P95 ms | Req/s | Peak GPU MiB |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| B0 7B zero-shot | 0.6867 | 0.6706 | N/A | Untuned | 1652.4 | 2428.6 | 4.34 | 34896 |
| M2 3B LoRA | 0.6733 | 0.6602 | 0.94 | Mix-2 LoRA | 248.0 | 290.5 | 30.68 | 34914 |
| M3 3B full FT | 0.7133 | 0.7018 | 4.26 | Full FT/FSDP | N/A | N/A | N/A | N/A |

At concurrency 8, M2 has **7.07×** the historical 7B request throughput, with P50 lower by **85.0%** and P95 lower by **88.0%**. The comparison combines the smaller model with the concise v2 schema: M2 emitted about 13.7 tokens/request, while v1 emitted explanations. Fixed 72% vLLM reservation made peak memory essentially identical, so no measured memory saving is claimed.

M3 serving was skipped: it is optional, needs another full server cycle, and would not change the scientific conclusion. M3's higher quality point estimate was not statistically superior to M2.

## Commands and artifacts

```bash
/home/boweiye2/rag/envs/serving/bin/python scripts/preflight_posttraining_serving.py
scripts/start_posttraining_vllm.sh
scripts/start_posttraining_api.sh
/home/boweiye2/rag/envs/serving/bin/python -m scripts.validate_posttraining_serving
/home/boweiye2/rag/envs/serving/bin/python -m scripts.benchmark_posttraining_serving
/home/boweiye2/rag/envs/serving/bin/python -m pytest -q tests/posttraining/test_serving_integration.py
```

Raw measurements and telemetry: `/home/boweiye2/rag/runs/posttraining/serving-v1/benchmark.json`. Regression: `/home/boweiye2/rag/runs/posttraining/serving-v1/functional-regression.json`. Logs: `/home/boweiye2/rag/logs/posttraining/serving-v1`. Snapshot SHA-256: `47f017c9b1a08dfc0812b7ce8a37b2ef47dc14bcd253bde23dc2d03f653531c5`.

Environment: Python 3.11.17, PyTorch 2.7.0+cu126, CUDA 12.6, transformers 4.53.2, vLLM 0.9.2, FastAPI 0.116.1, FAISS 1.11.0, driver 565.57.01, NVIDIA L40S.

M2 remains the recommended deployment model: its frozen quality is near the 7B baselines, it is substantially faster in this production workload, and its training cost is far below M3. Gold-only M1 fixed CONTRADICT behavior but collapsed abstention; mixed-evidence M2 restored INSUFFICIENT behavior while retaining strong CONTRADICT performance; M3 improved point estimates without reliable superiority.
