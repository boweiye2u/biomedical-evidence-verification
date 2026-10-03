# Milestone 6B — local vLLM + FastAPI serving and performance

> **Scope:** This is an engineering evaluation of the frozen final system. No retriever, checkpoint, prompt, evidence depth, label mapping, decoding policy, or TEST result was changed or selected from serving measurements.

## Frozen service

The service exposes `GET /health`, `POST /retrieve`, and `POST /verify`. Production `/verify` is fixed to zero-shot BGE top-1 over all 5,183 SciFact documents, the full abstract in the unchanged grounded-v2 prompt, and Qwen2.5-7B-Instruct with greedy decoding and a 192-token production maximum. Corpus embeddings and the exact FAISS `IndexFlatIP` index load once at startup. Invalid generations are returned with parsing errors and retained in structured JSONL logs; they are never repaired into labels.

Both BGE and vLLM ran on one NVIDIA L40S. vLLM reserved 72% of GPU memory, leaving room for the BGE encoder in the FastAPI process. Requests use the exact locally rendered Qwen chat prompt through the completion endpoint, avoiding an extra server-side chat template. The checkpoint-retained repetition penalty of 1.05 is explicit; temperature is 0.

## Environment and compatibility

Python 3.11.17; PyTorch 2.7.0+cu126 / CUDA 12.6; vLLM 0.9.2; transformers 4.53.2; FastAPI 0.116.1; FAISS 1.11.0; NVIDIA L40S; NVIDIA driver 565.57.01.

The initially installed current vLLM 0.30.0 pulled a CUDA 13.0 PyTorch build, which driver 565.57.01 cannot run. The serving environment was corrected to vLLM 0.9.2 with PyTorch 2.7.0+cu126. The optional FlashInfer sampler also could not JIT-compile with the host CUDA toolkit (`nvcc` rejected `--compress-mode=size`), so `VLLM_USE_FLASHINFER_SAMPLER=0` selects vLLM's PyTorch-native sampler. This does not alter greedy decoding. `pip check` reports no broken requirements.

## Functional validation

On the first 10 numeric frozen DEV IDs, live serving matched the frozen Milestone 5A/6A behavior as follows:

| Check | Result |
|---|---:|
| Exact top-1 retrieval | 10/10 |
| Valid strict JSON/schema | 10/10 |
| Verification-label agreement | 10/10 |
| Citation validity | 10/10 |
| Byte-identical raw generation | 4/10 |

All labels and citations agree, although wording is byte-identical on only 4/10 examples. This is semantic/schema equivalence, not bitwise engine equivalence. Health, validation errors, deterministic retrieval, top-1 enforcement, malformed-output handling, frozen hashes/revisions, timing fields, and restart behavior were also checked.

## Generator-only benchmark

The generator benchmark used 20 warmup requests followed by 100 measured streaming requests per row. The frozen DEV set contains no prompt near 2,000 tokens; its longest representative prompt is 1,503 tokens, which was retained instead of fabricating padding. The benchmark maximum is 128 output tokens; production remains 192.

| Input tokens | Concurrency | Mean ms | P50 ms | P95 ms | TTFT P50/P95 ms | Mean TPOT ms | Req/s | Output tok/s | Failure | Peak GPU MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 511 | 1 | 1155.64 | 1152.20 | 1178.49 | 28.89 / 63.92 | 20.78 | 0.86 | 47.57 | 0.0% | 34894 |
| 511 | 4 | 1245.18 | 1236.95 | 1284.60 | 46.60 / 79.43 | 22.12 | 3.20 | 175.79 | 0.0% | 34894 |
| 511 | 8 | 1242.40 | 1240.96 | 1257.52 | 47.82 / 77.53 | 22.01 | 6.17 | 339.46 | 0.0% | 34894 |
| 990 | 1 | 1220.45 | 1214.10 | 1251.95 | 30.36 / 66.91 | 20.80 | 0.82 | 47.51 | 0.0% | 34894 |
| 990 | 4 | 1307.07 | 1304.61 | 1320.74 | 47.10 / 91.87 | 21.98 | 3.05 | 176.95 | 0.0% | 34894 |
| 990 | 8 | 1321.10 | 1316.11 | 1349.57 | 51.40 / 86.81 | 22.17 | 5.82 | 337.32 | 0.0% | 34894 |
| 1503 | 1 | 2186.08 | 2180.87 | 2215.39 | 32.48 / 73.42 | 20.86 | 0.46 | 47.56 | 0.0% | 34894 |
| 1503 | 4 | 2340.13 | 2335.25 | 2359.46 | 47.48 / 86.34 | 22.21 | 1.71 | 177.39 | 0.0% | 34894 |
| 1503 | 8 | 2356.25 | 2352.15 | 2395.65 | 56.41 / 100.01 | 22.28 | 3.27 | 339.40 | 0.0% | 34894 |

GPU utilization averaged about 96% in each measured configuration. Continuous batching raises generated-token throughput from roughly 47.5 tokens/s at concurrency 1 to about 337–339 tokens/s at concurrency 8. Median and P95 TTFT remain below 57 ms and 101 ms respectively at concurrency 8. Latency rises moderately with concurrency while throughput increases substantially.

## End-to-end benchmark

The end-to-end benchmark used 20 warmups and 100 measured requests per concurrency over the frozen 162-claim DEV workload.

| Concurrency | Mean/P50/P95 total ms | Embed mean ms | FAISS mean ms | Retrieval mean ms | Generation mean ms | Req/s | Failure | Invalid | Peak GPU MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1578.65 / 1501.68 / 2209.01 | 12.00 | 0.90 | 12.93 | 1552.15 | 0.63 | 0.0% | 0.0% | 34896 |
| 4 | 1686.19 / 1614.06 / 2367.98 | 19.53 | 0.87 | 20.44 | 1650.92 | 2.32 | 0.0% | 0.0% | 34896 |
| 8 | 1740.16 / 1652.38 / 2428.55 | 21.01 | 0.76 | 21.80 | 1695.25 | 4.34 | 0.0% | 0.0% | 34896 |

Generation is the clear bottleneck: it accounts for about 98% of mean service time, while exact retrieval averages 13–22 ms and FAISS itself remains below 1 ms. Concurrency 8 is the throughput-oriented operating point for this measured workload: 4.34 req/s, 1.65 s median, and 2.43 s P95, with zero failures and zero invalid outputs. Concurrency 1 is appropriate when the lowest single-request latency matters: 0.63 req/s, 1.50 s median, and 2.21 s P95.

The measured peak was 34,896 MiB of 46,068 MiB on GPU 0, leaving about 11 GiB of device capacity. System RAM use remained 1.5–1.6% on the approximately 1 TiB host. These numbers describe this model, input/output mix, software stack, and one L40S; they are not universal capacity guarantees.

## Commands

```bash
source scripts/env.sh
scripts/start_vllm_6b.sh       # terminal 1
scripts/start_api_6b.sh        # terminal 2
python -m scripts.validate_serving_6b
python -m scripts.benchmark_serving_6b --phase generator
python -m scripts.benchmark_serving_6b --phase end-to-end
python -m pytest -q tests/test_serving_6b.py
python -m pytest -q --ignore=tests/test_serving_6b.py
```

## Artifacts

- Frozen serving config: `configs/serving-6b-v1.json`
- FastAPI implementation: `serving/`
- Reproducible launch/benchmark scripts: `scripts/start_vllm_6b.sh`, `scripts/start_api_6b.sh`, `scripts/benchmark_serving_6b.py`
- Dependency freeze: `environments/serving-6b-pip-freeze.txt`
- External run: `/home/boweiye2/rag/runs/milestone6b-serving-v1`
- Structured request log: `/home/boweiye2/rag/runs/milestone6b-serving-v1/service-requests.jsonl`
- Functional equivalence: `/home/boweiye2/rag/runs/milestone6b-serving-v1/functional-equivalence.json`
- Raw summaries, per-request measurements, and telemetry: `/home/boweiye2/rag/runs/milestone6b-serving-v1/{generator-benchmark.json,end_to_end-benchmark.json}`
- Environment manifest: `/home/boweiye2/rag/runs/milestone6b-serving-v1/environment.json`
- Exact snapshot: `/home/boweiye2/rag/runs/milestone6b-serving-v1/source/exact-source-config-dependencies.tar.gz`

No TEST evaluation, retraining, reranking, passage localization, prompt change, or quality optimization was performed. Milestone 6B stops here.
