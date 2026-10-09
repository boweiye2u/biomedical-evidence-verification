# Biomedical Evidence Verification

[![Tests](https://github.com/boweiye2u/biomedical-evidence-verification/actions/workflows/tests.yml/badge.svg)](https://github.com/boweiye2u/biomedical-evidence-verification/actions/workflows/tests.yml)

An evidence-grounded biomedical claim verification system built with BGE, exact FAISS retrieval, Qwen2.5, LoRA/full-parameter post-training, FastAPI, and vLLM. The v2 study asks whether task-specific post-training can make a compact verifier robust to imperfect evidence while preserving abstention and structured rationale output.

The recommended deployment system is **M2**: Qwen2.5-3B-Instruct with a mixed-evidence LoRA adapter. It retains quality near the tested 7B baselines while serving substantially faster in the measured workload.

## Why this project

The first project stage showed that better cited-document ranking did not necessarily improve downstream verification. Reranking and one-sentence localization could improve an intermediate retrieval target while reducing label accuracy, and the verifier still made polarity and causal-direction errors when given annotated evidence.

The v2 study therefore holds the BGE retriever fixed and compares zero-shot, few-shot, gold-only LoRA, mixed-evidence LoRA, and full-parameter training. Its central question is how the verifier behaves with gold evidence, retrieved non-gold evidence, constructed hard negatives, and random negatives.

## System

![Biomedical retrieval and verification architecture](docs/images/architecture.svg)

```text
biomedical claim
  -> BGE query encoder
  -> exact FAISS search over 5,183 SciFact documents
  -> top-1 full abstract
  -> M2 Qwen2.5-3B-Instruct verifier
  -> SUPPORT / CONTRADICT / INSUFFICIENT + rationale sentence IDs
```

Training conditions are **A** gold evidence, **B** retrieved non-gold top-1, **C** constructed hard negatives, and **D** random negatives. Evaluation conditions are **E1** retrieved top-1 (primary), **E2** annotated evidence, **E3** constructed hard negatives, and **E4** constructed insufficient evidence.

## Experimental design

| ID | Model | Method |
|---|---|---|
| B0 | Qwen2.5-7B-Instruct | zero-shot |
| B1 | Qwen2.5-7B-Instruct | three-shot |
| B2 | Qwen2.5-3B-Instruct | zero-shot |
| B3 | Qwen2.5-3B-Instruct | three-shot |
| M1 | Qwen2.5-3B-Instruct | gold-evidence LoRA |
| M2 | Qwen2.5-3B-Instruct | mixed-evidence LoRA |
| M3 | Qwen2.5-3B-Instruct | full-parameter FSDP training |

Model selection used the frozen training-derived TRAIN/DEV split. Final comparisons use **the fixed SciFact benchmark split, previously evaluated in v1 and not used for v2 training or model selection.** It is a reused benchmark and does not provide a pristine unseen estimate of generalization.

## Main findings

- **M1** corrected the 3B baselines' severe CONTRADICT underprediction, but gold-only training collapsed INSUFFICIENT behavior.
- **M2** restored INSUFFICIENT behavior while retaining strong CONTRADICT performance. Its fixed-benchmark quality remained near the 7B baselines.
- **M3** produced the highest point estimates, but required substantially more compute and multi-GPU FSDP. Its advantage over M2 was not statistically reliable.
- Mixed evidence was more important than simply adding supervised gold-evidence examples: M2 clearly outperformed M1 under the frozen evaluation.

## Results

All accuracy and macro-F1 values below use primary condition E1 on the fixed 300-claim benchmark.

| System | Accuracy | Macro F1 | Training strategy | Recorded experimental training compute |
|---|---:|---:|---|---:|
| B0 | 0.6867 | 0.6706 | 7B zero-shot | N/A |
| B1 | 0.7033 | 0.6728 | 7B three-shot | N/A |
| B2 | 0.6167 | 0.4753 | 3B zero-shot | N/A |
| B3 | 0.6267 | 0.4881 | 3B three-shot | N/A |
| M1 | 0.4567 | 0.3794 | 3B gold-only LoRA | 0.4582 GPU-hours |
| **M2** | **0.6733** | **0.6602** | **3B mixed-evidence LoRA** | **0.9376 GPU-hours** |
| M3 | 0.7133 | 0.7018 | 3B full FT with FSDP | 4.26 GPU-hours |

The reported compute totals include each milestone's cross-validation and final training runs; model evaluation and server startup are excluded. M1 and M2 used one GPU. M3 reports aggregate GPU-hours across four-GPU FSDP jobs, so the totals describe the executed experimental programs rather than the cost of one comparable training run.

## Statistical evaluation

- Accuracy comparisons used exact paired McNemar tests.
- The five pre-registered main comparisons used Holm correction.
- Macro F1 used paired claim-bootstrap confidence intervals.
- M3 versus M2 was the separately authorized secondary test 6, outside the original Holm family.
- For M3 minus M2, the accuracy difference was +0.0400 with McNemar `p = 0.1686`; the macro-F1 difference was +0.0416 with 95% CI `[-0.0119, 0.0958]`.

These results do not support a formal equivalence claim, and they do not establish statistically reliable superiority of M3 over M2.

## Serving

![Local serving architecture](docs/images/serving.svg)

The M2 service uses FastAPI for request validation and retrieval, exact `IndexFlatIP` search over cached corpus embeddings, and vLLM for structured Qwen generation. At concurrency 8 on one NVIDIA L40S:

| System | P50 | P95 | Throughput |
|---|---:|---:|---:|
| Historical B0/v1 7B service | 1,652 ms | 2,429 ms | 4.34 requests/s |
| **M2 v2 3B service** | **248 ms** | **291 ms** | **30.68 requests/s** |

M2 delivered **7.07x** higher request throughput, **85.0%** lower P50, and **88.0%** lower P95 in this workload. The improvement combines the smaller model with v2's shorter structured output; it cannot be attributed entirely to parameter count. M2 emitted about 13.7 output tokens per request, while the historical v1 schema included explanations.

Start the frozen M2 service from the repository root after placing its external checkpoint and retrieval artifacts at the paths declared in [`serving-tuned-3b-v1.json`](configs/posttraining/serving-tuned-3b-v1.json):

```bash
export RAG_ROOT="${RAG_ROOT:-$HOME/rag}"
scripts/start_posttraining_vllm.sh  # terminal 1
scripts/start_posttraining_api.sh   # terminal 2
```

## Reproducibility

- Model and tokenizer revisions are pinned in versioned configs.
- Prompt text, output schemas, split hashes, training mixtures, seeds, and checkpoint-selection rules are recorded.
- Exact Conda and pip snapshots are under `environments/`.
- Machine-readable result summaries accompany the milestone reports.
- Source/config/dependency snapshots and large run artifacts are preserved externally under `$RAG_ROOT`.
- Datasets, model weights, adapters, checkpoints, embeddings, indexes, raw generations, and logs are intentionally excluded from Git.

The portable setup starts with the retrieval environment and adds the post-training packages:

```bash
export RAG_ROOT="${RAG_ROOT:-$HOME/rag}"
source scripts/env.sh
conda env create --prefix "$RAG_ROOT/envs/retrieval" -f environments/retrieval.yml
conda create --clone "$RAG_ROOT/envs/retrieval" --prefix "$RAG_ROOT/envs/posttraining"
"$RAG_ROOT/envs/posttraining/bin/python" -m pip install -r environments/posttraining-requirements.txt
```

See [`environments/README.md`](environments/README.md), the exact freezes, and each milestone report for the recorded environment and commands. Replaying model training or evaluation requires the external datasets and pinned model snapshots; ordinary tests do not.

## Repository layout

```text
configs/posttraining/  Frozen v2 protocols, training, evaluation, and serving configs
docs/posttraining/     Human-readable reports and small machine-readable results
posttraining/          Data, training, evaluation, calibration, scaling, and utilities
retrieval/             Retrieval, metrics, evidence mapping, and v1 evaluation code
serving/               Historical v1 and frozen M2 FastAPI integrations
scripts/               Reproducible milestone and service entry points
tests/posttraining/    v2 protocol, data, training, evaluation, and serving tests
environments/          Portable requirements and exact dependency snapshots
```

## Running tests

The project keeps serving dependencies separate from retrieval/post-training dependencies:

```bash
export RAG_ROOT="${RAG_ROOT:-$HOME/rag}"

"$RAG_ROOT/envs/posttraining/bin/python" -m pytest -q \
  --ignore=tests/test_serving_6b.py \
  --ignore=tests/posttraining/test_serving_integration.py

"$RAG_ROOT/envs/serving/bin/python" -m pytest -q \
  tests/test_serving_6b.py \
  tests/posttraining/test_serving_integration.py
```

Hosted CI runs a model-free CPU subset. It does not download models, start vLLM, train models, or regenerate benchmark results.

## Reports

- [Protocol verification](docs/posttraining/milestone0-protocol-verification.md)
- [Matched DEV baselines](docs/posttraining/milestone1-baselines-dev.md)
- [SFT data audit](docs/posttraining/milestone2-sft-data-audit.md)
- [M1 gold-evidence LoRA](docs/posttraining/milestone3-m1-lora-gold.md)
- [M2 mixed-evidence LoRA](docs/posttraining/milestone4-m2-lora-mixed.md)
- [Main fixed-benchmark evaluation](docs/posttraining/milestone6-main-benchmark.md)
- [M3 full fine-tuning](docs/posttraining/milestone5-m3-full-ft.md)
- [FSDP scaling](docs/posttraining/milestone5-fsdp-scaling.md)
- [M2 serving integration](docs/posttraining/milestone7-serving-integration.md)

Historical retrieval, evidence-selection, verification, and serving reports from v1 remain under [`docs/`](docs/). Their original terminology and recorded paths are preserved for reproducibility; the benchmark disclosure above governs the current v2 interpretation.

## Docker status

The checked-in Dockerfile packages the historical v1 7B serving path. Its image build passes in GitHub Actions from a clean checkout; GPU container serving has not been runtime-validated. The M2 v2 service was validated with the local environment and launch scripts above, and its model/data artifacts remain external.

## Citation

If this repository supports your work, cite the repository URL and the specific version or commit used. No DOI has been assigned.

## Acknowledgments

This project uses [SciFact](https://github.com/allenai/scifact), [BEIR](https://github.com/beir-cellar/beir), [BGE](https://huggingface.co/BAAI/bge-base-en-v1.5), [Qwen2.5](https://huggingface.co/Qwen), [FAISS](https://github.com/facebookresearch/faiss), [PEFT](https://github.com/huggingface/peft), [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html), [vLLM](https://github.com/vllm-project/vllm), and [FastAPI](https://github.com/fastapi/fastapi). Dataset and model artifacts retain their own licenses and terms.

## License

Project-authored code and documentation are released under the [MIT License](LICENSE). Third-party datasets, models, and dependencies are governed by their respective licenses.
