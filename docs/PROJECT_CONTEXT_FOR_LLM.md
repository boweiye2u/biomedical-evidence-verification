# Biomedical Evidence Retrieval & Verification — Project Context for LLMs

> **Historical scope:** This document records the completed v1 retrieval and verification study. The current v2 post-training study is summarized in the repository README and `docs/posttraining/`. Its 300-claim evaluation set is the fixed SciFact benchmark split, previously evaluated in v1 and not used for v2 training or model selection.

This document gives a new language model or collaborator enough context to understand the completed project without reconstructing its history from every report. It records the objective, filesystem layout, frozen experimental choices, main results, engineering work, and limits. Detailed reports and machine-readable artifacts remain the source of truth.

## Project status

The project is complete through final held-out SciFact evaluation and local serving benchmarks. It includes reproducible data preparation, retrieval baselines, controlled retriever training, statistical analysis, evidence-grounded claim verification, reranking and passage diagnostics, final TEST evaluation, and a FastAPI/vLLM service.

The final system is:

```text
Biomedical claim
  -> zero-shot BAAI/bge-base-en-v1.5 query encoder
  -> exact FAISS search over 5,183 SciFact documents
  -> top-1 full abstract
  -> Qwen/Qwen2.5-7B-Instruct with grounded-v2 prompt
  -> SUPPORT / CONTRADICT / INSUFFICIENT
     plus supplied evidence IDs and a short explanation
```

## Filesystem layout

Source code and small reproducibility artifacts live in Dropbox so they can be versioned. Datasets, model weights, caches, checkpoints, run outputs, and logs live outside Dropbox.

| Purpose | Path |
|---|---|
| Git repository and source code | `/home/boweiye2/Dropbox/Non-coding-variant/scientific-rag` |
| Heavy-artifact root (`RAG_ROOT`) | `/home/boweiye2/rag` |
| SciFact and derived data | `/home/boweiye2/rag/data/scifact` |
| Downloaded model weights | `/home/boweiye2/rag/models` |
| Hugging Face cache | `/home/boweiye2/rag/cache/huggingface` |
| Cached embeddings and indexes | `/home/boweiye2/rag/embeddings` |
| Trained checkpoints | `/home/boweiye2/rag/checkpoints` |
| Versioned experiment outputs | `/home/boweiye2/rag/runs` |
| Logs | `/home/boweiye2/rag/logs` |
| Retrieval Conda environment | `/home/boweiye2/rag/envs/retrieval` |
| Serving Conda environment | `/home/boweiye2/rag/envs/serving` |

The main repository directories are:

| Directory | Contents |
|---|---|
| `configs/` | Frozen split, training, verification, reranking, final-test, and serving configurations |
| `retrieval/` | Retrieval and metric implementation |
| `serving/` | FastAPI service models, startup logic, and endpoints |
| `scripts/` | Data preparation, training, evaluation, analysis, and benchmark entry points |
| `tests/` | CPU-safe unit tests plus milestone-specific correctness tests |
| `docs/` | Human-readable reports and public project documentation |
| `environments/` | Reproducible dependency specifications and freezes |

Do not place large data, model weights, generated rankings, checkpoints, or logs in the Git repository. `scripts/env.sh` defines the external storage convention.

## Data and task definitions

The dataset is SciFact, represented through BEIR for document retrieval and the original SciFact annotations for verification labels and rationale evidence.

- Corpus: 5,183 scientific documents.
- Training-derived split: 647 TRAIN claims and 162 frozen DEV claims.
- Official BEIR TEST: 300 queries with 339 known-positive document assignments.
- Retrieval target: BEIR cited-document relevance.
- Verification target: original SciFact `SUPPORT`, `CONTRADICT`, or `INSUFFICIENT` label.
- These targets are related but not identical. A cited document is not automatically an annotated verification-evidence document.
- Retrieval always searches the full 5,183-document corpus.
- The official TEST split remained untouched until the final frozen evaluation.

The mapping work verified claim IDs, query text, cited documents, annotated evidence documents, rationale sentence indices, and label semantics. Empty evidence maps to the benchmark's `INSUFFICIENT` class, but it does not prove that no relevant evidence exists in the literature.

## Frozen models and representations

### Retrieval

- Model: `BAAI/bge-base-en-v1.5`
- Revision: `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`
- Query prefix: `Represent this sentence for searching relevant passages: `
- Document format: title followed by abstract
- Pooling: CLS
- Embeddings: FP32 and L2-normalized
- Search: exact inner-product search using FAISS `IndexFlatIP`

BM25 provides the lexical baseline. MedCPT dual encoders and `ncbi/MedCPT-Cross-Encoder` were evaluated as biomedical retrieval/reranking diagnostics.

### Verification

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Precision: BF16 on one NVIDIA L40S
- Prompt: `grounded-v2`
- Prompt SHA-256: `21c9360a72834e37fcc37936190e290638a924e8985c50bf6dcff99472ba50b4`
- Decoding: greedy, with at most 192 new tokens
- Output: strict JSON containing the decision, supplied evidence IDs, and explanation
- Grounding rule: the model may cite only evidence IDs present in the request and should abstain when evidence is inadequate

## Work completed

### Environment and data preparation

The project established separate retrieval and serving environments to prevent PyTorch/CUDA/vLLM conflicts. SciFact was downloaded outside Git, mapped to original annotations, checked manually and programmatically, and split deterministically. Metric implementations were validated against trusted IR implementations and hand-checkable cases.

### Initial retrieval baselines on DEV

| System | NDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.6982 | 0.8276 | 0.9144 | 0.6718 |
| BGE zero-shot | 0.7604 | 0.8938 | 0.9815 | 0.7279 |

BGE zero-shot was the stronger baseline and became the frozen production retriever.

### Controlled BGE training

The controlled comparison used the same BGE checkpoint, positive pairs, five explicit negatives, loss, batch size, optimizer, learning rate, scheduler, temperature, steps, and checkpoint-selection rule. The only experimental difference was the negative policy:

- Random negatives sampled from eligible corpus documents.
- BGE-mined negatives selected from the frozen zero-shot BGE top 50.

Each arm ran for three epochs using seeds `20261002`, `20261003`, and `20261004`. Random-negative adaptation approximately preserved zero-shot performance. The specific zero-shot-BGE-mined policy degraded DEV retrieval consistently as training continued. Hard-arm training loss decreased while DEV retrieval worsened and pre-clipping gradient norms remained much larger.

The predefined cluster-aware primary contrast averaged selected seed checkpoints within each arm:

- Mined minus random NDCG@10: **-0.0346**
- Cluster bootstrap 95% CI: **[-0.0739, 0.0055]**
- Cluster sign-flip p-value: **0.0903**
- Per-query outcomes: 20 improved, 101 unchanged, 41 degraded

This result concerns one fixed mining policy and one frozen training configuration. It does not show that hard negatives generally help or hurt, and it does not identify false-negative noise as the cause.

### Evidence-grounded verification on DEV

Prompt and evidence-depth development selected `grounded-v2` with the top-1 full abstract. More documents increased annotated-evidence coverage but reduced label metrics and output validity for this model/prompt setup.

| Condition | Accuracy | Macro F1 |
|---|---:|---:|
| Zero-shot BGE top-1 full abstract | 0.7654 | 0.7531 |
| Selected random-trained BGE top-1 full abstract | 0.7654 | 0.7531 |
| Annotated full document diagnostic | 0.8765 | 0.8636 |

The tied aggregate results for the two retrievers do not establish retrieval equivalence; their contexts and individual errors differed. The annotated-document diagnostic shows substantial headroom while also demonstrating that verification reasoning remains imperfect even with annotated evidence.

### Reranking diagnostic

MedCPT cross-encoder reranking of BGE's top 10 improved the BEIR cited-document ranking target:

- NDCG@10: 0.7604 -> 0.7872
- MRR@10: 0.7279 -> 0.7593

It slightly reduced annotated-evidence top-1 coverage and reduced downstream verification:

- Accuracy: 0.7654 -> 0.7407
- Macro F1: 0.7531 -> 0.7309

The paired uncertainty intervals crossed zero. This is a negative DEV result for this reranker, candidate depth, truncation rule, and frozen verifier. It also shows why optimizing cited-document retrieval alone need not optimize evidence-grounded verification.

### Passage-localization diagnostic

The passage experiment scored original SciFact sentences from BGE's top three documents with MedCPT and supplied only the highest-scoring sentence.

- Exact annotated rationale at rank 1: 42/98 claims, or 0.4286.
- Exact annotated rationale within rank 3: 75/98, or 0.7653.
- Passage condition: 0.6852 accuracy and 0.6647 macro F1.
- Full-abstract baseline: 0.7654 accuracy and 0.7531 macro F1.
- Gold rationale: 0.8457 accuracy and 0.8213 macro F1.
- Gold full abstract: 0.8765 accuracy and 0.8636 macro F1.

One selected sentence often removed relations or surrounding context required for verification. Even an exact annotated rationale could be insufficient in isolation. This does not establish that passage retrieval is generally harmful.

### Final held-out TEST evaluation

The production pipeline was frozen before the first official TEST evaluation. An implementation correction changed the original-annotation path from `claims_test.jsonl` to the release-aligned `claims_dev.jsonl` before any TEST ranking, generation, or metric result existed. No model, prompt, evidence depth, label rule, or metric changed.

Retrieval results:

| Metric | TEST result |
|---|---:|
| NDCG@10 | 0.7404 |
| NDCG@10 95% CI | [0.6998, 0.7795] |
| Recall@10 | 0.8742 |
| Recall@100 | 0.9667 |
| MRR@10 | 0.7034 |
| Top-1 BEIR known-positive rate | 0.6167 |

Verification results:

| Condition | Accuracy | Macro F1 | Invalid rate | Citation validity |
|---|---:|---:|---:|---:|
| Frozen final system | 0.7000 | 0.6859 | 0.0033 | 0.9967 |
| Annotated full-document diagnostic | 0.8300 | 0.8003 | 0.0033 | 0.9967 |

- Accuracy 95% CI: **[0.6467, 0.7500]**
- Macro-F1 95% CI: **[0.6298, 0.7391]**
- Class F1: SUPPORT 0.7480, CONTRADICT 0.6080, INSUFFICIENT 0.7018
- Annotated evidence top-1 coverage among 188 evidence-bearing claims: 149/188, or 0.7926
- Annotated evidence top-3 coverage: 171/188, or 0.9096

The final system made 90 errors: 32 empty-annotation/INSUFFICIENT ambiguity cases, 22 cases with annotated evidence absent at rank 1, 35 cases where annotated evidence was selected but Qwen was wrong, and one invalid output. These conservative categories are descriptive and do not assign causal shares.

### Serving and performance

The local service exposes:

- `GET /health`
- `POST /retrieve`
- `POST /verify`

Corpus embeddings and exact FAISS index load once at startup. FastAPI handles retrieval and validation; vLLM serves Qwen. The validated serving stack uses Python 3.11.17, PyTorch 2.7.0+cu126, vLLM 0.9.2, transformers 4.53.2, FastAPI 0.116.1, and FAISS 1.11.0 on one L40S.

At concurrency 8 on the measured 162-claim DEV workload:

- Throughput: 4.34 requests/second
- Median end-to-end latency: 1.65 seconds
- P95 end-to-end latency: 2.43 seconds
- Retrieval: approximately 13-22 ms
- FAISS search: below 1 ms
- Peak GPU memory: approximately 34.9 GiB
- Failures and invalid outputs: 0%

Generation accounts for about 98% of service time. Generator throughput rose from about 47.5 output tokens/second at concurrency 1 to about 337-339 tokens/second at concurrency 8 through continuous batching.

The Docker image build passes in GitHub Actions from a clean checkout. GPU container serving has not been runtime-validated.

## Main findings

1. Zero-shot BGE is a strong SciFact retriever and generalized to TEST with a modest descriptive decline.
2. Random-negative adaptation did not produce a reliable improvement over zero-shot BGE; it approximately preserved performance.
3. This particular zero-shot-BGE hard-negative mining policy degraded DEV retrieval across seeds, but the cluster-aware primary confidence interval crossed zero.
4. Cited-document ranking and verification-evidence selection are different objectives. MedCPT reranking improved the former while worsening the latter and downstream verification.
5. One-sentence evidence localization was too aggressive. It frequently removed context required for relation, polarity, and causal-direction reasoning.
6. Evidence selection is a major bottleneck, but it is not the only bottleneck. Qwen still made polarity and relation errors when given annotated evidence.
7. The final pipeline is operational and reproducible, and generation is the dominant serving cost.

## Experimental safeguards and interpretation

- TRAIN, DEV, and TEST roles were kept separate.
- All model and prompt selection occurred on DEV.
- TEST was evaluated once after the final system was frozen.
- Retrieval and verification relevance targets are reported separately.
- Related DEV claims were grouped for cluster-aware uncertainty analysis.
- Checkpoint, prompt, configuration, data, and artifact hashes were recorded.
- Invalid generations remained invalid; they were not repaired into favorable labels.
- Diagnostic gold-evidence conditions are not strict upper bounds.
- Post-selection DEV confidence intervals characterize the selected systems on DEV; they are not unseen-test generalization estimates.

## Important reports and artifacts

Repository reports:

- `docs/scifact-mapping-report.md`
- `docs/scifact-dev-baselines.md`
- `docs/milestone4b-replication-review.md`
- `docs/milestone4c-statistical-failure-analysis.md`
- `docs/milestone5a-verification-dev-report.md`
- `docs/milestone5b-reranking-dev-report.md`
- `docs/milestone5c-passage-localization-dev-report.md`
- `docs/milestone6a-final-test-report.md`
- `docs/milestone6b-serving-report.md`
- `docs/final-project-summary.md`

Key external run directories:

- Final TEST: `/home/boweiye2/rag/runs/milestone6a-test-v1`
- Serving: `/home/boweiye2/rag/runs/milestone6b-serving-v1`
- Earlier versioned runs: `/home/boweiye2/rag/runs/`

Each major run retains configuration, raw outputs, per-query records, metrics, and an exact source/config/dependency snapshot where applicable.

## Git and public repository

- Local repository: `/home/boweiye2/Dropbox/Non-coding-variant/scientific-rag`
- Public remote: `https://github.com/boweiye2u/biomedical-evidence-verification.git`
- Primary branch: `main`

This context file is documentation only. It does not change any experiment, result, model, configuration, or service behavior.
