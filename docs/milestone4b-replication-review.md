# Milestone 4B three-seed replication — 2026-10-02

The frozen comparison is complete for seeds 20261002, 20261003, and 20261004.
Seed 20261002 comes from the preserved pilot; the other two seeds come from the
versioned replication run. No TEST qrels, new ablations, reranking, LLM, RAG,
vLLM, or serving work was used. Work stopped after this report.

## Every DEV epoch

All runs evaluate the same 162 frozen DEV claims against the complete 5,183-document
SciFact corpus. Metrics use BEIR cited-document relevance.

| Seed | Arm | Epoch | NDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---:|---|---:|---:|---:|---:|---:|
| 20261002 | Random | 1 | 0.7526 | 0.8710 | 0.9877 | 0.7223 |
| 20261002 | Random | 2 | 0.7584 | 0.8648 | 0.9877 | 0.7319 |
| 20261002 | **Random** | **3** | **0.7622** | **0.8710** | **0.9877** | **0.7353** |
| 20261002 | **BGE-mined** | **1** | **0.7251** | **0.8360** | **0.9537** | **0.7023** |
| 20261002 | BGE-mined | 2 | 0.6944 | 0.7813 | 0.9259 | 0.6815 |
| 20261002 | BGE-mined | 3 | 0.6718 | 0.7638 | 0.8920 | 0.6577 |
| 20261003 | Random | 1 | 0.7582 | 0.8679 | 0.9877 | 0.7289 |
| 20261003 | Random | 2 | 0.7622 | 0.8741 | 0.9877 | 0.7343 |
| 20261003 | **Random** | **3** | **0.7636** | **0.8741** | **0.9877** | **0.7364** |
| 20261003 | **BGE-mined** | **1** | **0.7205** | **0.8278** | **0.9352** | **0.7010** |
| 20261003 | BGE-mined | 2 | 0.7080 | 0.8031 | 0.9167 | 0.6904 |
| 20261003 | BGE-mined | 3 | 0.6887 | 0.7918 | 0.9043 | 0.6690 |
| 20261004 | **Random** | **1** | **0.7595** | **0.8691** | **0.9877** | **0.7306** |
| 20261004 | Random | 2 | 0.7538 | 0.8617 | 0.9877 | 0.7267 |
| 20261004 | Random | 3 | 0.7515 | 0.8710 | 0.9877 | 0.7208 |
| 20261004 | **BGE-mined** | **1** | **0.7359** | **0.8360** | **0.9321** | **0.7155** |
| 20261004 | BGE-mined | 2 | 0.6943 | 0.7936 | 0.9198 | 0.6785 |
| 20261004 | BGE-mined | 3 | 0.6739 | 0.7700 | 0.8889 | 0.6553 |

Bold rows are selected by the frozen rule: maximum epoch-end DEV NDCG@10,
earliest epoch on a tie.

## Selected checkpoints

| Seed | Arm | Epoch | Checkpoint under `~/rag/` |
|---:|---|---:|---|
| 20261002 | Random | 3 | `checkpoints/milestone4b-pilot-v1/random/epoch-3` |
| 20261002 | BGE-mined | 1 | `checkpoints/milestone4b-pilot-v1/hard/epoch-1` |
| 20261003 | Random | 3 | `checkpoints/milestone4b-replication-v1/seed-20261003/random/epoch-3` |
| 20261003 | BGE-mined | 1 | `checkpoints/milestone4b-replication-v1/seed-20261003/hard/epoch-1` |
| 20261004 | Random | 1 | `checkpoints/milestone4b-replication-v1/seed-20261004/random/epoch-1` |
| 20261004 | BGE-mined | 1 | `checkpoints/milestone4b-replication-v1/seed-20261004/hard/epoch-1` |

## Selected-checkpoint metrics

| Arm / seed | NDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---:|---:|---:|
| Zero-shot fixed baseline | 0.7604 | 0.8938 | 0.9815 | 0.7279 |
| Random 20261002 | 0.7622 | 0.8710 | 0.9877 | 0.7353 |
| Random 20261003 | 0.7636 | 0.8741 | 0.9877 | 0.7364 |
| Random 20261004 | 0.7595 | 0.8691 | 0.9877 | 0.7306 |
| BGE-mined 20261002 | 0.7251 | 0.8360 | 0.9537 | 0.7023 |
| BGE-mined 20261003 | 0.7205 | 0.8278 | 0.9352 | 0.7010 |
| BGE-mined 20261004 | 0.7359 | 0.8360 | 0.9321 | 0.7155 |

## Three-seed mean ± sample standard deviation

The standard deviation uses `n - 1` for the three predefined seeds.

| Arm | NDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---:|---:|---:|
| Random | 0.7617 ± 0.0021 | 0.8714 ± 0.0025 | 0.9877 ± 0.0000 | 0.7341 ± 0.0031 |
| BGE-mined | 0.7271 ± 0.0079 | 0.8333 ± 0.0048 | 0.9403 ± 0.0117 | 0.7063 ± 0.0080 |

Relative to zero-shot, random sampling approximately preserves NDCG@10 and modestly
raises MRR@10 and Recall@100, while lowering Recall@10. Its mean NDCG@10 difference
is +0.0013, too small to treat as an established improvement from these DEV runs.
One random seed selected epoch 1 and finished slightly below zero-shot NDCG@10.

The original BGE-mined pattern replicated. Every seed selected epoch 1, every
selected BGE-mined checkpoint was below zero-shot on all four metrics, and later
epochs generally degraded further. The mean selected NDCG@10 difference from
zero-shot is -0.0333. This supports the narrow conclusion that the frozen
zero-shot-BGE mining policy underperformed random sampling under this exact setup.
It does not show that hard negatives in general are harmful, and it does not
identify false-negative noise as the cause.

## Training behavior

Across seeds, mean random-arm training loss fell from 0.08360 at epoch 1 to
0.00754 and 0.00344 at epochs 2 and 3. Mean pre-clipping gradient norm fell from
2.302 to 0.375 and 0.163. The random examples became easy rapidly.

Mean BGE-mined loss fell from 1.14563 to 0.57835 and 0.34086. Mean pre-clipping
gradient norm remained high at 9.258, 9.369, and 8.644 against a fixed clipping
threshold of 1.0. DEV retrieval declined as optimization continued in all three
seeds. Overfitting, optimization sensitivity, and unjudged relevance remain
possible explanations; no causal attribution is justified here.

## Checks and reproducibility

- The seed-20261002 pilot inventory hash is unchanged before and after replication:
  `7b0f7e99789e683cd5741eb3feec7ea97ba0c101be1011dbf7a27a513ac64d2d`.
- Frozen pilot config, original 4A config, split, and both training-data hashes match.
- No extracted TEST qrels exist; the runner asserts and records that only
  `data/scifact/beir/qrels/train.tsv` was loaded.
- All four new arms started from model-state hash
  `24ba89273c20027dac89c0d0a395e88c94d89e7ebb8a03c3144caafb35b7a92a`.
- Arms within each seed used identical epoch row-order hashes.
- All 12 new and six pilot epochs have steps 47/94/141, 47 trace records per
  epoch, matching saved rankings and per-query metrics.
- All 18 epochs were independently checked against pytrec_eval and BEIR; every
  checkpoint weight hash and frozen checkpoint selection was verified.
- The existing eight-test suite passed before execution. The replication runner
  compiled successfully.
- Exact source/config/dependency archive:
  `~/rag/runs/milestone4b-replication-v1/source/exact-source-config-dependencies.tar.gz`.
  SHA-256: `cb3aaedafc3d35836f9914892461737eb5e802abdfa46d01f84e2a6a641cf798`.

## Commands

Preflight:

```bash
source scripts/env.sh
python -m py_compile scripts/train_replication_4b.py
python -m pytest -q -p no:cacheprovider tests/
```

Replication:

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 \
  "$RAG_ROOT/envs/retrieval/bin/python" -m scripts.train_replication_4b \
  > "$RAG_ROOT/logs/milestone4b-replication.log" 2>&1
```

The runner uses `exist_ok=False` for the versioned run and checkpoint targets, so
it cannot silently replay into or overwrite those outputs. A user status message
arrived while the process was running; the existing process continued and no epoch
was restarted. There were no execution failures or environment changes.

## Artifacts

- New checkpoints: `~/rag/checkpoints/milestone4b-replication-v1/`.
- New raw results: `~/rag/runs/milestone4b-replication-v1/`.
- Log: `~/rag/logs/milestone4b-replication.log`.
- Concise machine-readable summary:
  [milestone4b-replication-results.json](milestone4b-replication-results.json).
- Exact new entry point: `scripts/train_replication_4b.py`.

Replication is complete. No new experiment follows from this report.
