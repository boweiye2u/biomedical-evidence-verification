# Milestone 4B pilot review — 2026-10-02

Completed exactly two training jobs: random and BGE-mined negatives, seed
20261002, three epochs / 141 scheduled optimizer steps each. **Stopped before
replication, test evaluation, LLM, or serving work.**

## DEV results

162 frozen DEV claims; search all 5,183 documents. Metrics measure the unchanged
BEIR cited-document retrieval target, not verification evidence.

| Arm | Epoch | NDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Zero-shot | — | 0.7604 | 0.8938 | 0.9815 | 0.7279 |
| Random | 1 | 0.7526 | 0.8710 | 0.9877 | 0.7223 |
| Random | 2 | 0.7584 | 0.8648 | 0.9877 | 0.7319 |
| Random | 3 | 0.7622 | 0.8710 | 0.9877 | 0.7353 |
| BGE-mined | 1 | 0.7251 | 0.8360 | 0.9537 | 0.7023 |
| BGE-mined | 2 | 0.6944 | 0.7813 | 0.9259 | 0.6815 |
| BGE-mined | 3 | 0.6718 | 0.7638 | 0.8920 | 0.6577 |

Frozen selection rule: highest DEV NDCG@10 among epoch-end checkpoints, earliest
epoch on a tie. Selected **random epoch 3** and **BGE-mined epoch 1**. Zero-shot is
retained as a separate baseline, not an eligible trained epoch.

Random's selected NDCG improvement over zero-shot is only **0.00173** absolute;
Recall@10 is lower. This one-seed DEV result does not establish a reliable gain.
Hard-negative training underperforms here, with DEV quality declining across all
three epochs despite lower training loss. This is evidence about this fixed mining
policy/training configuration, not that hard negatives are universally harmful.

## Learning behavior

| Arm | Epoch | Mean training loss | Mean pre-clip gradient norm |
|---|---:|---:|---:|
| random | 1 | 0.085059 | 2.3085 |
| random | 2 | 0.007138 | 0.3436 |
| random | 3 | 0.003574 | 0.1740 |
| hard | 1 | 1.167017 | 9.0024 |
| hard | 2 | 0.587623 | 8.7515 |
| hard | 3 | 0.324050 | 8.5581 |

Losses and gradient norms stayed finite. Random negatives became easy quickly.
The hard arm retained larger gradients (clipping threshold 1.0 in both arms) and
showed divergence between training loss and DEV quality. Overfitting, optimization
sensitivity, and unjudged relevant negatives are plausible explanations; this
experiment does not isolate their contributions. Do not attribute the drop to label
noise without further controlled evidence.

Training losses are averages of the batch loss weighted by number of rows; optimizer
updates average each batch, including the final one-row batch. No in-batch negatives,
negative resampling, early stopping, or parameter search was introduced.

## Correctness and frozen settings

- Original 4A config and datasets preserved; pilot config saved separately as
  `configs/training-4b-pilot-v1.json`.
- Both initial full model-state hashes match. Optimizer/scheduler and random states
  reset separately. Paired epoch row-order hashes match across arms.
- Standard cross-entropy over six cosine scores divided by 0.05, target column 0.
  Shared encoder receives gradients through queries and documents.
- Unit smoke check verified the loss formula, both gradient branches, finite
  gradients, and a real optimizer update using a tiny randomly initialized BERT;
  this was a correctness test, not an extra BGE training job.
- Actual jobs checked both gradient branches on their first batch and parameter
  changes after the second step. The first warmup step has LR 0, as defined by the
  scheduler; each job still executes the same 141 optimizer/scheduler steps.
- FP32, batch 16, AdamW 2e-5, 512-token limits, 15 warmup steps, linear decay,
  maximum gradient norm 1.0. Non-reentrant gradient checkpointing with RNG
  preservation enabled for both arms; no change to effective batch size or loss.
- Deterministic PyTorch algorithms enabled, TF32 disabled, one L40S, four CPU
  threads, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Bitwise reproducibility on different
  hardware/software is not promised.
- Each epoch freshly encoded the complete corpus with its current checkpoint.
  DEV query and document encoding used batches of 32, exact CPU FAISS inner-product
  search over normalized CLS embeddings. Eval mode with gradients disabled.
- All six epoch metric outputs independently verified against pytrec_eval and BEIR;
  all checkpoint weight-file hashes verified. Eight tests passed before training.
- No test qrels read. No downstream prompts, generations, or serving created.

## Runtime and artifacts

Each epoch took about 69–71 seconds of training and 21–22 seconds of DEV evaluation,
excluding checkpoint writes. Peak PyTorch allocation was approximately 6,402 MiB
(not total driver-level GPU memory).

External artifacts:
- Checkpoints: `~/rag/checkpoints/milestone4b-pilot-v1/{random,hard}/epoch-{1,2,3}/`.
  Each includes model/tokenizer and optimizer/scheduler state.
- Per-epoch rankings, per-query metrics, step losses, gradient norms, learning rates,
  and summary: `~/rag/runs/milestone4b-pilot-v1/`.
- Exact source/config/dependency snapshot: the run's `source/` directory.
- Log: `~/rag/logs/milestone4b-pilot.log`.
- Repository summary: [milestone4b-pilot-results.json](milestone4b-pilot-results.json).

Original execution command (refuses to overwrite the existing run directory):

```bash
source scripts/env.sh
conda activate "$RAG_ROOT/envs/retrieval"
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 \
  python -m scripts.train_pilot_4b
```

## Review point

No execution blocker remains. Preserve these results as the frozen pilot. Before
launching any additional seeds, review whether to replicate this exact comparison
or explicitly version a new development experiment. Changing the mining/loss/LR
would create a different experiment, not a replication. Seeds 20261003 and 20261004
have **not** been launched.
