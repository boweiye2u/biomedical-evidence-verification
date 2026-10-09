# Biomedical Evidence Verification v2 — Milestone 0 protocol verification

**Date:** 2026-10-08
**Status:** Complete; all Milestone 0 exit conditions satisfied
**Scope:** Configuration recovery, split/label validation, B0 compatibility reproduction, context-budget validation, environment creation, and repository/hardware inspection. No SFT or new v2 baseline comparison was run.

## Review outcome

The frozen proposal and implementation checklist are internally workable. No scientific-design blocker was found. Four implementation clarifications were made without changing the study:

1. The supplied frozen documents were copied from the repository root to their canonical `docs/posttraining/` paths. The originals were preserved.
2. Week 0 B0 is explicitly a historical-v1 compatibility reproduction using the v1 output schema (`label`, `evidence_ids`, `explanation`). Later v2 baselines and SFT use the frozen v2 schema (`decision`, `rationale_sentences`). These results must not be silently mixed.
3. The repository intentionally has separate retrieval and serving environments. A blanket `pytest` in the retrieval environment fails while importing FastAPI; the validated suite therefore runs serving tests in the serving environment and all other tests in the retrieval/post-training environment.
4. The 300-claim split is called the **fixed SciFact benchmark split, previously evaluated in v1 and not used for v2 training or model selection**. It is not described as a pristine unseen test set.

## Frozen specifications

Canonical copies:

- `docs/posttraining/2026-10-08-biomedical-llm-finetuning-proposal-final-v4-FROZEN-v2.md`
- `docs/posttraining/2026-10-08-biomedical-llm-posttraining-implementation-FINAL-FROZEN.md`

Milestone 0 configuration:

- `configs/posttraining/protocol-v1.json`
- `configs/posttraining/b0-reproduction-v1.json`

The scientific protocol controls design. The implementation checklist controls paths, environments, and commands. Later naming and benchmark-status patches in those documents override older wording.

## Split identity and label structure

The original SciFact `claims_train.jsonl` has 809 claims. The frozen grouped split partitions it exactly:

| Split | Claims | SUPPORT | CONTRADICT | INSUFFICIENT | MIXED |
|---|---:|---:|---:|---:|---:|
| TRAIN | 647 | 262 | 145 | 240 | 0 |
| DEV | 162 | 70 | 28 | 64 | 0 |

Validation results:

- TRAIN/DEV overlap: 0.
- TRAIN union DEV equals all 809 original training-file claims.
- TRAIN annotated claim-document pairs: 456.
- DEV annotated claim-document pairs: 108.
- Claims whose annotated documents carry different SUPPORT/CONTRADICT labels: 0 in TRAIN and 0 in DEV.
- No `MIXED` claim handling is activated for the present dataset, but the code and tests retain that state explicitly.

The fixed benchmark contains 300 claims. BEIR `scifact/qrels/test.tsv` query IDs/texts align with original SciFact archive member `data/claims_dev.jsonl`, reflecting the already documented release naming mismatch. It has zero claim-text mismatches and zero overlap with the 809 training-derived claims.

Frozen hashes:

- Split: `6a4dcc3d654621590b860ce6ec4ed5722c91913546386470d545c425f01a2ff0`
- Original `claims_train.jsonl`: `f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0`

## Exact v1 scoring rule

The v1 evaluation score uses the original claim-level SciFact label for every prediction. Whether retrieved top-1 is an annotated evidence document is recorded as an evidence-coverage diagnostic; it does not replace the claim label or automatically change the scoring target to INSUFFICIENT.

- Empty original evidence maps to claim label `INSUFFICIENT`.
- A non-annotated top-1 for a SUPPORT/CONTRADICT claim is still scored against that original claim label in the historical v1 endpoint.
- Therefore the final-v4 fallback scoring rule is not needed for B0 reproduction.
- For v2 training records, `claim_gold_label`, `pair_annotation`, and evidence-conditioned `context_label` remain distinct as required by the frozen protocol.

This distinction also preserves the known limitation that SciFact evidence annotations may be incomplete.

## Frozen retrieval configuration

| Field | Value |
|---|---|
| Model | `BAAI/bge-base-en-v1.5` |
| Revision | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |
| Query prefix | `Represent this sentence for searching relevant passages: ` |
| Document format | title + one space + abstract |
| Pooling | CLS |
| Embedding | normalized FP32 |
| Search | exact inner product |
| Index | `faiss.IndexFlatIP` |
| Corpus | 5,183 documents |
| Cache | `/home/boweiye2/rag/embeddings/scifact-bge-a53fd0d3b47586f0` |

The primary v2 study keeps this retriever frozen.

## Frozen vLLM B0 configuration

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| vLLM | 0.9.2 |
| Precision | BF16 |
| Tensor parallel size | 1 |
| Maximum model length | 16,384 |
| Maximum sequences | 16 |
| Fixed request concurrency | 8 |
| GPU-memory utilization | 0.72 |
| Temperature | 0 |
| Repetition penalty | 1.05 |
| Maximum new tokens | 192 |
| Seed | 20261009 |
| Prompt | grounded-v2 |
| Prompt SHA-256 | `21c9360a72834e37fcc37936190e290638a924e8985c50bf6dcff99472ba50b4` |
| Evidence | BGE top-1 full abstract |

The local launch uses `VLLM_USE_FLASHINFER_SAMPLER=0`, as documented in v1, because the host CUDA toolkit cannot compile the optional FlashInfer sampler. Greedy decoding is unchanged.

## B0 compatibility reproduction

B0 was rerun once on the allowed fixed 300-claim benchmark through the frozen FastAPI/vLLM stack. The original v1 artifacts were not modified.

| Result | Historical HF runner | Week 0 frozen vLLM |
|---|---:|---:|
| Accuracy | 0.7000 | 0.7033 |
| Macro F1 | 0.6859 | 0.6865 |
| Invalid outputs | 1/300 | 0/300 |
| Exact saved top-1 retrieval | — | 300/300 |

The two engines agree on 296/300 labels (98.67%). The four differing claims are recorded in `historical-comparison.json`. The net accuracy difference is +1 correct prediction out of 300 and the macro-F1 difference is +0.00057. This satisfies compatibility reproduction while preserving the documented fact that Hugging Face and vLLM greedy generation are not byte-identical.

External artifacts:

- `/home/boweiye2/rag/runs/posttraining/protocol-check-v1/b0-reproduction-v1/predictions.jsonl`
- `/home/boweiye2/rag/runs/posttraining/protocol-check-v1/b0-reproduction-v1/metrics.json`
- `/home/boweiye2/rag/runs/posttraining/protocol-check-v1/b0-reproduction-v1/historical-comparison.json`
- `/home/boweiye2/rag/runs/posttraining/protocol-check-v1/b0-service-requests.jsonl`

B1-B3 were not run on the fixed benchmark.

## Few-shot context budget

The official pinned Qwen2.5-3B-Instruct tokenizer was downloaded without model weights:

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Local tokenizer/config path: `/home/boweiye2/rag/models/Qwen2.5-3B-Instruct-tokenizer`

A conservative upper bound used the five longest SciFact corpus abstracts as demonstrations and the sixth-longest abstract as the target context. The prompt included the v2 JSON target and all five demonstrations.

| Quantity | Tokens |
|---|---:|
| Rendered prompt | 13,643 |
| Reserved generation | 192 |
| Total | 13,835 |
| Frozen maximum | 16,384 |
| Remaining | 2,549 |

The 3B and 7B tokenizer vocabularies and chat templates are equal, and both counted 13,643 prompt tokens. Therefore any eventual fixed set of up to five full-abstract demonstrations fits the frozen budget without silent truncation. Actual B1/B3 exemplar IDs must still be frozen before their DEV evaluation.

Artifact: `/home/boweiye2/rag/runs/posttraining/protocol-check-v1/fewshot-budget.json`.

## Hardware and software

Hardware:

- Host: `sn4622122456`
- GPUs: 8 × NVIDIA L40S
- VRAM: 46,068 MiB per GPU
- Driver: 565.57.01
- CUDA toolkit (`nvcc`): 12.2
- PyTorch CUDA runtime: 12.6
- NCCL: 2.27.5
- SLURM: not active; this is a single non-SLURM host

Frozen v1 environments remain unchanged:

| Environment | Python | PyTorch | Transformers | Other key packages |
|---|---|---|---|---|
| Retrieval | 3.11.16 | 2.9.1+cu126 | 4.57.6 | FAISS 1.15.1 |
| Serving | 3.11.17 | 2.7.0+cu126 | 4.53.2 | vLLM 0.9.2, FastAPI 0.116.1, FAISS 1.11.0 |

A separate post-training environment was created by cloning the validated retrieval environment and adding only the resolved training packages:

- Path: `/home/boweiye2/rag/envs/posttraining`
- PEFT: 0.21.2
- TRL: 1.15.0
- bitsandbytes: 0.50.2
- Accelerate: 1.15.0
- Datasets: 4.8.5

`pip check` reports no broken requirements. A tiny randomly initialized Qwen2 model with LoRA adapters completed a CUDA forward pass with output shape `[1, 8, 128]`. This is an environment smoke test, not model training.

Reproducibility files:

- `environments/posttraining-requirements.txt`
- `environments/posttraining-v1-pip-freeze.txt`
- `environments/posttraining-v1-conda-linux-64.explicit.txt`
- `/home/boweiye2/rag/runs/posttraining/protocol-check-v1/posttraining-environment.json`

## Repository hygiene

- Repository: `https://github.com/boweiye2u/biomedical-evidence-verification.git`
- Visibility: public, verified through the GitHub repository API
- Default branch: `main`
- Start commit: `db145b5e6526afefd7ad9fead083e4100fe53de2`
- Resume PDFs are present locally but ignored by `*.pdf`.
- Resume PDFs are not tracked in the current index and were not found in reachable Git history.
- The Git repository is inside Dropbox. Avoid concurrent multi-machine Git writes and push after each work session.
- Large data, tokenizers, models, generated predictions, logs, and run outputs remain under `/home/boweiye2/rag`.

## Validation

The original suite passes when run in its intended environments:

- Non-serving v1 tests: 34 passed.
- Serving v1 tests: 7 passed, with dependency deprecation warnings only.
- New Milestone 0 tests: 5 passed.
- Focused new-plus-relevant-v1 run: 17 passed.

The new tests cover claim/pair label separation, MIXED-state detection, empty-evidence labeling, pair-level conflict rejection, invalid-output scoring, and retrieval identity accounting.

## Commands

```bash
source scripts/env.sh

# Existing suites
/home/boweiye2/rag/envs/retrieval/bin/python -m pytest -q --ignore=tests/test_serving_6b.py
/home/boweiye2/rag/envs/serving/bin/python -m pytest -q tests/test_serving_6b.py

# Protocol check
/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.protocol_check \
  --artifact-root /home/boweiye2/rag \
  --output /home/boweiye2/rag/runs/posttraining/protocol-check-v1/protocol-check.json

# B0 service
CUDA_VISIBLE_DEVICES=0 scripts/start_vllm_6b.sh
SERVING_6B_CONFIG=configs/posttraining/b0-reproduction-v1.json \
  CUDA_VISIBLE_DEVICES=0 scripts/start_api_6b.sh

# B0 client
/home/boweiye2/rag/envs/serving/bin/python -m posttraining.evaluation.reproduce_b0 \
  --mapping /home/boweiye2/rag/runs/milestone6a-test-v1/test-claim-mapping.json \
  --rankings /home/boweiye2/rag/runs/milestone6a-test-v1/retrieval-rankings.json \
  --output-dir /home/boweiye2/rag/runs/posttraining/protocol-check-v1/b0-reproduction-v1 \
  --concurrency 8
```

## Exit criteria

| Criterion | Result |
|---|---|
| B0 reproduced | Pass |
| v1 scoring known | Pass |
| Splits and benchmark status documented | Pass |
| Retrieval frozen | Pass |
| vLLM configuration frozen | Pass |
| Compute availability known | Pass |
| Few-shot prompts fit `max_model_len` | Pass |

Milestone 0 stops here. No SFT, B1-B3 DEV run, Pool B/C audit, full fine-tuning, FSDP scaling, or new benchmark evaluation was started.
