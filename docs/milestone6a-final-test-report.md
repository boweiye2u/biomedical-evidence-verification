# Milestone 6A — final held-out SciFact TEST evaluation

> **Scope:** This is the first and only confirmatory evaluation of the frozen final system on the official BEIR SciFact TEST split. No architecture, prompt, checkpoint, evidence depth, label mapping, or decoding choice was selected from these results.

## Pre-run validation

Before any TEST record was opened:

- The final zero-shot BGE → top-1 full abstract → Qwen system was frozen in `configs/final-test-6a-v1.json`.
- The new external run directory was verified absent and then created with a pre-TEST source/config/dependency snapshot.
- Model revisions, prompt hash, archive hashes, seeds, and prior DEV artifact hashes were recorded.
- The 809 training-derived TRAIN/DEV claim IDs were fixed for an overlap check.
- No DEV-dependent parameter is recomputed from TEST.
- The repository is not a Git repository, so no commit hash exists; exact file hashes replace it.
- All 34 tests passed before TEST access.

Ordinary query bootstrap was frozen for uncertainty because the 112 related-claim groups used in Milestone 4C cover only the training-derived DEV split. No groups were invented from TEST labels.

## Documented implementation correction

The first TEST read stopped before retrieval, generation, or metric results because BEIR `qrels/test.tsv` did not align with original `claims_test.jsonl`. Exact inspection established the release mapping:

- BEIR TEST has 300 queries.
- Original SciFact `claims_dev.jsonl` has the same 300 IDs and zero claim-text mismatches.
- Original `claims_test.jsonl` has a disjoint 300-ID set and no evidence field.
- `claims_dev.jsonl` contains the released verification evidence annotations for 188 claims.

The original-annotation path was therefore corrected to `data/claims_dev.jsonl`. This is an allowed file-path/split-name correction. It was made before any ranking or model output existed. No model, prompt, evidence depth, label rule, or metric changed. A later reporting-only correction added an `INVALID` confusion-matrix column and linked this correction into the final summary; no prediction or primary metric changed.

## Frozen system

The primary T1 pipeline is:

```text
claim
→ BAAI/bge-base-en-v1.5 zero-shot retrieval
→ top-1 full abstract
→ Qwen/Qwen2.5-7B-Instruct grounded-v2
→ SUPPORT / CONTRADICT / INSUFFICIENT
```

BGE revision is `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, with the established query prefix, CLS pooling, L2-normalized FP32 embeddings, and exact FAISS inner-product search over all 5,183 documents.

Qwen revision is `a09a35458c702b33eeacc393d103063234e8bc28`, loaded in BF16 with the grounded-v2 prompt SHA-256 `21c9360a72834e37fcc37936190e290638a924e8985c50bf6dcff99472ba50b4`, greedy decoding, and `max_new_tokens=192`.

T_gold is diagnostic: it supplies the first numeric annotated evidence document, matching Milestone 5A D3. It is not a strict upper bound.

## TEST retrieval

TEST contains 300 queries and 339 BEIR known-positive assignments. Most queries have one known relevant document: min/median/max are 1/1/5 and the mean is 1.13.

| Metric | TEST | 95% CI where predefined |
|---|---:|---:|
| NDCG@10 | 0.7404 | [0.6998, 0.7795] |
| Recall@10 | 0.8742 | — |
| Recall@100 | 0.9667 | — |
| MRR@10 | 0.7034 | — |
| Top-1 BEIR known-positive rate | 0.6167 | — |

Of the 339 known positives, 185 rank first, 111 rank 2–10, 33 rank 11–100, and 10 fall beyond the saved top 100. The query-level NDCG interval uses 20,000 ordinary bootstrap replicates with seed 20261016.

## TEST verification

Gold label distribution is 124 SUPPORT, 64 CONTRADICT, and 112 INSUFFICIENT.

| Condition | Accuracy | Macro F1 | Invalid | Citation validity | Abstention |
|---|---:|---:|---:|---:|---:|
| T1 — frozen final system | 0.7000 | 0.6859 | 0.0033 | 0.9967 | 0.3867 |
| T_gold — annotated full document | 0.8300 | 0.8003 | 0.0033 | 0.9967 | 0.4533 |

T1 uncertainty:

- Accuracy 95% CI: **[0.6467, 0.7500]** using 20,000 query-bootstrap replicates, seed 20261017.
- Macro-F1 95% CI: **[0.6298, 0.7391]** using 20,000 query-bootstrap replicates, seed 20261018.

T1 class results:

| Label | Precision | Recall / class accuracy | F1 |
|---|---:|---:|---:|
| SUPPORT | 0.7541 | 0.7419 | 0.7480 |
| CONTRADICT | 0.6230 | 0.5938 | 0.6080 |
| INSUFFICIENT | 0.6897 | 0.7143 | 0.7018 |

T_gold class F1 is 0.8368 SUPPORT, 0.6607 CONTRADICT, and 0.9032 INSUFFICIENT. Its lower CONTRADICT recall of 0.5781 shows that supplying an annotated document does not remove verifier reasoning difficulty.

T1 confusion matrix:

| Gold \ Predicted | SUPPORT | CONTRADICT | INSUFFICIENT | INVALID |
|---|---:|---:|---:|---:|
| SUPPORT | 92 | 8 | 23 | 1 |
| CONTRADICT | 13 | 38 | 13 | 0 |
| INSUFFICIENT | 17 | 15 | 80 | 0 |

There are 299 schema-valid outputs and one truncated invalid JSON output, which remains invalid under the frozen parser. Citation validity is also 299/300. Mean evidence length is 572.7 tokens with no evidence truncation. Mean input length is 776.6 tokens (363–2,459), and mean generated output length is 69.7 tokens (34–153).

## Evidence-selection diagnostics

The targets remain separate:

- BEIR top-1 known-positive rate, measuring cited-document retrieval: **0.6167**.
- Original SciFact annotated verification-evidence top-1 coverage among 188 labeled-evidence claims: **149/188 = 0.7926**.
- Annotated verification-evidence top-3 coverage: **171/188 = 0.9096**.

T1 verification accuracy is **0.7584** when an annotated evidence document is selected and **0.4359** when it is not. Absence of an annotated document does not prove the selected article is useless, and these conditional subsets are descriptive.

## Error decomposition

T1 makes 90 errors:

| Conservative category | Count |
|---|---:|
| Empty-annotation / INSUFFICIENT ambiguity | 32 |
| Annotated evidence absent at rank 1 | 22 |
| Annotated evidence selected but Qwen wrong | 35 |
| Invalid output | 1 |

The categories do not assign causal shares. Selected reviewed examples show recurring polarity and direction errors:

- CX3CR1 promotes Th2 survival, but Qwen labels a claim that it impairs survival as SUPPORT.
- Chenodeoxycholic acid increases energy expenditure, but Qwen supports a claim that it reduces expenditure.
- PD-1 triggering induces IL-10, but Qwen supports a claim that it reduces IL-10.
- Female prisoner self-harm rates exceed male rates, but Qwen supports the reversed comparison.
- Loss of PKCζ enhances tumorigenesis, but Qwen treats this as support that PKCζ causes enhancement.
- Several correct-document cases end in abstention because Qwen demands a more literal match than the benchmark annotation.

The selected review is descriptive and was produced after evaluation solely for reporting. It was not used to alter the system.

## DEV versus TEST

All differences are descriptive; no DEV–TEST significance test was predefined.

Retrieval:

| Metric | DEV | TEST | TEST − DEV |
|---|---:|---:|---:|
| NDCG@10 | 0.7604 | 0.7404 | −0.0200 |
| Recall@10 | 0.8938 | 0.8742 | −0.0196 |
| Recall@100 | 0.9815 | 0.9667 | −0.0148 |
| MRR@10 | 0.7279 | 0.7034 | −0.0245 |

Verification:

| Condition/metric | DEV | TEST | TEST − DEV |
|---|---:|---:|---:|
| T1-equivalent accuracy | 0.7654 | 0.7000 | −0.0654 |
| T1-equivalent macro F1 | 0.7531 | 0.6859 | −0.0672 |
| Gold-document accuracy | 0.8765 | 0.8300 | −0.0465 |
| Gold-document macro F1 | 0.8636 | 0.8003 | −0.0634 |

Zero-shot BGE generalizes with a modest descriptive decline across all four retrieval metrics. End-to-end verification declines more, but remains within the predefined bootstrap uncertainty. The DEV qualitative conclusion is preserved: evidence selection matters substantially, while verifier reasoning remains an independent limitation even with annotated documents.

## Reproducibility and validation

All **34 tests pass**. Final independent checks reconstruct retrieval and verification metrics, compare retrieval against trusted IR metrics, verify the bootstrap seeds, validate exact TRAIN/DEV/TEST disjointness, verify all 600 raw outputs, confirm prompt and model hashes, and recheck prior DEV artifact hashes.

Commands:

```bash
source scripts/env.sh
python -m scripts.preflight_final_test_6a
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_final_test_retrieval_6a
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_final_test_verification_6a
python -m scripts.evaluate_final_test_6a
python -m pytest -q
```

## Artifacts

- Repository results: `docs/milestone6a-final-test-results.json`
- Frozen configuration and documented path correction: `configs/final-test-6a-v1.json`
- External run: `/home/boweiye2/rag/runs/milestone6a-test-v1`
- Pre-run manifest: `/home/boweiye2/rag/runs/milestone6a-test-v1/pre-run-manifest.json`
- Implementation correction: `/home/boweiye2/rag/runs/milestone6a-test-v1/implementation-corrections.json`
- Retrieval rankings and per-query metrics: `/home/boweiye2/rag/runs/milestone6a-test-v1/{retrieval-rankings.json,retrieval-cosine-scores.json,retrieval-per-query.json}`
- T1 and T_gold generations/evaluations: `/home/boweiye2/rag/runs/milestone6a-test-v1/verification/`
- Error decomposition: `/home/boweiye2/rag/runs/milestone6a-test-v1/{error-decomposition.json,selected-error-review.json}`
- Final summary: `/home/boweiye2/rag/runs/milestone6a-test-v1/final-summary.json`
- Pre-TEST snapshot: `/home/boweiye2/rag/runs/milestone6a-test-v1/source/pre-test-source-config-dependencies.tar.gz`
- Final exact snapshot: `/home/boweiye2/rag/runs/milestone6a-test-v1/source/exact-source-config-dependencies.tar.gz`
- Logs: `/home/boweiye2/rag/logs/milestone6a-{preflight,retrieval,verification,evaluation}.log`

**No post-TEST tuning, model selection, prompt change, reranking, passage change, retraining, or additional architecture evaluation was performed. Milestone 6A stops here.**
