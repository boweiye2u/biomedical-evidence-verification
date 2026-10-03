# Milestone 5A — evidence-grounded biomedical claim verification on DEV

> **Scope:** Prompt and evidence-budget selection and all reported metrics use the same frozen 162-claim DEV set. These results characterize the completed DEV pipeline and are not estimates of unseen TEST generalization.

No retriever was trained, no TEST qrels or verification labels were read, and no reranking, serving, or performance benchmark was run.

## Verification mapping

All 162 frozen claim IDs align exactly with original SciFact training annotations. Labels are derived from the original evidence sets: **70 SUPPORT, 28 CONTRADICT, and 64 INSUFFICIENT**. The 98 claims with annotated evidence contain 108 document assignments and 189 alternative evidence sets. Empty evidence maps to the benchmark INSUFFICIENT label; it does not prove that no relevant evidence exists.

## Frozen model and prompt

The local model is **Qwen/Qwen2.5-7B-Instruct** at revision `a09a35458c702b33eeacc393d103063234e8bc28`, loaded in BF16 with SDPA on one L40S. It was selected as a single practical 7B instruction model because it supports structured JSON generation, fits comfortably on one GPU, and is compatible with later vLLM work.

Greedy decoding uses `max_new_tokens=192`, seed `20261009`, a 16,384-token input limit, and a 12,000-token evidence limit. The frozen prompt is **grounded-v2** with SHA-256 `21c9360a72834e37fcc37936190e290638a924e8985c50bf6dcff99472ba50b4`. It requires evidence-only decisions, exact supplied citation IDs, strict JSON, and abstention when the evidence is inadequate.

A three-example feasibility check exposed Qwen's tendency to convert `[DOC 123]` to `DOC_123`. The single permitted evidence-format variant changed identifiers to `[EVIDENCE_ID: 123]`; the final feasibility run produced exact IDs on SUPPORT and CONTRADICT and an empty list on INSUFFICIENT.

## Bounded DEV development

The complete search was two prompts × top-1/top-3/top-5 using D1 only. Selection was fixed as macro F1, accuracy, invalid-output rate, smaller top-k, then prompt name.

| Prompt | k | Accuracy | Macro F1 | Invalid | Citation validity | Annotated-evidence coverage |
|---|---:|---:|---:|---:|---:|---:|
| grounded-v2 | 1 | 0.7654 | 0.7531 | 0.0000 | 1.0000 | 0.8571 |
| grounded-v2 | 3 | 0.7284 | 0.7269 | 0.0185 | 0.9815 | 0.9490 |
| grounded-v2 | 5 | 0.6790 | 0.7034 | 0.0679 | 0.9321 | 0.9694 |
| baseline-v1 | 3 | 0.3025 | 0.3988 | 0.6049 | 0.5679 | 0.9490 |
| baseline-v1 | 5 | 0.2346 | 0.3686 | 0.6914 | 0.4259 | 0.9694 |
| baseline-v1 | 1 | 0.1790 | 0.2241 | 0.7469 | 0.4630 | 0.8571 |

The selected configuration is **grounded-v2, top-1**. More documents increased annotated-evidence coverage but reduced label metrics and strict-output validity for this model/prompt setup. No contexts reached the 12,000-token evidence cap.

## Final DEV results

D1 uses frozen zero-shot BGE. D2 uses the already-selected random-negative seed 20261003 epoch 3 checkpoint because it had the highest frozen retrieval NDCG among the selected random checkpoints; verification results played no role in that choice. D3 supplies the first annotated evidence document under the same top-1 budget and full-abstract format.

| Condition | Accuracy | Macro F1 | SUPPORT acc/F1 | CONTRADICT acc/F1 | INSUFFICIENT acc/F1 | Invalid rate | Citation validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0.7654 | 0.7531 | 0.7857 / 0.8148 | 0.7857 / 0.6984 | 0.7344 / 0.7460 | 0.0000 | 1.0000 |
| D2 | 0.7654 | 0.7531 | 0.7857 / 0.8088 | 0.7857 / 0.6984 | 0.7344 / 0.7520 | 0.0000 | 1.0000 |
| D3 | 0.8765 | 0.8636 | 0.8000 / 0.8682 | 0.7857 / 0.8148 | 1.0000 / 0.9078 | 0.0000 | 1.0000 |

D1 and D2 both reach **0.7654 accuracy**; macro F1 is 0.75309 for D1 and 0.75308 for D2. They share the same top document on 129/162 claims and the same label on 156/162. D1 is uniquely correct on two claims and D2 on two, so the aggregate tie does not establish retrieval equivalence or a causal explanation.

D3 reaches **0.8765 accuracy / 0.8636 macro F1**. It is a diagnostic reference, not a strict upper bound: it uses only one annotated document, some claims have alternative evidence sets or documents, and full abstracts can obscure short rationales.

## Evidence diagnostics

| Condition | Annotated evidence coverage | Accuracy when retrieved | Accuracy when not retrieved | Abstention rate | Mean evidence tokens | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 0.8571 | 0.8333 | 0.5000 | 0.3827 | 534.2 | 0 |
| D2 | 0.8571 | 0.8214 | 0.5714 | 0.3765 | 528.2 | 0 |
| D3 | 1.0000 | 0.7959 | — | 0.4753 | 334.8 | 0 |

Coverage is computed only for the 98 SUPPORT/CONTRADICT claims with annotated evidence. Absence means no annotated gold document was supplied; it does not prove that the retrieved document is useless.

## Conservative error analysis

| Condition | Annotated evidence not retrieved | Evidence retrieved, model misclassified | Insufficient/ambiguous annotation case | Citation/output failure |
|---|---:|---:|---:|---:|
| D1 | 7 | 14 | 17 | 0 |
| D2 | 6 | 15 | 17 | 0 |
| D3 | 0 | 20 | 0 | 0 |

Observed examples include abstention despite an annotated full abstract, difficulty reasoning about intervention direction (for example, knockout causing BBB disruption), and cases with empty original annotations where a retrieved abstract appears related enough for the model to predict SUPPORT or CONTRADICT. These are descriptive patterns. The artifacts do not establish annotation error, false-negative evidence, or a single model failure mechanism.

## Reproducibility and limitations

Final D1 reproduced every selected-development raw output exactly. All three conditions contain 162 aligned records, strict schema and citation validation is automatic, and all 22 tests pass. Model revision, prompt hash, frozen config, generations, token counts, evaluation records, and dependency versions are saved.

The main limitation is deliberate DEV reuse: the prompt and top-k were selected on the same claims reported here. TEST remains untouched and is required for confirmatory evaluation. D1/D2 differences cannot be attributed solely to retriever quality because changed contexts interact with the LLM.

## Commands

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_verification_5a --phase feasibility
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_verification_5a --phase development
python -m scripts.evaluate_verification_5a --phase development
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_verification_5a --phase final
python -m scripts.evaluate_verification_5a --phase final
python -m pytest -q
```

## Artifacts

- Versioned external run: `/home/boweiye2/rag/runs/milestone5a-verification-v1`
- Mapping: `/home/boweiye2/rag/runs/milestone5a-verification-v1/verification-mapping.json`
- Development summary: `/home/boweiye2/rag/runs/milestone5a-verification-v1/development-summary.json`
- Final summary and per-condition evaluations: `/home/boweiye2/rag/runs/milestone5a-verification-v1/final-summary.json` and `/home/boweiye2/rag/runs/milestone5a-verification-v1/final`
- Conservative error records: `/home/boweiye2/rag/runs/milestone5a-verification-v1/error-analysis.json`
- Exact source/config/dependency snapshot: `/home/boweiye2/rag/runs/milestone5a-verification-v1/source/exact-source-config-dependencies.tar.gz`
- Frozen config: `configs/verification-5a-v1.json`

Milestone 5A stops here.
