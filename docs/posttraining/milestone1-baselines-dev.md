# Biomedical Evidence Verification v2 — Milestone 1 DEV baselines

**Date:** 2026-10-08  
**Status:** Complete; Milestone 1 stops here  
**Scope:** Frozen 162-claim DEV evaluation of matched zero-shot and few-shot Qwen2.5 baselines. No fixed 300-claim benchmark record was loaded, and no training, calibration, retrieval change, or model selection was performed.

## Systems and frozen settings

All four systems use the frozen zero-shot `BAAI/bge-base-en-v1.5` top-1 ranking over the 5,183-document SciFact corpus and supply one full abstract with original zero-based sentence indices. B0 was run on DEV because the historical Milestone 0 B0 used a different output schema and benchmark split, so it was not a strictly matched comparator for B1.

| ID | Generator | Prompting | Config |
|---|---|---|---|
| B0 | Qwen2.5-7B-Instruct | zero-shot | `configs/posttraining/baseline-b0-7b-zero-dev-v2.json` |
| B1 | Qwen2.5-7B-Instruct | 3-shot | `configs/posttraining/baseline-b1-7b-fewshot-v2.json` |
| B2 | Qwen2.5-3B-Instruct | zero-shot | `configs/posttraining/baseline-b2-3b-zero-v2.json` |
| B3 | Qwen2.5-3B-Instruct | 3-shot | `configs/posttraining/baseline-b3-3b-fewshot-v2.json` |

Model and tokenizer revisions are pinned identically within each model size:

- 7B: `a09a35458c702b33eeacc393d103063234e8bc28`
- 3B: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- BGE: `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`

Inference used vLLM 0.9.2 and Transformers 4.53.2 on one NVIDIA L40S: BF16, tensor parallel size 1, maximum model length 16,384, maximum sequences 16, request concurrency 8, GPU-memory utilization 0.72, temperature 0, repetition penalty 1.05, maximum 192 new tokens, and seed 20261009. B1 and B3 differ from their zero-shot counterparts only by the frozen exemplars.

The output schema is:

```json
{"decision":"SUPPORT|CONTRADICT|INSUFFICIENT","rationale_sentences":[0]}
```

The prompt SHA-256 is `000f68b1f43603bbd3710275e61a81bd09358dfac7eccacaf84762715efd9211`.

## Frozen few-shot exemplars

The three examples were selected and written to `configs/posttraining/fewshot-exemplars-v1.json` before any B1 or B3 DEV generation. The artifact SHA-256 is `9a559d0fe392d7230510b4e02d038448cf6fc7675a86e657c2522e43e5459137`.

| Claim | Document | Label | Rationale | Selection reason |
|---:|---:|---|---|---|
| 34 | 11705328 | SUPPORT | sentence 4 | One direct annotated rationale for a concise positive relation |
| 2 | 13734012 | CONTRADICT | sentence 4 | One direct numerical contradiction |
| 4 | 31460499 | INSUFFICIENT | none | Unrelated TRAIN-associated document demonstrating evidence-conditioned abstention |

All three claim IDs are in frozen TRAIN, none is in DEV, all labels are represented, and no fixed-benchmark example is used. B1 and B3 consume the identical serialized exemplar artifact.

The longest actual B1/B3 DEV prompt is 2,568 tokens. With the 192-token generation reserve, the maximum is 2,760/16,384 tokens, leaving 13,624 tokens. The 3B and 7B rendered prompt lengths are identical, and no input was truncated.

## DEV results

Metrics use the original claim-level SciFact label under the scoring rule recovered in Milestone 0. Invalid decisions would be assigned `INVALID` and scored incorrect; rationale-index validity is reported separately.

| System | Accuracy | Macro F1 | SUPPORT F1 | CONTRADICT F1 | INSUFFICIENT F1 |
|---|---:|---:|---:|---:|---:|
| B0 — 7B zero-shot | 0.7531 | 0.7476 | 0.7642 | 0.7213 | 0.7571 |
| B1 — 7B few-shot | 0.7531 | 0.7356 | 0.7656 | 0.6667 | 0.7746 |
| B2 — 3B zero-shot | 0.6358 | 0.4963 | 0.7273 | 0.1290 | 0.6325 |
| B3 — 3B few-shot | 0.6852 | 0.5388 | 0.7534 | 0.1333 | 0.7297 |

| System | SUPPORT recall | CONTRADICT recall | INSUFFICIENT recall | Predicted S / C / I |
|---|---:|---:|---:|---:|
| B0 | 0.6714 | 0.7857 | 0.8281 | 53 / 33 / 76 |
| B1 | 0.7000 | 0.6429 | 0.8594 | 58 / 26 / 78 |
| B2 | 0.9143 | 0.0714 | 0.5781 | 106 / 3 / 53 |
| B3 | 0.7857 | 0.0714 | 0.8438 | 76 / 2 / 84 |

B1 and B0 have equal accuracy. B1 macro F1 is 0.0119 lower, with 20 changed predictions: each system is uniquely correct on nine claims. Few-shot prompting therefore did not help the 7B model on this DEV comparison.

B3 improves over B2 by 0.0494 accuracy and 0.0426 macro F1. It is uniquely correct on 19 claims versus 11 for B2. This is an aggregate DEV improvement, but it does not produce balanced verification: both 3B systems recover only 2/28 CONTRADICT claims. The gain is primarily a redistribution between SUPPORT and INSUFFICIENT predictions.

## Output validity

| System | JSON valid | Schema valid | Decision valid | Rationale valid | Invalid decisions |
|---|---:|---:|---:|---:|---:|
| B0 | 162/162 | 162/162 | 162/162 | 162/162 | 0 |
| B1 | 162/162 | 162/162 | 162/162 | 162/162 | 0 |
| B2 | 162/162 | 162/162 | 162/162 | 161/162 | 0 |
| B3 | 162/162 | 162/162 | 162/162 | 162/162 | 0 |

B2 claim 590 predicts `CONTRADICT` with an empty rationale list. Its decision remains valid and is scored normally; only the separately reported rationale constraint fails.

## Efficiency

| System | Wall time (s) | Requests/s | Mean request (ms) | P50 (ms) | P95 (ms) | Peak GPU MiB |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 10.14 | 15.97 | 489.59 | 475.20 | 593.61 | 33,850 |
| B1 | 10.56 | 15.34 | 512.17 | 501.45 | 654.26 | 33,850 |
| B2 | 5.67 | 28.57 | 275.13 | 244.85 | 427.50 | 33,892 |
| B3 | 5.90 | 27.47 | 279.63 | 261.76 | 399.44 | 33,892 |

The memory value is observed process-level GPU use while vLLM reserves 72% of the device; it is not the size of model weights alone. Timing covers each 162-request evaluation after the server was ready and does not include server startup or model loading.

## Documented formatting correction

The first version of the v2 prompt described the schema but did not contain a literal JSON object. A B0 DEV format-feasibility run returned shorthand such as `INSUFFICIENT,[]` for all 162 records, producing no valid JSON. That failed run is preserved at `/home/boweiye2/rag/runs/posttraining/baselines-v1/B0` and was not overwritten.

Before B1-B3 evaluation, one versioned formatting-only correction added a literal JSON example and required `{` and `}` as the first and last characters. It did not change the labels, evidence, scoring, exemplars, retrieval, or scientific decision rule. A structural feasibility check on only the three frozen TRAIN exemplars then produced 3/3 valid JSON objects. The corrected prompt and configs carry the `v2` suffix, and all reported comparisons use those matched configs. No DEV result was used to select exemplars or scientific content.

This matched B0 is distinct from the historical grounded-v2 baseline reported in earlier project milestones. Prompt/schema interactions prevent treating their small metric difference as a direct model-quality change.

## Reproducibility and validation

The runner refuses an existing output directory, validates immutable model revisions and prompt hash, checks the frozen 647/162 split, confirms exact DEV ID alignment for mappings and rankings, and rejects a prompt whose rendered length plus generation reserve exceeds 16,384 tokens. Tests cover TRAIN-only exemplars, no DEV leakage, no benchmark use, shared B1/B3 exemplars, schema parsing, revision pinning, scoring, prompt length, and generated-ID alignment.

Commands used, with the corresponding 7B or 3B vLLM server already running on port 8001, were:

```bash
source scripts/env.sh

# Run once for each corrected config, changing CONFIG, TOKENIZER, and OUTPUT.
/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.baselines \
  --config "$CONFIG" \
  --mapping /home/boweiye2/rag/runs/milestone5a-verification-v1/verification-mapping.json \
  --rankings /home/boweiye2/rag/runs/scifact_dev_v1/bge-rankings.json \
  --corpus /home/boweiye2/rag/data/scifact/original/corpus.jsonl \
  --tokenizer "$TOKENIZER" \
  --output-dir "$OUTPUT" \
  --base-url http://127.0.0.1:8001

/home/boweiye2/rag/envs/posttraining/bin/python -m pytest -q \
  tests/posttraining \
  tests/test_verification.py \
  tests/test_scifact_preparation.py
```

Repository artifacts:

- `posttraining/evaluation/baselines.py`
- `tests/posttraining/test_baselines.py`
- `configs/posttraining/fewshot-exemplars-v1.json`
- `configs/posttraining/baseline-{b0-7b-zero-dev,b1-7b-fewshot,b2-3b-zero,b3-3b-fewshot}-v2.json`
- `docs/posttraining/milestone1-baselines-dev-results.json`
- this report

The corresponding `v1` baseline configs and failed B0 output are retained as an audit trail for the documented formatting correction.

External artifacts:

- Full run: `/home/boweiye2/rag/runs/posttraining/baselines-v1/`
- Corrected outputs: `B0-v2/`, `B1-v2/`, `B2-v2/`, and `B3-v2/`
- Raw generations and per-example predictions: each corrected output directory's `predictions.jsonl`
- Per-run metrics, prompt lengths, and frozen config: each corrected output directory
- TRAIN-only feasibility: `feasibility-v2/summary.json`
- Actual prompt-budget validation: `preflight-v2.json`
- Aggregate summary: `summary.json`
- Exact source/config/dependency snapshot: `source/exact-source-config-dependencies.tar.gz`

## Exit decision

Milestone 1 passes. The requested B1-B3 DEV baselines and matched B0 comparator are complete, outputs are structurally valid apart from one separately recorded B2 rationale constraint, and no fixed-benchmark data was loaded. The principal issue for Milestone 2 planning is the 3B model's severe CONTRADICT-class failure under both prompting conditions. No SFT data construction, Pool B/C labeling, LoRA training, calibration, benchmark evaluation, or other Milestone 2 work was started.
