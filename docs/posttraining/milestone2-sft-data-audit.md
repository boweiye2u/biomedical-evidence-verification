# Biomedical Evidence Verification v2 — Milestone 2 SFT data construction and audit

**Date:** 2026-10-08  
**Status:** Complete; both audited pools retained  
**Scope:** TRAIN-only construction of Conditions A–D. No DEV label, rationale, or example and no fixed 300-claim benchmark record was loaded. No mixed dataset was created and no model was trained.

## Frozen inputs and semantics

The source population is the 647-claim frozen TRAIN partition of original SciFact `claims_train.jsonl`. Retrieval uses the unchanged zero-shot `BAAI/bge-base-en-v1.5` revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, with the established query prefix, CLS pooling, normalized FP32 embeddings, and exact inner-product search over all 5,183 documents.

Every row keeps three separate fields:

- `claim_gold_label`: the original claim-level SciFact status.
- `pair_annotation`: SUPPORT or CONTRADICT only for an originally annotated exact claim–document pair; otherwise null.
- `context_label`: the evidence-conditioned SFT target.

No missing pair annotation was automatically converted to INSUFFICIENT in Pool B or C. Those labels were assigned only after the corresponding independently frozen filter and audit passed. Condition D follows the protocol's random-negative INSUFFICIENT rule. The MIXED guard remains implemented; the frozen TRAIN population contains zero MIXED claims.

## Condition A — annotated gold evidence

Condition A contains one row for every annotated TRAIN claim–document pair.

| Diagnostic | Count |
|---|---:|
| Rows | 456 |
| Unique claims | 407 |
| Unique documents | 284 |
| SUPPORT | 292 |
| CONTRADICT | 164 |

The **164 CONTRADICT gold examples** are explicitly retained as the pre-SFT availability diagnostic. Their count does not establish why the 3B baselines recovered only 2/28 CONTRADICT DEV claims.

SciFact supplies multiple alternative sufficient rationale sets for some exact pairs, while this milestone requires one row per pair and no duplicate claim–document pairs. The deterministic clarification is to choose the shortest rationale set, breaking ties by the lexicographically earliest sentence-index tuple, and retain all alternatives in `alternative_rationale_sets`. This preserves 768 original alternative sets across the 456 rows. Selected rationales contain one sentence for 432 pairs, two for 20, three for three, and four for one. Every index was validated against its supplied abstract.

## Pool B — retrieved non-gold top 1

Of 647 TRAIN claims, 310 have a BGE top-1 document that is not an annotated evidence document. These form the candidate pool with null `pair_annotation` and null `context_label` before auditing.

The filter-development sample contains 50 candidates drawn with seed `20261011`:

| Judgment | Count |
|---|---:|
| SUPPORT | 6 |
| CONTRADICT | 4 |
| INSUFFICIENT | 40 |

Pre-filter contamination is **10/50 = 20%**, Wilson 95% CI **[11.24%, 33.04%]**.

The frozen Pool B filter is an absolute BGE-cosine ceiling: retain scores **≤ 0.660**. The lowest contaminated development-sample score was 0.664836764. This rule was written to `configs/posttraining/data-audit-v1.json` before drawing the fresh audit. It retains 86/310 candidates.

The fresh 50-item sample uses seed `20261012`, is disjoint from filter development, and was drawn after applying the filter:

| Judgment | Count |
|---|---:|
| SUPPORT | 0 |
| CONTRADICT | 1 |
| INSUFFICIENT | 49 |

Post-filter contamination is **1/50 = 2%**, Wilson 95% CI **[0.35%, 10.50%]**. This is below the frozen strictly-greater-than-15% drop threshold, so **Pool B is retained**, yielding 86 usable Condition B rows.

## Pool C — non-gold ranks 2–10

Removing annotated exact pairs from BGE ranks 2–10 produces 5,716 candidates. Pool C was audited separately from Pool B.

The filter-development sample contains 50 candidates drawn with seed `20261013`:

| Judgment | Count |
|---|---:|
| SUPPORT | 0 |
| CONTRADICT | 1 |
| INSUFFICIENT | 49 |

Pre-filter contamination is **1/50 = 2%**, Wilson 95% CI **[0.35%, 10.50%]**.

Because this independent estimate was already below the frozen threshold, the Pool C rule was frozen as **no filter** before re-audit. The fresh disjoint 50-item audit uses seed `20261014`:

| Judgment | Count |
|---|---:|
| SUPPORT | 1 |
| CONTRADICT | 0 |
| INSUFFICIENT | 49 |

Frozen-filter contamination remains **1/50 = 2%**, Wilson 95% CI **[0.35%, 10.50%]**. Pool C therefore passes its independent stop rule and **is retained**, yielding all 5,716 Condition C rows.

## Condition D — random negatives

Condition D contains one deterministic random-negative document for every TRAIN claim: 647 rows using seed `20261010`. Sampling excludes that claim's annotated documents, cited documents, and frozen BGE top-10 documents. The last exclusion prevents a random pair from duplicating B or C. Every row has null `pair_annotation`, `context_label = INSUFFICIENT`, and an empty rationale list.

An initial integrity check found one random pair that duplicated Pool C. The sampler was corrected before finalization to exclude the full top 10, and Condition D was regenerated with the same seed. No audit, filter, or pool decision changed.

## Final usable availability

These are raw, deduplicated available rows, not a selected training mixture:

| Condition | Rows | Context labels |
|---|---:|---|
| A — gold | 456 | 292 SUPPORT, 164 CONTRADICT |
| B — filtered retrieved top 1 | 86 | 86 INSUFFICIENT |
| C — audited ranks 2–10 | 5,716 | 5,716 INSUFFICIENT |
| D — random | 647 | 647 INSUFFICIENT |
| **Total availability** | **6,905** | **292 SUPPORT, 164 CONTRADICT, 6,449 INSUFFICIENT** |

Pool B is the limiting source for Mix-1 and Mix-2 if future grouped TRAIN cross-validation samples without replacement. No proportions were forced, no oversampling or class rebalancing was applied, and `train_mixed.jsonl` was deliberately not created. Mixture selection remains a later TRAIN-only grouped-cross-validation task and was not informed by DEV.

## Audit provenance and limitation

The four reviews are analyst-assisted language-model judgments recorded in the requested `human_pair_label` compatibility field. They are **not independent human annotations**, and no inter-rater agreement is claimed. The active Codex runtime performed the semantic review; the requested GPT-5.6 Sol model was not exposed as a switchable runtime in this session. This is a tooling deviation, not a change to sampling, filter design, stop threshold, or label semantics.

The audits assess whether each supplied abstract directly supports or contradicts the complete claim with matching entities, relation, polarity, and setting. Missing SciFact annotation alone was never treated as evidence of insufficiency. The Wilson intervals correctly reflect that 50-item audits provide limited precision; the 15% cutoff is the protocol's project heuristic, not a benchmark standard.

## Validation

The completed artifacts satisfy:

- all claim IDs belong to the 647 frozen TRAIN claims;
- zero DEV claim leakage and zero fixed-benchmark access;
- no duplicate claim–document pair within or across A–D;
- A pair annotations equal context labels;
- every selected A rationale index is within the supplied abstract;
- every retained B/C row has a versioned frozen-filter approval;
- B and C use independent seeds, samples, filters, and contamination estimates;
- each development/audit pair is disjoint;
- D is deterministic and excludes annotated, cited, and top-10 documents;
- all labels and provenance fields are valid;
- MIXED handling remains enforced despite zero observed MIXED TRAIN claims.

## Commands

```bash
source scripts/env.sh

# Construct A/D, B/C candidates, TRAIN top-10 rankings, and development samples.
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/boweiye2/rag/envs/posttraining/bin/python \
  -m posttraining.data.build_sft_data prepare

# After development judgments and freezing each filter in data-audit-v1.json:
/home/boweiye2/rag/envs/posttraining/bin/python \
  -m posttraining.data.build_sft_data make-audit --pool B
/home/boweiye2/rag/envs/posttraining/bin/python \
  -m posttraining.data.build_sft_data make-audit --pool C

# After the two fresh audits were completed:
/home/boweiye2/rag/envs/posttraining/bin/python \
  -m posttraining.data.build_sft_data finalize

/home/boweiye2/rag/envs/posttraining/bin/python -m pytest -q \
  tests/posttraining tests/test_verification.py tests/test_scifact_preparation.py
```

## Repository changes

- `configs/posttraining/data-audit-v1.json`
- `posttraining/data/build_sft_data.py`
- `tests/posttraining/test_sft_data.py`
- `docs/posttraining/milestone2-sft-data-audit-results.json`
- this report

## External artifacts

- Data root: `/home/boweiye2/rag/data/posttraining/` (29 MiB)
- `train_gold.jsonl`
- `train_retrieved.jsonl`
- `train_hard_negative.jsonl`
- `train_random_negative.jsonl`
- Candidate pools: `pool-{b,c}-candidates.jsonl`
- Audit CSVs: `audits/pool-{b,c}-{filter-development,frozen-filter-audit}.csv`
- Run root: `/home/boweiye2/rag/runs/posttraining/data-audit-v1/`
- Exact TRAIN top-10 rankings: `train-rankings-top10.jsonl`
- Final machine-readable summary: `summary.json`
- Logs: `/home/boweiye2/rag/logs/posttraining-milestone2-{prepare,finalize}.log`
- Exact source/config/dependency snapshot: `source/exact-source-config-dependencies.tar.gz`

Milestone 2 passes. Conditions A–D are constructed, both pool decisions are frozen, and no blocker remains for designing the next TRAIN-only gold-model training stage. The raw label distribution is highly dominated by available C/D insufficient contexts, so a future training milestone must apply only the predeclared mixture procedure rather than concatenate all 6,905 rows. No LoRA training or later milestone was started.
