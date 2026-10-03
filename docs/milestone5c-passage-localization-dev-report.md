# Milestone 5C — passage-level evidence localization on DEV

> **Scope:** All construction, scoring, generation, and analysis use the same frozen 162-claim DEV set. These are post-development diagnostics, not estimates of unseen-TEST generalization.

No TEST qrels or verification labels were read. No retriever, reranker, or verifier was trained, and no prompt, decoding, serving, or latency work was changed. The optional P3 condition was not run.

## Frozen setup and passage construction

P1 reuses the bitwise-identical Milestone 5A D1 outputs: zero-shot BGE top-1 full abstract with frozen Qwen. P2 takes the existing zero-shot BGE top three documents, scores every sentence globally with the existing MedCPT cross-encoder, and supplies only the highest-scoring sentence. G_full reuses Milestone 5A D3. G_rationale supplies one deterministically selected annotated rationale set, or no evidence for claims with empty annotations.

The deterministic splitter is the original SciFact corpus `abstract` sentence array. It preserves the benchmark's text, ordering, and zero-based rationale indices exactly; no external sentence segmentation or normalization is applied. A passage is one original sentence. MedCPT receives `[claim, title + sentence]`, while Qwen receives the document title and original indexed sentence with the frozen `[EVIDENCE_ID: <id>]` format.

The gold-rationale rule was frozen before evaluation: minimize `(sentence count, numeric document ID, sentence-index tuple)` across annotated alternative evidence sets. It never consults Qwen outputs. Empty original evidence maps to the existing INSUFFICIENT label and receives `[NO EVIDENCE DOCUMENTS PROVIDED]`.

The generator remains `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`, BF16, grounded-v2 prompt SHA-256 `21c9360a72834e37fcc37936190e290638a924e8985c50bf6dcff99472ba50b4`, greedy decoding, and `max_new_tokens=192`.

## Feasibility and localization

The first five numeric DEV IDs were used for label-free feasibility. Repeated MedCPT scoring had maximum absolute logit difference **0.0**. Sentence preservation, original rationale text/index alignment, document association, score direction, and deterministic ordering all passed.

The run scored 4,358 sentences from 364 unique documents. The longest claim-passage pair was 229 tokens, so none reached MedCPT's 512-token limit.

Among the 98 claims with annotated rationale sentences:

| Diagnostic | Count | Rate |
|---|---:|---:|
| Exact annotated rationale sentence at rank 1 | 42/98 | 0.4286 |
| Exact annotated rationale sentence within rank 3 | 75/98 | 0.7653 |
| Selected sentence belongs to an annotated evidence document | 85/98 | 0.8673 |

MedCPT often selected the right document without selecting an annotated sentence: 43 claims fall into that category at rank 1. A non-annotated sentence is not automatically wrong or irrelevant; these rates measure agreement with the original SciFact rationale annotations.

## Gold-rationale diagnostic

| Condition | Accuracy | Macro F1 | SUPPORT acc/F1 | CONTRADICT acc/F1 | INSUFFICIENT acc/F1 | Invalid | Citation validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| G_full | 0.8765 | 0.8636 | 0.8000 / 0.8682 | 0.7857 / 0.8148 | 1.0000 / 0.9078 | 0.0000 | 1.0000 |
| G_rationale | 0.8457 | 0.8213 | 0.7429 / 0.8189 | 0.7500 / 0.7241 | 1.0000 / 0.9209 | 0.0000 | 1.0000 |

Gold rationale does **not** outperform the annotated full abstract. Accuracy changes by −0.0309 and macro F1 by −0.0423; five predictions improve, ten degrade, and 147 retain the same correctness.

For one claim, the frozen shortest-rationale rule selects a different annotated document from reused G_full. Excluding that claim leaves 161 aligned cases and an accuracy difference of −0.0248, with five improvements and nine degradations; the macro-F1 difference is −0.0372. The conclusion is unchanged.

Focused rationale can omit context needed to interpret even an annotated sentence. This diagnostic therefore does not support context dilution as the dominant remaining limitation. It remains consistent with verifier reasoning difficulty, incomplete single-sentence context, and task or annotation ambiguity contributing in different cases.

## Automatic passage verification

| Condition | Accuracy | Macro F1 | SUPPORT acc/F1 | CONTRADICT acc/F1 | INSUFFICIENT acc/F1 | Invalid | Citation validity | Abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 — BGE top-1 full abstract | 0.7654 | 0.7531 | 0.7857 / 0.8148 | 0.7857 / 0.6984 | 0.7344 / 0.7460 | 0.0000 | 1.0000 | 0.3827 |
| P2 — BGE top-3 → MedCPT top sentence | 0.6852 | 0.6647 | 0.6714 / 0.7402 | 0.6786 / 0.5507 | 0.7031 / 0.7031 | 0.0000 | 1.0000 | 0.3951 |

The mean evidence length falls from 534.2 to 99.3 Qwen tokens. All 162 P2 outputs are valid and cite only supplied document IDs.

Verification accuracy depends strongly on localization outcome:

| P2 evidence subset | Claims | Accuracy |
|---|---:|---:|
| Exact annotated rationale localized | 42 | 0.8571 |
| Annotated document selected, annotated sentence missed | 43 | 0.6047 |
| Annotated document not selected | 13 | 0.3077 |
| No exact annotated rationale localized, combined | 56 | 0.5357 |

These are descriptive subsets with different claims. They show that exact sentence localization is useful when achieved, while selecting only the correct document is insufficient for this one-sentence setup.

## Paired analysis

The predefined primary comparison is P2 minus P1 per-query accuracy. The mean difference is **−0.0802**: 10 claims improve, 129 retain the same correctness, and 23 degrade.

The frozen 112-group cluster bootstrap gives a 20,000-replicate 95% CI of **[−0.1529, −0.0063]** (seed 20261013). The two-sided 100,000-permutation cluster sign-flip p-value is **0.0587** with plus-one correction (seed 20261014). The bootstrap interval excludes zero while the randomization test narrowly exceeds 0.05; both are reported without selecting between them post hoc.

Macro F1 changes by **−0.0884**. Its exploratory cluster-bootstrap 95% CI is **[−0.1599, −0.0189]** (20,000 replicates; seed 20261015). No sign-flip test is assigned to non-decomposable macro F1.

## Failure analysis

P1 and P2 produce the same label on 124 claims and different labels on 38. Passage localization fixes 10 errors and creates 23 new errors; five changed predictions retain the same correctness. Sixteen changes end in abstention. Qwen misclassifies six claims even when the selected sentence exactly matches an annotated rationale.

Observed patterns include:

- Correct rationale fixes clear baseline errors, including increased DBP and abdominal aortic aneurysm (claim 607), reduced IME1 expression (815), and minus-strand cytidine deamination (1209).
- Single sentences omit conclusions or surrounding relations. For insomnia treatment (267), MedCPT selects a sentence saying treatments can be beneficial but omits the trial conclusion needed to contradict the claim.
- Qwen reverses causal direction or polarity despite an exact annotated sentence, including decreased DBP (317) and Gpr124 knockout causing BBB disruption (479/480).
- An annotated sentence can be insufficient in isolation. For the Eilat-virus claim (372), the selected sentence mentions mouse models while the claim refers to nonhuman primates.
- Some empty-annotation claims receive related passages that induce SUPPORT or CONTRADICT. These remain annotation-ambiguity cases rather than evidence of annotation error.

All changed cases and twelve conservative illustrative reviews are retained in the artifacts.

## Answers to the milestone questions

1. **Does gold rationale outperform gold full abstract?** No. G_rationale is lower by 0.0309 accuracy and 0.0423 macro F1; the same-document sensitivity reaches the same conclusion.
2. **Can automatic localization recover annotated rationales reliably?** Partially. MedCPT reaches 42.9% exact top-1 and 76.5% exact top-3 rationale recall within BGE's top three documents.
3. **Does passage evidence improve verification over the full-abstract baseline?** No. P2 is lower by 0.0802 accuracy and 0.0884 macro F1.

This one-sentence passage-selection configuration did not translate focused evidence into better verification under the frozen Qwen setup. It does not establish that passage localization or chunking is generally harmful. One sentence can remove necessary context, non-annotated sentences may still be useful, all analysis is on DEV, and G_rationale is a diagnostic rather than a strict upper bound.

## Validation and commands

All **30 tests passed**, covering deterministic sentence preservation, rationale mapping, passage/document alignment, score direction, frozen Qwen equality, metrics, and cluster resampling. Independent final validation reconstructs aggregates and verifies exact reuse of P1 and G_full. No TEST references occur in the milestone runners.

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_passage_localization_5c
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_passage_verification_5c
python -m scripts.evaluate_passage_localization_5c
python -m pytest -q
```

## Artifacts

- Repository results: `docs/milestone5c-passage-localization-dev-results.json`
- Frozen config: `configs/passage-localization-5c-v1.json`
- External run: `/home/boweiye2/rag/runs/milestone5c-passage-v1`
- Segmentation: `/home/boweiye2/rag/runs/milestone5c-passage-v1/sentence-segmentation.json`
- Rationale mapping: `/home/boweiye2/rag/runs/milestone5c-passage-v1/rationale-mapping.json`
- Candidate scores and passage rankings: `/home/boweiye2/rag/runs/milestone5c-passage-v1/passage-rankings.json`
- Qwen generations/evaluations: `/home/boweiye2/rag/runs/milestone5c-passage-v1/verification/`
- Paired statistics: `/home/boweiye2/rag/runs/milestone5c-passage-v1/verification-summary.json`
- Failure analysis: `/home/boweiye2/rag/runs/milestone5c-passage-v1/{failure-analysis.json,selected-failure-review.json}`
- Dependency freeze: `environments/passage-localization-5c-pip-freeze.txt`
- Logs: `/home/boweiye2/rag/logs/milestone5c-{localization,verification,evaluation}.log`
- Exact source/config/dependency snapshot: `/home/boweiye2/rag/runs/milestone5c-passage-v1/source/exact-source-config-dependencies.tar.gz`

Milestone 5C stops here.
