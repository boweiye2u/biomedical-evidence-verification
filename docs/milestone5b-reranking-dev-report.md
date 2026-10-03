# Milestone 5B — reranking and evidence selection on DEV

> **Scope:** This is post-development analysis on the same frozen 162-claim DEV set used in prior milestones. It does not estimate unseen-TEST generalization.

No TEST qrels or verification labels were read. No retriever, reranker, or verifier was trained; no prompt, decoding, serving, or latency optimization was performed.

## Frozen systems and reranker

R1 reuses the frozen zero-shot `BAAI/bge-base-en-v1.5` ranking and the bitwise-identical Milestone 5A D1 outputs. R2 reranks only BGE's top 10 and supplies its selected full abstract to the unchanged Qwen verifier. R3 is the permitted retrieval-only top-20 diagnostic. R_gold reuses Milestone 5A D3.

The single reranker is **`ncbi/MedCPT-Cross-Encoder`**, revision `71caf65d4927987813984f54c284405a13fcca49`, loaded locally in FP32 from `/home/boweiye2/rag/models/MedCPT-Cross-Encoder`. MedCPT was selected because it is a biomedical cross-encoder already consistent with this project's model family. Each input is `[claim, title + abstract]`; paired inputs are truncated to 512 tokens. The model produces one raw relevance logit, and higher scores rank first.

The verifier remains `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`, BF16, grounded-v2 prompt SHA-256 `21c9360a72834e37fcc37936190e290638a924e8985c50bf6dcff99472ba50b4`, top-1 full abstract, greedy decoding, and `max_new_tokens=192`.

## Feasibility and validation

The first five numeric DEV IDs were selected without consulting labels. Repeated scoring produced maximum absolute logit difference **0.0**. Candidate identities were preserved, higher logits sorted first, claim/document alignment was checked, and tie-breaking is deterministic by original BGE rank. Full DEV scoring covered 3,240 pairs in 12.2 seconds. Of those, 205 exceeded 512 tokens before paired truncation; the maximum was 1,866 tokens.

An initial diagnostic mistakenly measured the outer two-element pair length rather than tokenized pair length. Scoring itself always used the documented paired tokenizer call and was unaffected. The length diagnostics were corrected and independently checked against single-pair tokenization.

All **27 tests passed**, including candidate preservation, score direction, deterministic ranking, pair alignment, exact frozen-generator validation, metric helpers, and reproducible cluster bootstrap. All 162 outputs satisfy the strict JSON and citation checks. Retrieval aggregates reconstruct from per-query records, and no TEST files were loaded.

## Retrieval and evidence selection

Metrics use the BEIR cited-document relevance target over the full 5,183-document corpus.

| Condition | NDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---:|---:|---:|
| R1 — BGE top-1 baseline | 0.7604 | 0.8938 | 0.9815 | 0.7279 |
| R2 — BGE top-10 → MedCPT | 0.7872 | 0.8938 | 0.9815 | 0.7593 |
| R3 — BGE top-20 → MedCPT | 0.7866 | 0.8928 | 0.9815 | 0.7574 |

R2 improves NDCG@10 by 0.0268 and MRR@10 by 0.0314. Recall@10 and Recall@100 are unchanged because R2 only reorders the original top 10.

Original SciFact annotated verification evidence behaves differently:

| Condition | Annotated evidence top-1 | Annotated evidence top-3 |
|---|---:|---:|
| R1 | 0.8571 | 0.9490 |
| R2 | 0.8367 | 0.9694 |
| R3 | 0.8265 | 0.9796 |

For R2, rank 1 stayed unchanged on 118/162 claims. MedCPT promoted annotated evidence to rank 1 on 5 claims and demoted it on 7. It promoted a BEIR known positive on 13 claims and demoted one on 9. Thus the reranker improved the cited-document ranking target while slightly reducing verification-evidence top-1 coverage. The rise in top-3 coverage did not help the verifier because the frozen condition supplies only top 1.

## Downstream verification

| Condition | Accuracy | Macro F1 | SUPPORT acc/F1 | CONTRADICT acc/F1 | INSUFFICIENT acc/F1 | Invalid | Citation validity | Abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 0.7654 | 0.7531 | 0.7857 / 0.8148 | 0.7857 / 0.6984 | 0.7344 / 0.7460 | 0.0000 | 1.0000 | 0.3827 |
| R2 | 0.7407 | 0.7309 | 0.7571 / 0.7852 | 0.7857 / 0.6875 | 0.7031 / 0.7200 | 0.0000 | 1.0000 | 0.3765 |
| R_gold | 0.8765 | 0.8636 | 0.8000 / 0.8682 | 0.7857 / 0.8148 | 1.0000 / 0.9078 | 0.0000 | 1.0000 | 0.4753 |

Among the 98 SUPPORT/CONTRADICT claims with annotated evidence, R1 coverage is 84/98 and R2 coverage is 82/98. Accuracy when annotated evidence is selected is 0.8333 for R1 and 0.8537 for R2; when it is absent, accuracy is 0.5000 and 0.3125 respectively. These conditional values are descriptive and use different query subsets.

## Paired analysis

The primary inferential metric was frozen before scoring as **per-query accuracy**, because it has an exact paired query contribution. Macro F1 remains the principal aggregate class-balanced metric but is non-decomposable, so its cluster bootstrap is exploratory and no sign-flip p-value is assigned to it.

For R2 minus R1 accuracy, the mean difference is **−0.0247**: 4 queries improve, 150 are unchanged, and 8 degrade. Resampling the frozen 112 related-claim groups gives a 20,000-replicate 95% CI of **[−0.0702, 0.0138]** (seed 20261010). The two-sided 100,000-permutation cluster sign-flip p-value is **0.3984** with plus-one correction (seed 20261011).

Macro F1 changes by **−0.0222**. Its exploratory 20,000-replicate cluster-bootstrap 95% CI is **[−0.0651, 0.0162]** (seed 20261012). Both intervals cross zero. These results do not establish a reliable improvement or degradation beyond this frozen DEV sample.

## Failure analysis

Reranking changes the selected document on 44 claims. Prediction and correctness transitions are:

| Transition | Count |
|---|---:|
| Top-1 unchanged | 118 |
| Top-1 changed, prediction unchanged | 31 |
| Changed context caused another new error | 6 |
| Annotated evidence demoted and a new error appeared | 2 |
| Changed context fixed a prediction by another route | 4 |
| Annotated evidence promoted but model remained wrong | 1 |

Two especially direct cases are claims 157 and 780: BGE selected annotated evidence and Qwen correctly predicted SUPPORT; MedCPT demoted that evidence and Qwen abstained. Claim 1261 shows the other bottleneck: MedCPT promoted the annotated document, but Qwen predicted SUPPORT for a CONTRADICT claim, consistent with a polarity or relation error. Four new errors occur on claims with empty original evidence annotations; these are retained as annotation-ambiguity cases and do not establish annotation error. Every changed top-1 case and a conservative selected-case review are saved in the run artifacts.

## Interpretation and limits

Under this fixed setup, MedCPT cross-encoder reranking improved BEIR cited-document ranking but did not improve single-abstract evidence selection or downstream verification. Annotated-evidence top-1 coverage and both verification aggregates decreased. The paired uncertainty intervals include zero, so the result should be described as a negative DEV result for this specific reranker, candidate depth, truncation rule, and frozen Qwen configuration.

This does not show that reranking is generally harmful, that retrieval is the only bottleneck, or that R_gold is a strict upper bound. Prompt and system choices were developed on DEV, 205 reranker pairs were truncated at 512 tokens, and TEST remains necessary for confirmatory generalization.

## Commands

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_reranking_5b
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_reranking_verification_5b
python -m scripts.evaluate_reranking_5b
python -m pytest -q
```

## Artifacts

- Repository results: `docs/milestone5b-reranking-dev-results.json`
- Frozen config: `configs/reranking-5b-v1.json`
- External run: `/home/boweiye2/rag/runs/milestone5b-reranking-v1`
- Candidate scores/order: `/home/boweiye2/rag/runs/milestone5b-reranking-v1/per-query-reranking.json`
- Rankings and retrieval diagnostics: `/home/boweiye2/rag/runs/milestone5b-reranking-v1/{rankings.json,retrieval-per-query.json,reranking-summary.json}`
- R2 generations/evaluation: `/home/boweiye2/rag/runs/milestone5b-reranking-v1/verification/R2/`
- Paired analysis: `/home/boweiye2/rag/runs/milestone5b-reranking-v1/verification-summary.json`
- Failure analysis: `/home/boweiye2/rag/runs/milestone5b-reranking-v1/{failure-analysis.json,selected-failure-review.json}`
- Dependency freeze: `environments/reranking-5b-pip-freeze.txt`
- Logs: `/home/boweiye2/rag/logs/{milestone5b-reranking.log,milestone5b-verification.log}`
- Exact source/config/dependency snapshot: `/home/boweiye2/rag/runs/milestone5b-reranking-v1/source/exact-source-config-dependencies.tar.gz`

Milestone 5B stops here.
