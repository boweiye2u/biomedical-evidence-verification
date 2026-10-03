# Milestone 4C — cluster-aware statistical and failure analysis

> **Scope:** These are post-selection diagnostics on the same 162 DEV claims used to select checkpoints. Confidence intervals and p-values characterize the selected systems on this frozen DEV set; they are not confirmatory estimates of unseen-test generalization.

No model was trained and no TEST qrels were read. Retrieval relevance remains the BEIR cited-document target, which is distinct from verification evidence.

## Frozen data and validation

The analysis reconstructed every aggregate from exactly 162 aligned per-query records over the full 5,183-document corpus within 1e-12. The frozen grouping contains 112 groups with size distribution 1×69, 2×39, 3×2, 4×1, and 5×1 (min/median/max = 1/1/5). All 16 tests passed.

## Predefined primary result

For each query, the three selected checkpoints were averaged within each arm. The paired effect is mined average minus random average on NDCG@10. Groups, rather than individual queries, were resampled or sign-flipped.

- Mean difference: **-0.0346**
- Median difference: **0.0000**
- Cluster bootstrap 95% CI: **[-0.0739, 0.0055]** (20,000 replicates; seed 20261005)
- Two-sided cluster sign-flip p: **0.0903** (100,000 permutations; seed 20261006; plus-one correction)
- Query outcomes: **20 improved / 101 unchanged / 41 degraded**

The group-aware interval crosses zero and the predefined two-sided test does not reach 0.05. The observed mean is nevertheless negative and degradation occurs in more queries than improvement. Ordinary query-level sensitivity gives a 95% CI of [-0.0689, -0.0005] and p=0.0505; this difference shows why related-claim clustering matters.

## Supporting and exploratory results

All entries below except the predefined primary are exploratory. Raw cluster sign-flip p-values are accompanied by Holm adjustment over the 11 exploratory comparisons.

| Comparison and metric | Mean Δ | Group-bootstrap 95% CI | Raw p | Holm p |
|---|---:|---:|---:|---:|
| Mined − random, NDCG@10 | -0.0346 | [-0.0739, 0.0055] | 0.0903 | — |
| Mined − random, Recall@10 | -0.0381 | [-0.0768, -0.0021] | 0.0460 | 0.4144 |
| Mined − random, Recall@100 | -0.0473 | [-0.1114, -0.0073] | 0.0152 | 0.1523 |
| Mined − random, MRR@10 | -0.0278 | [-0.0759, 0.0210] | 0.2627 | 1.0000 |
| Random − zero, NDCG@10 | 0.0013 | [-0.0231, 0.0265] | 0.9201 | 1.0000 |
| Random − zero, Recall@10 | -0.0224 | [-0.0667, 0.0118] | 0.3847 | 1.0000 |
| Random − zero, Recall@100 | 0.0062 | [-0.0042, 0.0218] | 0.7490 | 1.0000 |
| Random − zero, MRR@10 | 0.0062 | [-0.0207, 0.0344] | 0.6655 | 1.0000 |
| Mined − zero, NDCG@10 | -0.0333 | [-0.0705, 0.0048] | 0.0884 | 0.7070 |
| Mined − zero, Recall@10 | -0.0606 | [-0.1116, -0.0175] | 0.0084 | 0.0929 |
| Mined − zero, Recall@100 | -0.0412 | [-0.1059, 0.0040] | 0.1557 | 1.0000 |
| Mined − zero, MRR@10 | -0.0216 | [-0.0645, 0.0234] | 0.3403 | 1.0000 |

No exploratory comparison has Holm-adjusted p < 0.05. Random minus zero NDCG@10 is +0.0013 with group-aware CI [−0.0231, 0.0265]. Mined minus zero NDCG@10 is −0.0333 with CI [−0.0705, 0.0048].

## Query and positive-rank diagnostics

NDCG@10 consistency relative to zero-shot:

| Arm | All 3 improve | All 3 tie | All 3 degrade | Mixed |
|---|---:|---:|---:|---:|
| Random | 18 | 117 | 16 | 11 |
| Mined | 14 | 101 | 31 | 16 |

Across corresponding seed checkpoints, mined versus random is 13 all-seed improvements, 100 all-seed ties, 24 all-seed degradations, and 25 mixed queries. This pairing is descriptive because the selected epoch can differ by arm.

All 182 query-positive pairs are retained separately. Across seed-specific positive instances, zero-shot has 3 positives absent from saved top 100 and 161 in the top 10; random has 6/546 absent and 470/546 in the top 10; mined has 31/546 absent and 446/546 in the top 10. Missing ranks are recorded only as `>100`.

## Existing training and DEV trajectories

| Arm | Epoch | Mean loss | Mean pre-clip grad norm | Mean DEV NDCG@10 |
|---|---:|---:|---:|---:|
| Random | 1 | 0.083598 | 2.3018 | 0.7568 |
| Random | 2 | 0.007541 | 0.3750 | 0.7581 |
| Random | 3 | 0.003437 | 0.1634 | 0.7591 |
| Mined | 1 | 1.145634 | 9.2580 | 0.7271 |
| Mined | 2 | 0.578349 | 9.3691 | 0.6989 |
| Mined | 3 | 0.340860 | 8.6436 | 0.6781 |

Random negatives became easy quickly: mean loss fell from 0.0836 to 0.0034 and gradient norms from 2.30 to 0.16, while mean DEV NDCG stayed near 0.76. For the mined arm, mean loss fell from 1.1456 to 0.3409 while mean DEV NDCG fell from 0.7271 to 0.6781; pre-clipping gradients remained about 8.6–9.4. This is divergence between the improving training objective and worsening DEV retrieval. It does not identify overfitting, false-negative noise, optimization failure, or clipping as the cause.

## Structured analyst-assisted review

The 40 query IDs were written before categories were assigned. Selection deliberately enriched large changes and contrasts, so the 23 degradation, 11 improvement, and 6 near-tie cases are not prevalence estimates. Only three unique queries met the all-seed top-10-loss stratum; two remaining slots were filled deterministically from the largest absolute primary effects.

The largest descriptive groups were mechanism/relation specificity (8), a visible semantic gap between the claim and designated BEIR positive (8), lexical/entity collision (6), polarity/relation insensitivity (5), and exact-positive promotion (5). Every entry includes its claim, positive ranks, category, and conservative note. `benchmark_positive_semantic_gap` signals an interpretation limitation; it does not assert that an unjudged document is relevant or that the benchmark is wrong. This was analyst-assisted review, not independent blinded human annotation.

## Interpretation and limits

On the frozen, checkpoint-selecting DEV set, random-negative adaptation approximately preserved zero-shot NDCG, while this specific zero-shot-BGE mining policy produced a consistent and larger degradation across seeds. Cluster-aware paired analysis measures how broadly that difference appears across DEV query groups, but held-out TEST evaluation is still required for confirmatory generalization.

The experiment compares the frozen random-sampling policy with one zero-shot-BGE mining policy under one loss and optimization setup. It does not establish that hard negatives generally help or hurt. The current analysis cannot attribute degradation to false negatives or any other mechanism.

## Artifacts

- External per-query analysis: `~/rag/runs/milestone4c-analysis-v1/per-query-analysis.json`
- Frozen sample IDs: `~/rag/runs/milestone4c-analysis-v1/analysis-sample-ids.json`
- Review candidates and structured notes: `~/rag/runs/milestone4c-analysis-v1/{review-candidates.json,structured-analyst-assisted-review.json}`
- Full statistical output: `~/rag/runs/milestone4c-analysis-v1/statistical-summary.json`
- Log: `~/rag/logs/milestone4c-analysis.log`
- Exact source/config/dependency snapshot: `~/rag/runs/milestone4c-analysis-v1/source/exact-source-config-dependencies.tar.gz`

Milestone 4C stops here. No TEST evaluation, training, ablation, reranking, LLM, RAG, or serving work was run.
