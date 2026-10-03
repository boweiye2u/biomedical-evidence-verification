"""Assemble the Milestone 4C repository summary from frozen analysis artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("RAG_ROOT", str(Path.home() / "rag")))
RUN = ART / "runs/milestone4c-analysis-v1"
SEEDS = ["20261002", "20261003", "20261004"]
TOLERANCE = 1e-12


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(values):
    if all(x > TOLERANCE for x in values): return "all_3_mined_improve"
    if all(x < -TOLERANCE for x in values): return "all_3_mined_degrade"
    if all(abs(x) <= TOLERANCE for x in values): return "all_3_tie"
    return "mixed"


def epoch_record(seed, arm, epoch):
    source_arm = "hard" if arm == "mined" else arm
    if seed == "20261002":
        path = ART / "runs/milestone4b-pilot-v1" / source_arm / f"epoch-{epoch}.json"
    else:
        path = ART / "runs/milestone4b-replication-v1" / f"seed-{seed}" / source_arm / f"epoch-{epoch}.json"
    return load(path)


def main():
    stats = load(RUN / "statistical-summary.json")
    per_query = load(RUN / "per-query-analysis.json")
    review = load(RUN / "structured-analyst-assisted-review.json")
    sample = load(RUN / "analysis-sample-ids.json")

    paired = Counter()
    for query in per_query:
        paired[classify([
            query["metrics"]["mined"][seed]["NDCG@10"] - query["metrics"]["random"][seed]["NDCG@10"]
            for seed in SEEDS
        ])] += 1

    rank_summary = {}
    for system in ["zero_shot", "random", "mined"]:
        ranks = []
        missing = 0
        if system == "zero_shot": iterator = [(query, None) for query in per_query]
        else: iterator = [(query, seed) for query in per_query for seed in SEEDS]
        query_visibility = Counter()
        for query, seed in iterator:
            values = [item[system] if seed is None else item[system][seed] for item in query["positive_ranks"]]
            finite = [value for value in values if isinstance(value, int)]
            missing += sum(value == ">100" for value in values)
            ranks.extend(value if isinstance(value, int) else 101 for value in values)
            query_visibility["any_positive_top10" if any(value <= 10 for value in finite) else "no_positive_top10"] += 1
            query_visibility["any_positive_top100" if finite else "no_positive_top100"] += 1
        rank_summary[system] = {
            "positive_seed_instances": len(ranks),
            "median_rank_capped_at_101": statistics.median(ranks),
            "absent_from_saved_top100": missing,
            "positive_seed_instances_in_top10": sum(value <= 10 for value in ranks),
            "query_seed_visibility": dict(query_visibility),
        }

    trace = {arm: {} for arm in ["random", "mined"]}
    for arm in trace:
        for epoch in [1, 2, 3]:
            records = [epoch_record(seed, arm, epoch) for seed in SEEDS]
            trace[arm][str(epoch)] = {
                "mean_train_loss": statistics.mean(x["train_loss"] for x in records),
                "seed_train_losses": [x["train_loss"] for x in records],
                "mean_preclip_gradient_norm": statistics.mean(x["gradient_norm_mean"] for x in records),
                "seed_preclip_gradient_norms": [x["gradient_norm_mean"] for x in records],
                "mean_dev_ndcg_at_10": statistics.mean(x["metrics"]["NDCG@10"] for x in records),
                "seed_dev_ndcg_at_10": [x["metrics"]["NDCG@10"] for x in records],
            }

    summary = {
        "scope_warning": "Post-selection diagnostics on the same frozen DEV set used for checkpoint selection; not confirmatory evidence of unseen-test generalization.",
        "retrieval_target": "BEIR cited-document positives, not verification evidence",
        "query_count": stats["query_count"],
        "corpus_count": stats["corpus_count"],
        "grouping": stats["grouping"],
        "primary": stats["statistics"]["mined_minus_random"]["NDCG@10"],
        "all_statistics": stats["statistics"],
        "seed_consistency_vs_zero": stats["ndcg_seed_consistency"],
        "corresponding_seed_mined_vs_random_ndcg_consistency": dict(paired),
        "positive_rank_summary": rank_summary,
        "training_and_dev_trajectory": trace,
        "structured_review": {
            "sample_count": len(sample["query_ids"]),
            "outcome_counts": review["outcome_counts"],
            "category_counts": review["category_counts"],
            "not_prevalence_estimates": True,
            "reviewer": review["reviewer"],
        },
        "validation": {
            "tests_passed": 16,
            "exact_query_alignment": True,
            "exact_group_alignment": True,
            "aggregate_reconstruction_tolerance": 1e-12,
            "test_qrels_loaded": False,
            "cluster_bootstrap_replicates": 20000,
            "cluster_sign_flip_permutations": 100000,
            "plus_one_correction": True,
            "holm_family_size": 11,
        },
        "sampling_note": "Three unique queries met the all-seed top-10-loss stratum; the remaining two positions needed for the frozen 40-query sample were filled deterministically by absolute primary effect after all other strata.",
        "artifacts": {
            "external_run": str(RUN),
            "per_query": str(RUN / "per-query-analysis.json"),
            "sample_ids": str(RUN / "analysis-sample-ids.json"),
            "review": str(RUN / "structured-analyst-assisted-review.json"),
            "statistical_summary": str(RUN / "statistical-summary.json"),
            "source_config_dependency_snapshot": str(RUN / "source/exact-source-config-dependencies.tar.gz"),
        },
        "source_hashes": {
            "analysis_script": sha(ROOT / "scripts/analyze_milestone4c.py"),
            "review_script": sha(ROOT / "scripts/build_milestone4c_review.py"),
            "statistics_module": sha(ROOT / "retrieval/statistics.py"),
        },
    }

    json_path = ROOT / "docs/milestone4c-statistical-failure-analysis.json"
    md_path = ROOT / "docs/milestone4c-statistical-failure-analysis.md"
    for path in [json_path, md_path]:
        if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    def row(name, result):
        ci=result["cluster_bootstrap_95ci"]; p=result["cluster_sign_flip"]["p_value"]
        adj=result.get("holm_adjusted_exploratory_p")
        return f"| {name} | {result['mean_difference']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {p:.4f} | {'—' if adj is None else f'{adj:.4f}'} |"

    lines = [
        "# Milestone 4C — cluster-aware statistical and failure analysis",
        "",
        "> **Scope:** These are post-selection diagnostics on the same 162 DEV claims used to select checkpoints. Confidence intervals and p-values characterize the selected systems on this frozen DEV set; they are not confirmatory estimates of unseen-test generalization.",
        "",
        "No model was trained and no TEST qrels were read. Retrieval relevance remains the BEIR cited-document target, which is distinct from verification evidence.",
        "",
        "## Frozen data and validation",
        "",
        f"The analysis reconstructed every aggregate from exactly {stats['query_count']} aligned per-query records over the full {stats['corpus_count']:,}-document corpus within 1e-12. The frozen grouping contains {stats['grouping']['group_count']} groups with size distribution 1×69, 2×39, 3×2, 4×1, and 5×1 (min/median/max = 1/1/5). All 16 tests passed.",
        "",
        "## Predefined primary result",
        "",
        "For each query, the three selected checkpoints were averaged within each arm. The paired effect is mined average minus random average on NDCG@10. Groups, rather than individual queries, were resampled or sign-flipped.",
        "",
        f"- Mean difference: **{summary['primary']['mean_difference']:.4f}**",
        f"- Median difference: **{summary['primary']['median_difference']:.4f}**",
        f"- Cluster bootstrap 95% CI: **[{summary['primary']['cluster_bootstrap_95ci']['lower']:.4f}, {summary['primary']['cluster_bootstrap_95ci']['upper']:.4f}]** (20,000 replicates; seed {summary['primary']['cluster_bootstrap_95ci']['seed']})",
        f"- Two-sided cluster sign-flip p: **{summary['primary']['cluster_sign_flip']['p_value']:.4f}** (100,000 permutations; seed {summary['primary']['cluster_sign_flip']['seed']}; plus-one correction)",
        f"- Query outcomes: **{summary['primary']['counts']['improved']} improved / {summary['primary']['counts']['unchanged']} unchanged / {summary['primary']['counts']['degraded']} degraded**",
        "",
        "The group-aware interval crosses zero and the predefined two-sided test does not reach 0.05. The observed mean is nevertheless negative and degradation occurs in more queries than improvement. Ordinary query-level sensitivity gives a 95% CI of "
        f"[{summary['primary']['query_sensitivity']['bootstrap_95ci']['lower']:.4f}, {summary['primary']['query_sensitivity']['bootstrap_95ci']['upper']:.4f}] and p={summary['primary']['query_sensitivity']['sign_flip']['p_value']:.4f}; this difference shows why related-claim clustering matters.",
        "",
        "## Supporting and exploratory results",
        "",
        "All entries below except the predefined primary are exploratory. Raw cluster sign-flip p-values are accompanied by Holm adjustment over the 11 exploratory comparisons.",
        "",
        "| Comparison and metric | Mean Δ | Group-bootstrap 95% CI | Raw p | Holm p |",
        "|---|---:|---:|---:|---:|",
    ]
    labels={"mined_minus_random":"Mined − random","random_minus_zero":"Random − zero","mined_minus_zero":"Mined − zero"}
    for comp in ["mined_minus_random","random_minus_zero","mined_minus_zero"]:
        for metric in ["NDCG@10","Recall@10","Recall@100","MRR@10"]:
            lines.append(row(f"{labels[comp]}, {metric}", stats["statistics"][comp][metric]))
    lines += [
        "",
        "No exploratory comparison has Holm-adjusted p < 0.05. Random minus zero NDCG@10 is +0.0013 with group-aware CI [−0.0231, 0.0265]. Mined minus zero NDCG@10 is −0.0333 with CI [−0.0705, 0.0048].",
        "",
        "## Query and positive-rank diagnostics",
        "",
        "NDCG@10 consistency relative to zero-shot:",
        "",
        "| Arm | All 3 improve | All 3 tie | All 3 degrade | Mixed |",
        "|---|---:|---:|---:|---:|",
        f"| Random | 18 | 117 | 16 | 11 |",
        f"| Mined | 14 | 101 | 31 | 16 |",
        "",
        "Across corresponding seed checkpoints, mined versus random is 13 all-seed improvements, 100 all-seed ties, 24 all-seed degradations, and 25 mixed queries. This pairing is descriptive because the selected epoch can differ by arm.",
        "",
        "All 182 query-positive pairs are retained separately. Across seed-specific positive instances, zero-shot has 3 positives absent from saved top 100 and 161 in the top 10; random has 6/546 absent and 470/546 in the top 10; mined has 31/546 absent and 446/546 in the top 10. Missing ranks are recorded only as `>100`.",
        "",
        "## Existing training and DEV trajectories",
        "",
        "| Arm | Epoch | Mean loss | Mean pre-clip grad norm | Mean DEV NDCG@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ["random","mined"]:
        for epoch in ["1","2","3"]:
            x=trace[arm][epoch]
            lines.append(f"| {arm.title()} | {epoch} | {x['mean_train_loss']:.6f} | {x['mean_preclip_gradient_norm']:.4f} | {x['mean_dev_ndcg_at_10']:.4f} |")
    lines += [
        "",
        "Random negatives became easy quickly: mean loss fell from 0.0836 to 0.0034 and gradient norms from 2.30 to 0.16, while mean DEV NDCG stayed near 0.76. For the mined arm, mean loss fell from 1.1456 to 0.3409 while mean DEV NDCG fell from 0.7271 to 0.6781; pre-clipping gradients remained about 8.6–9.4. This is divergence between the improving training objective and worsening DEV retrieval. It does not identify overfitting, false-negative noise, optimization failure, or clipping as the cause.",
        "",
        "## Structured analyst-assisted review",
        "",
        "The 40 query IDs were written before categories were assigned. Selection deliberately enriched large changes and contrasts, so the 23 degradation, 11 improvement, and 6 near-tie cases are not prevalence estimates. Only three unique queries met the all-seed top-10-loss stratum; two remaining slots were filled deterministically from the largest absolute primary effects.",
        "",
        "The largest descriptive groups were mechanism/relation specificity (8), a visible semantic gap between the claim and designated BEIR positive (8), lexical/entity collision (6), polarity/relation insensitivity (5), and exact-positive promotion (5). Every entry includes its claim, positive ranks, category, and conservative note. `benchmark_positive_semantic_gap` signals an interpretation limitation; it does not assert that an unjudged document is relevant or that the benchmark is wrong. This was analyst-assisted review, not independent blinded human annotation.",
        "",
        "## Interpretation and limits",
        "",
        "On the frozen, checkpoint-selecting DEV set, random-negative adaptation approximately preserved zero-shot NDCG, while this specific zero-shot-BGE mining policy produced a consistent and larger degradation across seeds. Cluster-aware paired analysis measures how broadly that difference appears across DEV query groups, but held-out TEST evaluation is still required for confirmatory generalization.",
        "",
        "The experiment compares the frozen random-sampling policy with one zero-shot-BGE mining policy under one loss and optimization setup. It does not establish that hard negatives generally help or hurt. The current analysis cannot attribute degradation to false negatives or any other mechanism.",
        "",
        "## Artifacts",
        "",
        "- External per-query analysis: `~/rag/runs/milestone4c-analysis-v1/per-query-analysis.json`",
        "- Frozen sample IDs: `~/rag/runs/milestone4c-analysis-v1/analysis-sample-ids.json`",
        "- Review candidates and structured notes: `~/rag/runs/milestone4c-analysis-v1/{review-candidates.json,structured-analyst-assisted-review.json}`",
        "- Full statistical output: `~/rag/runs/milestone4c-analysis-v1/statistical-summary.json`",
        "- Log: `~/rag/logs/milestone4c-analysis.log`",
        "- Exact source/config/dependency snapshot: `~/rag/runs/milestone4c-analysis-v1/source/exact-source-config-dependencies.tar.gz`",
        "",
        "Milestone 4C stops here. No TEST evaluation, training, ablation, reranking, LLM, RAG, or serving work was run.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
