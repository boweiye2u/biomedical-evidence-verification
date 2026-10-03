"""Frozen, analysis-only Milestone 4C. Never loads test qrels or model weights."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from retrieval.statistics import (
    TOLERANCE,
    classify_differences,
    cluster_bootstrap_ci,
    cluster_sign_flip_test,
    holm_adjust,
    positive_rank,
    query_bootstrap_ci,
    query_sign_flip_test,
    validate_clusters,
)

ROOT = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("RAG_ROOT", str(Path.home() / "rag")))
OUTPUT = ART / "runs/milestone4c-analysis-v1"
SEEDS = ["20261002", "20261003", "20261004"]
METRICS = ["NDCG@10", "Recall@10", "Recall@100", "MRR@10"]
COMPARISONS = ["mined_minus_random", "random_minus_zero", "mined_minus_zero"]
BOOTSTRAP_SEED = 20261005
PERMUTATION_SEED = 20261006
QUERY_BOOTSTRAP_SEED = 20261007
QUERY_PERMUTATION_SEED = 20261008


def load_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_ranking(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [doc for doc, _ in sorted(value.items(), key=lambda item: -item[1])]
    raise TypeError(type(value))


def consistency(values, reference):
    differences = np.asarray(values) - reference
    if np.all(differences > TOLERANCE):
        return "all_3_improve"
    if np.all(differences < -TOLERANCE):
        return "all_3_degrade"
    if np.all(np.abs(differences) <= TOLERANCE):
        return "all_3_tie"
    return "mixed"


def selected_artifacts():
    pilot = load_json(ROOT / "docs/milestone4b-pilot-results.json")
    replication = load_json(ART / "runs/milestone4b-replication-v1/summary.json")
    runs = {"20261002": pilot, **replication["runs"]}
    result = {"random": {}, "mined": {}}
    for seed, payload in runs.items():
        for source_arm, target_arm in [("random", "random"), ("hard", "mined")]:
            arm = payload["arms"][source_arm]
            epoch = arm["selected_epoch"]
            if seed == "20261002":
                base = ART / "runs/milestone4b-pilot-v1" / source_arm
            else:
                base = ART / "runs/milestone4b-replication-v1" / f"seed-{seed}" / source_arm
            result[target_arm][seed] = {
                "epoch": epoch,
                "metrics": load_json(base / f"epoch-{epoch}-per-query.json"),
                "rankings": {
                    query: as_ranking(ranking)
                    for query, ranking in load_json(base / f"epoch-{epoch}-rankings.json").items()
                },
                "aggregate": next(item for item in arm["epochs"] if item["epoch"] == epoch),
            }
    return result


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty analysis directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    split_path = ROOT / "configs/splits/scifact_train_dev_v1.json"
    split = load_json(split_path)
    dev_ids = sorted(split["dev_ids"], key=int)
    assert len(dev_ids) == 162

    groups = []
    dev_set = set(dev_ids)
    group_map = {}
    for original_index, original_group in enumerate(split["groups"]):
        group = sorted(dev_set.intersection(original_group), key=int)
        if not group:
            continue
        group_id = f"group-{original_index:04d}"
        groups.append(group)
        for query_id in group:
            assert query_id not in group_map
            group_map[query_id] = group_id
    validate_clusters(dev_ids, groups)

    data = ART / "data/scifact"
    qrels_path = data / "beir/qrels/train.tsv"
    assert qrels_path.name == "train.tsv"
    assert not (data / "beir/qrels/test.tsv").exists()
    qrels = {query_id: [] for query_id in dev_ids}
    for row in csv.DictReader(qrels_path.open(), delimiter="\t"):
        if row["query-id"] in qrels and int(row["score"]) > 0:
            qrels[row["query-id"]].append(row["corpus-id"])
    assert all(qrels.values())
    qrels = {query: sorted(documents, key=int) for query, documents in qrels.items()}

    claims = {
        str(item["id"]): item
        for item in map(json.loads, (data / "original/claims_train.jsonl").read_text().splitlines())
    }
    corpus = {
        str(item["_id"]): item
        for item in map(json.loads, (data / "beir/corpus.jsonl").read_text().splitlines())
    }

    zero_metrics = load_json(ART / "runs/scifact_dev_v1/bge-per-query.json")
    zero_rankings = {
        query: as_ranking(ranking)
        for query, ranking in load_json(ART / "runs/scifact_dev_v1/bge-rankings.json").items()
    }
    selected = selected_artifacts()
    systems = [zero_metrics, zero_rankings]
    for arm in ["random", "mined"]:
        for seed in SEEDS:
            systems.extend([selected[arm][seed]["metrics"], selected[arm][seed]["rankings"]])
    assert all(set(system) == set(dev_ids) for system in systems)

    baseline_summary = load_json(ROOT / "docs/scifact-dev-baselines.json")
    expected_zero = baseline_summary["systems"]["bge"]["metrics"]
    for metric in METRICS:
        reconstructed = statistics.mean(zero_metrics[query][metric] for query in dev_ids)
        assert abs(reconstructed - expected_zero[metric]) < 1e-12
        for arm in ["random", "mined"]:
            for seed in SEEDS:
                reconstructed = statistics.mean(
                    selected[arm][seed]["metrics"][query][metric] for query in dev_ids
                )
                expected = selected[arm][seed]["aggregate"]["metrics"][metric]
                assert abs(reconstructed - expected) < 1e-12

    per_query = []
    difference_vectors = {comparison: {metric: [] for metric in METRICS} for comparison in COMPARISONS}
    consistency_counts = {arm: Counter() for arm in ["random", "mined"]}
    for query_id in dev_ids:
        record = {
            "query_id": query_id,
            "group_id": group_map[query_id],
            "claim": claims[query_id]["claim"],
            "known_beir_positive_ids": qrels[query_id],
            "metrics": {"zero_shot": zero_metrics[query_id], "random": {}, "mined": {}},
            "averages": {"random": {}, "mined": {}},
            "differences": {},
            "consistency": {},
            "positive_ranks": [],
        }
        for arm in ["random", "mined"]:
            for seed in SEEDS:
                record["metrics"][arm][seed] = selected[arm][seed]["metrics"][query_id]
            for metric in METRICS:
                record["averages"][arm][metric] = statistics.mean(
                    record["metrics"][arm][seed][metric] for seed in SEEDS
                )
        for metric in METRICS:
            differences = {
                "mined_minus_random": record["averages"]["mined"][metric] - record["averages"]["random"][metric],
                "random_minus_zero": record["averages"]["random"][metric] - zero_metrics[query_id][metric],
                "mined_minus_zero": record["averages"]["mined"][metric] - zero_metrics[query_id][metric],
            }
            record["differences"][metric] = differences
            for comparison, value in differences.items():
                difference_vectors[comparison][metric].append(value)
            for arm in ["random", "mined"]:
                key = f"{arm}_vs_zero_{metric}"
                category = consistency(
                    [record["metrics"][arm][seed][metric] for seed in SEEDS],
                    zero_metrics[query_id][metric],
                )
                record["consistency"][key] = category
                if metric == "NDCG@10":
                    consistency_counts[arm][category] += 1
        for positive_id in qrels[query_id]:
            rank_record = {
                "document_id": positive_id,
                "zero_shot": positive_rank(zero_rankings[query_id], positive_id),
                "random": {},
                "mined": {},
            }
            for arm in ["random", "mined"]:
                for seed in SEEDS:
                    rank_record[arm][seed] = positive_rank(
                        selected[arm][seed]["rankings"][query_id], positive_id
                    )
            record["positive_ranks"].append(rank_record)
        per_query.append(record)

    statistical_results = {}
    raw_exploratory_p = {}
    analysis_index = 0
    for comparison in COMPARISONS:
        statistical_results[comparison] = {}
        for metric in METRICS:
            values = np.asarray(difference_vectors[comparison][metric])
            cluster_seed = BOOTSTRAP_SEED + analysis_index
            permutation_seed = PERMUTATION_SEED + analysis_index
            query_bootstrap_seed = QUERY_BOOTSTRAP_SEED + analysis_index
            query_permutation_seed = QUERY_PERMUTATION_SEED + analysis_index
            wins = classify_differences(values)
            result = {
                "mean_difference": float(values.mean()),
                "median_difference": float(np.median(values)),
                "counts": wins,
                "proportions": {key: value / len(dev_ids) for key, value in wins.items()},
                "cluster_bootstrap_95ci": cluster_bootstrap_ci(
                    values, dev_ids, groups, replicates=20_000, seed=cluster_seed,
                ),
                "cluster_sign_flip": cluster_sign_flip_test(
                    values, dev_ids, groups, permutations=100_000, seed=permutation_seed,
                ),
                "query_sensitivity": {
                    "bootstrap_95ci": query_bootstrap_ci(
                        values, replicates=20_000, seed=query_bootstrap_seed,
                    ),
                    "sign_flip": query_sign_flip_test(
                        values, permutations=100_000, seed=query_permutation_seed,
                    ),
                },
                "role": "primary" if comparison == "mined_minus_random" and metric == "NDCG@10" else "exploratory",
            }
            statistical_results[comparison][metric] = result
            key = f"{comparison}|{metric}"
            if result["role"] == "exploratory":
                raw_exploratory_p[key] = result["cluster_sign_flip"]["p_value"]
            analysis_index += 1
    adjusted = holm_adjust(raw_exploratory_p)
    for key, value in adjusted.items():
        comparison, metric = key.split("|")
        statistical_results[comparison][metric]["holm_adjusted_exploratory_p"] = value

    def select_sample():
        chosen = []
        reasons = {}

        def add(records, reason, count):
            added = 0
            for record in records:
                query_id = record["query_id"]
                if query_id in reasons:
                    continue
                chosen.append(query_id)
                reasons[query_id] = reason
                added += 1
                if added == count:
                    break

        add(sorted(per_query, key=lambda r: r["differences"]["NDCG@10"]["mined_minus_random"]), "largest_mined_minus_random_ndcg_drop", 16)
        add(sorted(per_query, key=lambda r: r["differences"]["NDCG@10"]["mined_minus_zero"]), "largest_mined_minus_zero_ndcg_drop", 8)
        top10_losses = [
            record for record in per_query
            if any(
                isinstance(rank["zero_shot"], int) and rank["zero_shot"] <= 10
                and all(rank["mined"][seed] == ">100" or rank["mined"][seed] > 10 for seed in SEEDS)
                for rank in record["positive_ranks"]
            )
        ]
        add(top10_losses, "zero_top10_positive_lost_by_all_mined_seeds", 5)
        add(sorted(per_query, key=lambda r: -r["differences"]["NDCG@10"]["mined_minus_random"]), "largest_mined_minus_random_ndcg_improvement", 5)
        add(sorted(per_query, key=lambda r: -abs(r["differences"]["NDCG@10"]["random_minus_zero"])), "largest_absolute_random_minus_zero_change", 6)
        # A stratum can contain fewer unique queries than its requested quota.
        # Preserve all prior selections, then deterministically fill to the
        # frozen sample size from the largest remaining primary effects.
        add(
            sorted(
                per_query,
                key=lambda r: (-abs(r["differences"]["NDCG@10"]["mined_minus_random"]), int(r["query_id"])),
            ),
            "deterministic_primary_effect_fill",
            40 - len(chosen),
        )
        assert len(chosen) == 40
        return {"query_ids": chosen, "selection_reason": reasons, "count": len(chosen)}

    sample = select_sample()
    # This file is deliberately frozen before the later qualitative review stage.
    save_json(OUTPUT / "analysis-sample-ids.json", sample)

    by_id = {record["query_id"]: record for record in per_query}
    review_candidates = []
    for query_id in sample["query_ids"]:
        record = by_id[query_id]
        candidate = {
            "query_id": query_id,
            "selection_reason": sample["selection_reason"][query_id],
            "claim": record["claim"],
            "differences": record["differences"],
            "consistency": record["consistency"],
            "positive_ranks": record["positive_ranks"],
            "positive_documents": [
                {"document_id": doc, "title": corpus[doc]["title"], "text": corpus[doc]["text"]}
                for doc in qrels[query_id]
            ],
            "top_results": {"zero_shot": [], "random": {}, "mined": {}},
        }
        for doc in zero_rankings[query_id][:5]:
            candidate["top_results"]["zero_shot"].append({
                "document_id": doc, "title": corpus[doc]["title"], "text_excerpt": corpus[doc]["text"][:1600],
            })
        for arm in ["random", "mined"]:
            for seed in SEEDS:
                candidate["top_results"][arm][seed] = [
                    {"document_id": doc, "title": corpus[doc]["title"], "text_excerpt": corpus[doc]["text"][:1600]}
                    for doc in selected[arm][seed]["rankings"][query_id][:5]
                ]
        review_candidates.append(candidate)

    group_sizes = [len(group) for group in groups]
    summary = {
        "scope": "post-selection frozen DEV diagnostic; not unseen-test inference",
        "query_count": len(dev_ids),
        "corpus_count": len(corpus),
        "grouping": {
            "group_count": len(groups),
            "min_size": min(group_sizes),
            "median_size": statistics.median(group_sizes),
            "max_size": max(group_sizes),
            "size_distribution": dict(sorted(Counter(group_sizes).items())),
        },
        "seeds": SEEDS,
        "primary": {"comparison": "mined_minus_random", "metric": "NDCG@10"},
        "statistics": statistical_results,
        "ndcg_seed_consistency": {
            arm: {
                category: {"count": count, "percentage": count / len(dev_ids) * 100}
                for category, count in sorted(consistency_counts[arm].items())
            }
            for arm in ["random", "mined"]
        },
        "random_seeds": {
            "cluster_bootstrap_base": BOOTSTRAP_SEED,
            "cluster_permutation_base": PERMUTATION_SEED,
            "query_bootstrap_base": QUERY_BOOTSTRAP_SEED,
            "query_permutation_base": QUERY_PERMUTATION_SEED,
        },
        "frozen_inputs": {
            "split_sha256": sha256(split_path),
            "pilot_summary_sha256": sha256(ROOT / "docs/milestone4b-pilot-results.json"),
            "replication_summary_sha256": sha256(ART / "runs/milestone4b-replication-v1/summary.json"),
            "zero_per_query_sha256": sha256(ART / "runs/scifact_dev_v1/bge-per-query.json"),
            "qrels_path": str(qrels_path.relative_to(ART)),
            "test_qrels_loaded": False,
        },
        "sample_count": sample["count"],
    }
    save_json(OUTPUT / "statistical-summary.json", summary)
    save_json(OUTPUT / "per-query-analysis.json", per_query)
    save_json(OUTPUT / "review-candidates.json", review_candidates)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
