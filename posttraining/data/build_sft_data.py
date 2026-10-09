"""Construct and audit frozen TRAIN-only SciFact SFT data for Milestone 2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

from retrieval.bge import BGE

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"SUPPORT", "CONTRADICT", "INSUFFICIENT"}
PAIR_LABELS = {"SUPPORT", "CONTRADICT"}
AUDIT_LABELS = LABELS
AUDIT_FIELDS = [
    "claim_id", "doc_id", "claim", "evidence", "retrieval_rank",
    "retrieval_score", "human_pair_label", "notes", "filter_decision",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def claim_gold_label(claim: dict) -> str:
    labels = {
        evidence["label"]
        for evidence_sets in claim.get("evidence", {}).values()
        for evidence in evidence_sets
    }
    if not labels:
        return "INSUFFICIENT"
    if len(labels) > 1:
        return "MIXED"
    label = next(iter(labels))
    if label not in PAIR_LABELS:
        raise ValueError(f"Unexpected annotation label: {label}")
    return label


def select_rationale(evidence_sets: list[dict]) -> tuple[str, list[int], list[list[int]]]:
    labels = {item["label"] for item in evidence_sets}
    if len(labels) != 1 or not labels <= PAIR_LABELS:
        raise ValueError("A claim-document pair has inconsistent annotations")
    alternatives = sorted({tuple(item["sentences"]) for item in evidence_sets})
    if not alternatives:
        raise ValueError("Annotated pair has no rationale set")
    chosen = min(alternatives, key=lambda value: (len(value), value))
    return next(iter(labels)), list(chosen), [list(value) for value in alternatives]


def evidence_payload(document: dict) -> dict:
    return {
        "title": document["title"],
        "abstract": document["abstract"],
        "structured": document.get("structured"),
    }


def gold_rows(claims: dict[str, dict], corpus: dict[str, dict], train_ids: list[str]) -> list[dict]:
    rows = []
    for claim_id in train_ids:
        claim = claims[claim_id]
        gold = claim_gold_label(claim)
        for doc_id, evidence_sets in sorted(claim.get("evidence", {}).items(), key=lambda item: int(item[0])):
            annotation, rationale, alternatives = select_rationale(evidence_sets)
            document = corpus[str(doc_id)]
            if not rationale or any(index < 0 or index >= len(document["abstract"]) for index in rationale):
                raise ValueError(f"Invalid rationale indices for {claim_id}/{doc_id}")
            rows.append({
                "claim_id": claim_id,
                "doc_id": str(doc_id),
                "condition": "A_gold",
                "source_split": "TRAIN",
                "claim": claim["claim"],
                "evidence": evidence_payload(document),
                "claim_gold_label": gold,
                "pair_annotation": annotation,
                "context_label": annotation,
                "rationale_sentences": rationale,
                "alternative_rationale_sets": alternatives,
                "annotation_basis": "original_scifact_exact_pair_annotation",
                "retrieval_rank": None,
                "retrieval_score": None,
                "filter_version": None,
            })
    return rows


def candidate_row(claim_id: str, claim: dict, document: dict, rank: int, score: float, pool: str) -> dict:
    return {
        "claim_id": claim_id,
        "doc_id": str(document["doc_id"]),
        "condition": "B_retrieved_non_gold" if pool == "B" else "C_hard_negative",
        "source_split": "TRAIN",
        "claim": claim["claim"],
        "evidence": evidence_payload(document),
        "claim_gold_label": claim_gold_label(claim),
        "pair_annotation": None,
        "context_label": None,
        "rationale_sentences": [],
        "annotation_basis": "pending_independent_pool_audit",
        "retrieval_rank": rank,
        "retrieval_score": score,
        "candidate_pool": pool,
        "filter_status": "pending",
        "audit_status": "not_sampled",
        "filter_version": None,
    }


def build_candidates(
    claims: dict[str, dict], corpus: dict[str, dict], train_ids: list[str],
    rankings: dict[str, list[tuple[str, float]]],
) -> tuple[list[dict], list[dict]]:
    pool_b, pool_c = [], []
    for claim_id in train_ids:
        claim = claims[claim_id]
        annotated = set(map(str, claim.get("evidence", {})))
        seen = set()
        for rank, (doc_id, score) in enumerate(rankings[claim_id], 1):
            if doc_id in seen:
                raise ValueError(f"Duplicate ranked document for {claim_id}: {doc_id}")
            seen.add(doc_id)
            if doc_id in annotated:
                continue
            if rank == 1:
                pool_b.append(candidate_row(claim_id, claim, corpus[doc_id], rank, score, "B"))
            elif 2 <= rank <= 10:
                pool_c.append(candidate_row(claim_id, claim, corpus[doc_id], rank, score, "C"))
    return pool_b, pool_c


def random_negative_rows(
    claims: dict[str, dict], corpus: dict[str, dict], train_ids: list[str], seed: int,
    additional_excluded_by_claim: dict[str, set[str]] | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    doc_ids = sorted(corpus, key=int)
    additional_excluded_by_claim = additional_excluded_by_claim or {}
    rows, pairs = [], set()
    for claim_id in train_ids:
        claim = claims[claim_id]
        excluded = (
            set(map(str, claim.get("evidence", {})))
            | set(map(str, claim.get("cited_doc_ids", [])))
            | set(map(str, additional_excluded_by_claim.get(claim_id, set())))
        )
        eligible = [doc_id for doc_id in doc_ids if doc_id not in excluded]
        doc_id = eligible[rng.randrange(len(eligible))]
        pair = (claim_id, doc_id)
        if pair in pairs:
            raise AssertionError("Duplicate random pair")
        pairs.add(pair)
        rows.append({
            "claim_id": claim_id,
            "doc_id": doc_id,
            "condition": "D_random_negative",
            "source_split": "TRAIN",
            "claim": claim["claim"],
            "evidence": evidence_payload(corpus[doc_id]),
            "claim_gold_label": claim_gold_label(claim),
            "pair_annotation": None,
            "context_label": "INSUFFICIENT",
            "rationale_sentences": [],
            "annotation_basis": f"frozen_random_unrelated_pair_seed_{seed}",
            "retrieval_rank": None,
            "retrieval_score": None,
            "filter_version": None,
        })
    return rows


def sample_rows(rows: list[dict], size: int, seed: int, excluded: set[tuple[str, str]] | None = None) -> list[dict]:
    excluded = excluded or set()
    eligible = [row for row in rows if (row["claim_id"], row["doc_id"]) not in excluded]
    if len(eligible) < size:
        raise ValueError(f"Only {len(eligible)} eligible candidates for sample of {size}")
    return random.Random(seed).sample(eligible, size)


def write_audit_csv(path: Path, rows: list[dict], filter_decision: str = "not_yet_applied") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "claim_id": row["claim_id"],
                "doc_id": row["doc_id"],
                "claim": row["claim"],
                "evidence": json.dumps(row["evidence"], ensure_ascii=False),
                "retrieval_rank": row["retrieval_rank"],
                "retrieval_score": f'{row["retrieval_score"]:.9f}',
                "human_pair_label": "",
                "notes": "",
                "filter_decision": filter_decision,
            })


def read_audit(path: Path, require_labels: bool = True) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if require_labels:
        invalid = [row for row in rows if row["human_pair_label"] not in AUDIT_LABELS]
        if invalid:
            raise ValueError(f"{path} has {len(invalid)} missing/invalid judgments")
    return rows


def filter_accepts(row: dict, rule: dict) -> bool:
    if rule["type"] == "absolute_score_max":
        return float(row["retrieval_score"]) <= float(rule["max_score_inclusive"])
    if rule["type"] == "drop_all":
        return False
    if rule["type"] == "no_filter":
        return True
    raise ValueError(f"Unknown filter type: {rule['type']}")


def wilson_interval(contaminated: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        raise ValueError("Empty audit")
    p = contaminated / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def audit_summary(rows: list[dict]) -> dict:
    labels = Counter(row["human_pair_label"] for row in rows)
    contaminated = labels["SUPPORT"] + labels["CONTRADICT"]
    return {
        "count": len(rows),
        "label_counts": dict(labels),
        "contaminated_count": contaminated,
        "contamination_rate": contaminated / len(rows),
        "wilson_95_ci": wilson_interval(contaminated, len(rows)),
    }


def validate_rows(rows: list[dict], train_ids: set[str], condition: str) -> None:
    pairs = set()
    for row in rows:
        required = {
            "claim_id", "doc_id", "condition", "source_split", "claim", "evidence",
            "claim_gold_label", "pair_annotation", "context_label", "rationale_sentences",
            "annotation_basis", "retrieval_rank", "retrieval_score", "filter_version",
        }
        if not required <= set(row):
            raise ValueError(f"Missing provenance fields: {required - set(row)}")
        if row["claim_id"] not in train_ids or row["source_split"] != "TRAIN":
            raise ValueError("Non-TRAIN example")
        pair = (row["claim_id"], row["doc_id"])
        if pair in pairs:
            raise ValueError(f"Duplicate pair in {condition}: {pair}")
        pairs.add(pair)
        if row["claim_gold_label"] not in LABELS | {"MIXED"}:
            raise ValueError("Invalid claim label")
        if condition == "A":
            if row["pair_annotation"] not in PAIR_LABELS or row["context_label"] != row["pair_annotation"]:
                raise ValueError("Gold pair/context mismatch")
        elif condition in {"B", "C"}:
            if row["pair_annotation"] is not None or row["context_label"] != "INSUFFICIENT":
                raise ValueError("Audited negative label mismatch")
            if row["filter_version"] is None or row.get("filter_status") != "approved":
                raise ValueError("Pool example lacks frozen-filter approval")
        elif condition == "D":
            if row["pair_annotation"] is not None or row["context_label"] != "INSUFFICIENT":
                raise ValueError("Random negative label mismatch")
        if row["claim_gold_label"] == "MIXED" and condition != "A":
            raise ValueError("MIXED claims may appear only in Condition A")


def validate_cross_condition_pairs(conditions: dict[str, list[dict]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for condition, rows in conditions.items():
        for row in rows:
            pair = (row["claim_id"], row["doc_id"])
            if pair in seen:
                raise ValueError(f"Pair {pair} appears in both {seen[pair]} and {condition}")
            seen[pair] = condition


def load_inputs(config: dict, data_root: Path) -> tuple[dict, dict, list[str], dict]:
    split = json.loads((ROOT / "configs/splits/scifact_train_dev_v1.json").read_text())
    train_ids = sorted(map(str, split["train_ids"]), key=int)
    if len(train_ids) != config["scope"]["train_claim_count"]:
        raise ValueError("Frozen TRAIN count changed")
    claims = {str(row["id"]): row for row in read_jsonl(data_root / "original/claims_train.jsonl")}
    corpus = {str(row["doc_id"]): row for row in read_jsonl(data_root / "original/corpus.jsonl")}
    if not set(train_ids) <= set(claims):
        raise ValueError("Missing TRAIN claims")
    return claims, corpus, train_ids, split


def prepare(args: argparse.Namespace, config: dict) -> None:
    output = args.output_root
    if output.exists():
        raise FileExistsError(output)
    claims, corpus, train_ids, split = load_inputs(config, args.scifact_root)
    cache_meta = json.loads((args.embedding_cache / "metadata.json").read_text())
    if cache_meta["revision"] != config["retrieval"]["revision"]:
        raise ValueError("BGE cache revision mismatch")
    doc_ids = list(map(str, cache_meta["doc_ids"]))
    embeddings = np.load(args.embedding_cache / "embeddings.npy")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    encoder = BGE(config["retrieval"]["revision"])
    query_embeddings = encoder.encode([claims[q]["claim"] for q in train_ids], query=True)
    scores, positions = index.search(query_embeddings, config["retrieval"]["candidate_depth"])
    rankings = {
        query_id: [(doc_ids[int(position)], float(score)) for position, score in zip(row_positions, row_scores)]
        for query_id, row_positions, row_scores in zip(train_ids, positions, scores)
    }
    gold = gold_rows(claims, corpus, train_ids)
    pool_b, pool_c = build_candidates(claims, corpus, train_ids, rankings)
    random_rows = random_negative_rows(
        claims, corpus, train_ids, config["random_negative"]["seed"],
        {query_id: {doc_id for doc_id, _ in ranking} for query_id, ranking in rankings.items()},
    )
    output.mkdir(parents=True)
    audits = output / "audits"
    write_jsonl(output / "train_gold.jsonl", gold)
    write_jsonl(output / "pool-b-candidates.jsonl", pool_b)
    write_jsonl(output / "pool-c-candidates.jsonl", pool_c)
    write_jsonl(output / "train_random_negative.jsonl", random_rows)
    size = config["audit"]["sample_size"]
    write_audit_csv(audits / "pool-b-filter-development.csv", sample_rows(pool_b, size, config["audit"]["pool_b"]["filter_development_seed"]))
    write_audit_csv(audits / "pool-c-filter-development.csv", sample_rows(pool_c, size, config["audit"]["pool_c"]["filter_development_seed"]))
    validate_rows(gold, set(train_ids), "A")
    validate_rows(random_rows, set(train_ids), "D")
    manifest = {
        "status": "awaiting_filter_development_judgments",
        "benchmark_loaded": False,
        "split_sha256": sha256(ROOT / "configs/splits/scifact_train_dev_v1.json"),
        "config_sha256": sha256(args.config),
        "counts": {"A_gold": len(gold), "B_candidates": len(pool_b), "C_candidates": len(pool_c), "D_random": len(random_rows)},
        "condition_a_labels": dict(Counter(row["context_label"] for row in gold)),
        "mixed_train_claims": sum(claim_gold_label(claims[q]) == "MIXED" for q in train_ids),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.run_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.run_root / "train-rankings-top10.jsonl", [
        {"claim_id": query_id, "ranking": [{"doc_id": doc_id, "score": score} for doc_id, score in ranking]}
        for query_id, ranking in rankings.items()
    ])
    print(json.dumps(manifest, indent=2))


def make_frozen_audit(args: argparse.Namespace, config: dict) -> None:
    pool_key = args.pool.lower()
    pool_name = args.pool.upper()
    rule = config["audit"][f"pool_{pool_key}"]["filter"]
    if not rule:
        raise ValueError(f"Pool {pool_name} filter is not frozen in config")
    candidates = read_jsonl(args.output_root / f"pool-{pool_key}-candidates.jsonl")
    development_path = args.output_root / "audits" / f"pool-{pool_key}-filter-development.csv"
    development = read_audit(development_path)
    excluded = {(row["claim_id"], row["doc_id"]) for row in development}
    retained = [row for row in candidates if filter_accepts(row, rule)]
    sampled = sample_rows(
        retained, config["audit"]["sample_size"],
        config["audit"][f"pool_{pool_key}"]["frozen_filter_audit_seed"], excluded,
    )
    path = args.output_root / "audits" / f"pool-{pool_key}-frozen-filter-audit.csv"
    if path.exists():
        raise FileExistsError(path)
    write_audit_csv(path, sampled, "retained_by_frozen_filter")
    print(json.dumps({"pool": pool_name, "candidate_count": len(candidates), "retained_count": len(retained), "development": audit_summary(development), "filter": rule}, indent=2))


def finalize(args: argparse.Namespace, config: dict) -> None:
    claims, _, train_ids, _ = load_inputs(config, args.scifact_root)
    summaries, usable = {}, {}
    threshold = config["audit"]["stop_threshold"]
    for pool_name in ("B", "C"):
        key = pool_name.lower()
        rule = config["audit"][f"pool_{key}"]["filter"]
        if not rule:
            raise ValueError(f"Pool {pool_name} has no frozen filter")
        development = read_audit(args.output_root / "audits" / f"pool-{key}-filter-development.csv")
        audit = read_audit(args.output_root / "audits" / f"pool-{key}-frozen-filter-audit.csv")
        development_pairs = {(row["claim_id"], row["doc_id"]) for row in development}
        audit_pairs = {(row["claim_id"], row["doc_id"]) for row in audit}
        if development_pairs & audit_pairs:
            raise ValueError(f"Pool {pool_name} audit samples overlap")
        if any(not filter_accepts(row, rule) for row in audit):
            raise ValueError(f"Pool {pool_name} frozen audit contains rejected candidate")
        pre, post = audit_summary(development), audit_summary(audit)
        keep = post["contamination_rate"] <= threshold
        candidates = read_jsonl(args.output_root / f"pool-{key}-candidates.jsonl")
        retained = [row for row in candidates if filter_accepts(row, rule)] if keep else []
        filter_version = f"pool-{key}-frozen-v1"
        for row in retained:
            row.update({
                "context_label": "INSUFFICIENT",
                "annotation_basis": f"{filter_version}_audit_contamination_{post['contamination_rate']:.6f}",
                "filter_status": "approved",
                "audit_status": "pool_level_frozen_audit_passed",
                "filter_version": filter_version,
            })
        validate_rows(retained, set(train_ids), pool_name)
        output_name = "train_retrieved.jsonl" if pool_name == "B" else "train_hard_negative.jsonl"
        write_jsonl(args.output_root / output_name, retained)
        usable[pool_name] = retained
        summaries[pool_name] = {
            "candidate_count": len(candidates), "retained_by_filter": sum(filter_accepts(row, rule) for row in candidates),
            "usable_count": len(retained), "pre_filter_audit": pre, "frozen_filter_audit": post,
            "filter": rule, "threshold": threshold, "decision": "retain" if keep else "drop",
        }
    gold = read_jsonl(args.output_root / "train_gold.jsonl")
    random_rows = read_jsonl(args.output_root / "train_random_negative.jsonl")
    validate_rows(gold, set(train_ids), "A")
    validate_rows(random_rows, set(train_ids), "D")
    validate_cross_condition_pairs({"A": gold, "B": usable["B"], "C": usable["C"], "D": random_rows})
    all_usable = gold + usable["B"] + usable["C"] + random_rows
    summary = {
        "status": "milestone2_complete_no_training_started",
        "benchmark_loaded": False,
        "counts": {"A": len(gold), "B": len(usable["B"]), "C": len(usable["C"]), "D": len(random_rows)},
        "context_label_counts": dict(Counter(row["context_label"] for row in all_usable)),
        "condition_a_labels": dict(Counter(row["context_label"] for row in gold)),
        "audits": summaries,
        "random_seed": config["random_negative"]["seed"],
        "mixed_train_claims": sum(claim_gold_label(claims[q]) == "MIXED" for q in train_ids),
    }
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "make-audit", "finalize"))
    parser.add_argument("--pool", choices=("B", "C"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttraining/data-audit-v1.json")
    parser.add_argument("--scifact-root", type=Path, default=Path.home() / "rag/data/scifact")
    parser.add_argument("--embedding-cache", type=Path, default=Path.home() / "rag/embeddings/scifact-bge-a53fd0d3b47586f0")
    parser.add_argument("--output-root", type=Path, default=Path.home() / "rag/data/posttraining")
    parser.add_argument("--run-root", type=Path, default=Path.home() / "rag/runs/posttraining/data-audit-v1")
    args = parser.parse_args()
    if args.phase == "make-audit" and not args.pool:
        parser.error("--pool is required for make-audit")
    return args


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if config["scope"]["dev_or_benchmark_access_allowed"] is not False:
        raise ValueError("Milestone 2 must remain TRAIN-only")
    {"prepare": prepare, "make-audit": make_frozen_audit, "finalize": finalize}[args.phase](args, config)


if __name__ == "__main__":
    main()
