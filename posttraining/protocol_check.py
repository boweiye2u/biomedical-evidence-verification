"""Milestone 0 checks for the frozen biomedical verifier post-training study."""
from __future__ import annotations

import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

from retrieval.verification import prompt_hash

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def claim_label(record: dict) -> tuple[str, dict[str, str]]:
    pair_labels: dict[str, str] = {}
    for document_id, groups in record.get("evidence", {}).items():
        labels = {group["label"] for group in groups}
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels within pair {record['id']}/{document_id}: {labels}")
        pair_labels[str(document_id)] = next(iter(labels))
    unique = set(pair_labels.values())
    if not unique:
        return "INSUFFICIENT", pair_labels
    if len(unique) == 1:
        return next(iter(unique)), pair_labels
    return "MIXED", pair_labels


def inspect_claims(records: list[dict], ids: list[str]) -> dict:
    by_id = {str(record["id"]): record for record in records}
    if set(ids) - set(by_id):
        raise ValueError("Split contains claim IDs missing from claims_train.jsonl")
    labels: Counter[str] = Counter()
    mixed_ids: list[str] = []
    pair_count = 0
    for claim_id in ids:
        label, pairs = claim_label(by_id[claim_id])
        labels[label] += 1
        pair_count += len(pairs)
        if label == "MIXED":
            mixed_ids.append(claim_id)
    return {
        "count": len(ids),
        "label_counts": dict(sorted(labels.items())),
        "annotated_pair_count": pair_count,
        "mixed_claim_count": len(mixed_ids),
        "mixed_claim_ids": mixed_ids,
    }


def load_archive_jsonl(archive: Path, member: str) -> list[dict]:
    with tarfile.open(archive) as handle:
        stream = handle.extractfile(member)
        if stream is None:
            raise FileNotFoundError(member)
        return [json.loads(line) for line in stream.read().decode().splitlines() if line.strip()]


def run_checks(artifact_root: Path) -> dict:
    config_path = ROOT / "configs/posttraining/protocol-v1.json"
    config = read_json(config_path)
    split_path = ROOT / "configs/splits/scifact_train_dev_v1.json"
    split = read_json(split_path)
    data = artifact_root / "data/scifact"
    claims_path = data / "original/claims_train.jsonl"
    claims = read_jsonl(claims_path)
    train_ids = list(map(str, split["train_ids"]))
    dev_ids = list(map(str, split["dev_ids"]))
    if len(train_ids) != 647 or len(dev_ids) != 162:
        raise ValueError("Frozen TRAIN/DEV counts changed")
    if set(train_ids) & set(dev_ids):
        raise ValueError("TRAIN and DEV overlap")
    if set(train_ids) | set(dev_ids) != {str(row["id"]) for row in claims}:
        raise ValueError("TRAIN/DEV do not exactly partition claims_train.jsonl")

    mapping = read_json(artifact_root / "runs/milestone6a-test-v1/test-claim-mapping.json")
    benchmark_records = mapping["records"]
    archive_records = load_archive_jsonl(
        data / "archives/original.tar.gz", config["splits"]["benchmark_original_member"]
    )
    archive_by_id = {str(row["id"]): row for row in archive_records}
    if len(benchmark_records) != 300 or len(archive_by_id) != 300:
        raise ValueError("Fixed benchmark must contain 300 claims")
    text_mismatches = [
        row["claim_id"]
        for row in benchmark_records
        if archive_by_id.get(row["claim_id"], {}).get("claim") != row["claim"]
    ]
    if text_mismatches:
        raise ValueError(f"Benchmark claim-text mismatch: {text_mismatches[:5]}")
    if (set(train_ids) | set(dev_ids)) & set(archive_by_id):
        raise ValueError("Training-derived claims overlap fixed benchmark")

    final_config = read_json(ROOT / "configs/final-test-6a-v1.json")
    serving_config = read_json(ROOT / "configs/serving-6b-v1.json")
    expected_retriever = config["retriever"]
    for field in ("revision", "query_prefix", "pooling", "corpus_size"):
        if final_config["retriever"][field] != expected_retriever[field]:
            raise ValueError(f"Retriever mismatch in {field}")
    if final_config["retriever"]["index"] != expected_retriever["index"]:
        raise ValueError("FAISS index mismatch")
    if prompt_hash(config["b0_reproduction"]["prompt_name"]) != config["b0_reproduction"]["prompt_sha256"]:
        raise ValueError("Frozen prompt hash mismatch")
    if serving_config["generator"]["revision"] != config["b0_reproduction"]["revision"]:
        raise ValueError("B0 model revision mismatch")

    return {
        "status": "passed",
        "config_sha256": sha256(config_path),
        "split_sha256": sha256(split_path),
        "claims_train_sha256": sha256(claims_path),
        "train": inspect_claims(claims, train_ids),
        "dev": inspect_claims(claims, dev_ids),
        "train_dev_overlap": 0,
        "benchmark": {
            "count": len(benchmark_records),
            "original_member": config["splits"]["benchmark_original_member"],
            "claim_text_mismatches": 0,
            "training_derived_overlap": 0,
            "previously_evaluated_in_v1": True,
        },
        "v1_scoring": config["v1_scoring"],
        "retriever": expected_retriever,
        "b0_reproduction": config["b0_reproduction"],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path.home() / "rag")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_checks(args.artifact_root)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
