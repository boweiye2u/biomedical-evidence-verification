"""Reproduce historical B0 through the frozen FastAPI/vLLM serving stack."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import httpx

from retrieval.verification import LABELS, classification_metrics


def evaluate(records: list[dict]) -> dict:
    gold = [row["gold_label"] for row in records]
    predicted = [row["predicted_label"] for row in records]
    metrics = classification_metrics(gold, predicted)
    metrics.update(
        {
            "invalid_output_count": sum(not row["valid"] for row in records),
            "invalid_output_rate": sum(not row["valid"] for row in records) / len(records),
            "retrieval_top1_match_count": sum(row["retrieved_document_id"] == row["expected_document_id"] for row in records),
            "retrieval_top1_match_rate": sum(row["retrieved_document_id"] == row["expected_document_id"] for row in records) / len(records),
            "prediction_counts_including_invalid": dict(Counter(predicted)),
        }
    )
    return metrics


async def run_one(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, row: dict, expected: str) -> dict:
    async with semaphore:
        started = time.perf_counter()
        response = await client.post("/verify", json={"claim": row["claim"]})
        response.raise_for_status()
        payload = response.json()
    valid = bool(payload["valid"] and payload.get("label") in LABELS)
    return {
        "claim_id": row["claim_id"],
        "claim": row["claim"],
        "gold_label": row["label"],
        "predicted_label": payload.get("label") if valid else "INVALID",
        "valid": valid,
        "errors": payload.get("errors", []),
        "evidence_ids": payload.get("evidence_ids", []),
        "explanation": payload.get("explanation"),
        "retrieved_document_id": str(payload["retrieval"]["document_id"]),
        "expected_document_id": str(expected),
        "retrieval_score": payload["retrieval"]["score"],
        "request_id": payload["request_id"],
        "client_elapsed_ms": (time.perf_counter() - started) * 1000,
        "server_timing_ms": payload["timing_ms"],
    }


async def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    mapping = json.loads(args.mapping.read_text())["records"]
    rankings = json.loads(args.rankings.read_text())
    if len(mapping) != 300 or set(rankings) != {row["claim_id"] for row in mapping}:
        raise ValueError("Expected exactly 300 aligned benchmark claims")
    args.output_dir.mkdir(parents=True)
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        health = (await client.get("/health")).json()
        if health.get("status") != "ok":
            raise RuntimeError(f"Service is not healthy: {health}")
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [run_one(client, semaphore, row, rankings[row["claim_id"]][0]) for row in mapping]
        records = await asyncio.gather(*tasks)
    records.sort(key=lambda row: int(row["claim_id"]))
    with (args.output_dir / "predictions.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    metrics = evaluate(records)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output_dir / "run-config.json").write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "concurrency": args.concurrency,
                "claim_count": len(records),
                "mapping": str(args.mapping),
                "rankings": str(args.rankings),
                "benchmark_role": "fixed SciFact benchmark split, previously evaluated in v1 and not used for v2 training or model selection",
                "historical_v1_schema": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
