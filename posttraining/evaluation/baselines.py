"""Frozen DEV-only prompting baseline utilities and runner for Milestone 1."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from retrieval.verification import LABELS, classification_metrics, normalize_ranking

ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = (
    "Verify one biomedical claim using only the supplied document. Determine whether the "
    "document directly establishes the same entities, relation, population, and polarity. "
    "Choose SUPPORT only when the document entails the claim, CONTRADICT only when it "
    "establishes the opposite, and INSUFFICIENT otherwise. Do not use outside knowledge. "
    "Return exactly one JSON object with keys decision and rationale_sentences. decision must "
    "be SUPPORT, CONTRADICT, or INSUFFICIENT. rationale_sentences must be the zero-based indices "
    "of supplied sentences that justify the decision, and must be empty for INSUFFICIENT. "
    "The first output character must be { and the final output character must be }. Follow this "
    "literal form exactly: {\"decision\":\"SUPPORT\",\"rationale_sentences\":[0]}. "
    "Return JSON only, without markdown, labels outside the object, or explanation."
)


def prompt_sha256() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


def format_document(document: dict) -> str:
    sentences = document["abstract"]
    return "\n".join(
        [f"[EVIDENCE_ID: {document['doc_id']}]", f"TITLE: {document['title']}"]
        + [f"[{index}] {sentence}" for index, sentence in enumerate(sentences)]
    )


def user_content(claim: str, document: dict) -> str:
    return f"CLAIM:\n{claim}\n\nDOCUMENT:\n{format_document(document)}"


def build_messages(claim: str, document: dict, exemplars: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for exemplar in exemplars:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": user_content(exemplar["claim"], exemplar["document"]),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "decision": exemplar["decision"],
                            "rationale_sentences": exemplar["rationale_sentences"],
                        },
                        separators=(",", ":"),
                    ),
                },
            ]
        )
    messages.append({"role": "user", "content": user_content(claim, document)})
    return messages


def parse_output(raw: str, sentence_count: int) -> dict:
    errors: list[str] = []
    try:
        value = json.loads(raw.strip())
        json_valid = True
    except (json.JSONDecodeError, TypeError):
        return {
            "value": None,
            "decision": None,
            "json_valid": False,
            "schema_valid": False,
            "decision_valid": False,
            "rationale_index_valid": False,
            "errors": ["invalid_json"],
        }
    if not isinstance(value, dict):
        return {
            "value": value,
            "decision": None,
            "json_valid": True,
            "schema_valid": False,
            "decision_valid": False,
            "rationale_index_valid": False,
            "errors": ["output_not_object"],
        }
    schema_valid = set(value) == {"decision", "rationale_sentences"}
    if not schema_valid:
        errors.append("schema_keys")
    decision = value.get("decision")
    decision_valid = decision in LABELS
    if not decision_valid:
        errors.append("invalid_decision")
    rationale = value.get("rationale_sentences")
    rationale_type_valid = (
        isinstance(rationale, list)
        and all(isinstance(index, int) and not isinstance(index, bool) for index in rationale)
    )
    if not rationale_type_valid:
        errors.append("invalid_rationale_type")
        rationale_valid = False
    else:
        rationale_valid = (
            len(rationale) == len(set(rationale))
            and all(0 <= index < sentence_count for index in rationale)
            and ((decision == "INSUFFICIENT" and not rationale) or (decision in {"SUPPORT", "CONTRADICT"} and bool(rationale)))
        )
        if not rationale_valid:
            errors.append("invalid_rationale_indices")
    return {
        "value": value,
        "decision": decision if decision_valid else None,
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "decision_valid": decision_valid,
        "rationale_index_valid": rationale_valid,
        "errors": errors,
    }


def validate_config(config: dict, split: dict, exemplars: dict | None) -> None:
    if config["scope"]["split"] != "frozen_DEV_only" or config["scope"]["query_count"] != 162:
        raise ValueError("Milestone 1 baselines must be frozen DEV only")
    if config["model"]["revision"] in {"", "main", None} or config["model"]["tokenizer_revision"] in {"", "main", None}:
        raise ValueError("Model and tokenizer revisions must be immutable commits")
    if config["prompt"]["sha256"] != prompt_sha256():
        raise ValueError("Prompt hash mismatch")
    if config["inference"]["max_model_len"] != 16384 or config["inference"]["request_concurrency"] != 8:
        raise ValueError("Frozen serving settings changed")
    if len(split["dev_ids"]) != 162 or len(split["train_ids"]) != 647:
        raise ValueError("Frozen split changed")
    if config["prompt"]["fewshot"]:
        if exemplars is None or len(exemplars["examples"]) not in {3, 4, 5}:
            raise ValueError("Few-shot baseline requires 3-5 frozen exemplars")
        train, dev = set(map(str, split["train_ids"])), set(map(str, split["dev_ids"]))
        ids = {str(row["claim_id"]) for row in exemplars["examples"]}
        if not ids <= train or ids & dev:
            raise ValueError("Few-shot exemplars leak outside TRAIN")
        if {row["decision"] for row in exemplars["examples"]} != set(LABELS):
            raise ValueError("Few-shot exemplars must contain all three labels")


def summarize(records: list[dict], wall_seconds: float, peak_gpu_mib: int) -> dict:
    gold = [row["gold_label"] for row in records]
    predicted = [row["predicted_label"] for row in records]
    metrics = classification_metrics(gold, predicted)
    latencies = sorted(row["request_ms"] for row in records)
    output_tokens = [row["output_tokens"] for row in records]
    count = len(records)
    metrics.update(
        {
            "true_label_distribution": dict(Counter(gold)),
            "prediction_counts_including_invalid": dict(Counter(predicted)),
            "json_valid_count": sum(row["json_valid"] for row in records),
            "json_valid_rate": sum(row["json_valid"] for row in records) / count,
            "schema_valid_count": sum(row["schema_valid"] for row in records),
            "schema_valid_rate": sum(row["schema_valid"] for row in records) / count,
            "decision_valid_count": sum(row["decision_valid"] for row in records),
            "decision_valid_rate": sum(row["decision_valid"] for row in records) / count,
            "rationale_index_valid_count": sum(row["rationale_index_valid"] for row in records),
            "rationale_index_valid_rate": sum(row["rationale_index_valid"] for row in records) / count,
            "invalid_output_count": sum(not row["decision_valid"] for row in records),
            "prompt_tokens": {
                "mean": statistics.fmean(row["prompt_tokens"] for row in records),
                "max": max(row["prompt_tokens"] for row in records),
            },
            "output_tokens": {
                "mean": statistics.fmean(output_tokens),
                "max": max(output_tokens),
            },
            "efficiency": {
                "wall_seconds": wall_seconds,
                "requests_per_second": count / wall_seconds,
                "mean_request_ms": statistics.fmean(latencies),
                "p50_request_ms": statistics.median(latencies),
                "p95_request_ms": latencies[max(0, int(0.95 * count) - 1)],
                "peak_gpu_memory_mib": peak_gpu_mib,
            },
        }
    )
    return metrics


def gpu_memory_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]
    return max(values, default=0)


async def monitor_gpu(stop: asyncio.Event, values: list[int]) -> None:
    while not stop.is_set():
        try:
            values.append(await asyncio.to_thread(gpu_memory_mib))
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.2)
        except TimeoutError:
            continue


async def generate_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    model_name: str,
    seed: int,
    temperature: float,
    repetition_penalty: float,
    max_tokens: int,
    record: dict,
) -> dict:
    async with semaphore:
        started = time.perf_counter()
        response = await client.post(
            "/v1/completions",
            json={
                "model": model_name,
                "prompt": record["rendered_prompt"],
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "max_tokens": max_tokens,
                "seed": seed,
                "stream": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
        elapsed = (time.perf_counter() - started) * 1000
    raw = payload["choices"][0]["text"].strip()
    parsed = parse_output(raw, record["sentence_count"])
    usage = payload.get("usage") or {}
    return {
        "claim_id": record["claim_id"],
        "gold_label": record["gold_label"],
        "document_id": record["document_id"],
        "raw_output": raw,
        "predicted_label": parsed["decision"] or "INVALID",
        **{key: parsed[key] for key in ("json_valid", "schema_valid", "decision_valid", "rationale_index_valid", "errors")},
        "prompt_tokens": usage.get("prompt_tokens", record["prompt_tokens"]),
        "output_tokens": usage.get("completion_tokens", 0),
        "request_ms": elapsed,
    }


async def run(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    split = json.loads((ROOT / "configs/splits/scifact_train_dev_v1.json").read_text())
    exemplar_payload = json.loads(args.exemplars.read_text()) if config["prompt"]["fewshot"] else None
    validate_config(config, split, exemplar_payload)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    mapping_payload = json.loads(args.mapping.read_text())
    if mapping_payload.get("scope") != "frozen DEV only" or mapping_payload.get("test_labels_loaded") is not False:
        raise ValueError("Mapping is not the frozen DEV-only artifact")
    mapping = {row["claim_id"]: row for row in mapping_payload["records"]}
    dev_ids = sorted(map(str, split["dev_ids"]), key=int)
    if set(mapping) != set(dev_ids):
        raise ValueError("Generated IDs would not align with frozen DEV")
    rankings = json.loads(args.rankings.read_text())
    if set(rankings) != set(dev_ids):
        raise ValueError("Rankings do not align with frozen DEV")
    corpus = {str(row["doc_id"]): row for row in map(json.loads, args.corpus.read_text().splitlines())}
    exemplars = exemplar_payload["examples"] if exemplar_payload else []
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    inputs = []
    for claim_id in dev_ids:
        document_id = normalize_ranking(rankings[claim_id])[0]
        document = corpus[document_id]
        messages = build_messages(mapping[claim_id]["claim"], document, exemplars)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        if tokens + config["inference"]["max_new_tokens"] > config["inference"]["max_model_len"]:
            raise ValueError(f"Prompt truncation risk for claim {claim_id}")
        inputs.append(
            {
                "claim_id": claim_id,
                "gold_label": mapping[claim_id]["label"],
                "document_id": document_id,
                "sentence_count": len(document["abstract"]),
                "rendered_prompt": rendered,
                "prompt_tokens": tokens,
            }
        )
    args.output_dir.mkdir(parents=True)
    stop, memory = asyncio.Event(), []
    monitor = asyncio.create_task(monitor_gpu(stop, memory))
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(base_url=args.base_url, timeout=httpx.Timeout(300.0)) as client:
            health = await client.get("/health")
            health.raise_for_status()
            semaphore = asyncio.Semaphore(config["inference"]["request_concurrency"])
            tasks = [
                generate_one(
                    client,
                    semaphore,
                    config["model"]["served_model_name"],
                    config["inference"]["seed"],
                    config["inference"]["temperature"],
                    config["inference"]["repetition_penalty"],
                    config["inference"]["max_new_tokens"],
                    row,
                )
                for row in inputs
            ]
            records = await asyncio.gather(*tasks)
    finally:
        stop.set()
        await monitor
    wall = time.perf_counter() - started
    records.sort(key=lambda row: int(row["claim_id"]))
    metrics = summarize(records, wall, max(memory, default=0))
    with (args.output_dir / "predictions.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "prompt-lengths.json").write_text(
        json.dumps({row["claim_id"]: row["prompt_tokens"] for row in inputs}, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--exemplars", type=Path, default=ROOT / "configs/posttraining/fewshot-exemplars-v1.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
