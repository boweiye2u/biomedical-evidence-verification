"""One-shot frozen SciFact benchmark evaluation for B0-B3, M1, and M2."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import torch
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttraining.evaluation.baselines import (
    build_messages,
    parse_output,
    prompt_sha256,
)
from retrieval.verification import LABELS, classification_metrics, normalize_ranking

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/posttraining/final-eval-v1.json"


def digest(path: Path) -> str:
    block = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            block.update(chunk)
    return block.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_inputs(config: dict) -> tuple[list[dict], dict, dict]:
    inputs = config["inputs"]
    mapping_payload = json.loads(Path(inputs["mapping"]).read_text())
    rows = sorted(mapping_payload["records"], key=lambda row: int(row["claim_id"]))
    rankings = json.loads(Path(inputs["rankings"]).read_text())
    corpus_rows = read_jsonl(Path(inputs["corpus"]))
    corpus = {str(row["doc_id"]): row for row in corpus_rows}
    ids = [str(row["claim_id"]) for row in rows]
    if len(rows) != 300 or len(set(ids)) != 300 or set(rankings) != set(ids):
        raise ValueError("Frozen benchmark is not exactly 300 aligned unique claims")
    if len(corpus) != 5183:
        raise ValueError("Frozen corpus is not exactly 5,183 documents")
    split = json.loads((ROOT / "configs/splits/scifact_train_dev_v1.json").read_text())
    if set(ids) & (set(map(str, split["train_ids"])) | set(map(str, split["dev_ids"]))):
        raise ValueError("Fixed benchmark overlaps training-derived TRAIN/DEV")
    for row in rows:
        top = normalize_ranking(rankings[row["claim_id"]])[0]
        if top not in corpus:
            raise ValueError(f"Missing top document for {row['claim_id']}")
    return rows, rankings, corpus


def validate_frozen(config_path: Path, config: dict) -> dict:
    if config["status"] != "frozen_before_main_benchmark_predictions":
        raise ValueError("Final evaluation config is not frozen")
    if config["condition"] != {"id": "E1", "description": "frozen BGE retrieved top-1 full abstract", "primary": True}:
        raise ValueError("Primary condition changed")
    if config["prompt"]["sha256"] != prompt_sha256():
        raise ValueError("Prompt hash changed")
    inputs = config["inputs"]
    for key in ("mapping", "rankings", "corpus"):
        if digest(Path(inputs[key])) != inputs[f"{key}_sha256"]:
            raise ValueError(f"Frozen {key} hash mismatch")
    exemplar = ROOT / inputs["fewshot_exemplars"]
    if digest(exemplar) != inputs["fewshot_exemplars_sha256"]:
        raise ValueError("Few-shot exemplar hash mismatch")
    hashes = {"final_eval_config": digest(config_path), "inputs": {}, "systems": {}}
    for key in ("mapping", "rankings", "corpus"):
        hashes["inputs"][key] = digest(Path(inputs[key]))
    for system_id, system in config["systems"].items():
        cfg_path = ROOT / system["frozen_config"]
        if digest(cfg_path) != system["frozen_config_sha256"]:
            raise ValueError(f"{system_id} frozen config hash mismatch")
        model_path = Path(system["model_path"])
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        actual = {name: digest(model_path / name) for name in system["weight_sha256"]}
        if actual != system["weight_sha256"]:
            raise ValueError(f"{system_id} weight hash mismatch")
        hashes["systems"][system_id] = {
            "path": str(model_path),
            "frozen_config_sha256": digest(cfg_path),
            "weight_sha256": actual,
        }
    load_inputs(config)
    return hashes


def preflight(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    run_root = Path(config["run_root"])
    if run_root.exists():
        raise FileExistsError(f"Refusing populated or existing benchmark run: {run_root}")
    hashes = validate_frozen(args.config, config)
    run_root.mkdir(parents=True)
    environment = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    payload = {
        "status": "passed_before_predictions",
        "benchmark_role": config["benchmark_role"],
        "claim_count": 300,
        "claim_order": "ascending numeric claim_id",
        "identical_order_across_systems": True,
        "hashes": hashes,
        "environment": environment,
        "secondary_conditions_run": [],
        "primary_condition": "E1",
    }
    (run_root / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
    (run_root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def make_inputs(config: dict, system_id: str) -> tuple[list[dict], AutoTokenizer]:
    rows, rankings, corpus = load_inputs(config)
    system = config["systems"][system_id]
    exemplar_payload = json.loads((ROOT / config["inputs"]["fewshot_exemplars"]).read_text())
    exemplars = exemplar_payload["examples"] if system["fewshot"] else []
    tokenizer = AutoTokenizer.from_pretrained(system["model_path"], local_files_only=True)
    prepared = []
    for row in rows:
        doc_id = normalize_ranking(rankings[row["claim_id"]])[0]
        document = corpus[doc_id]
        rendered = tokenizer.apply_chat_template(
            build_messages(row["claim"], document, exemplars),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        if prompt_tokens + config["inference"]["max_new_tokens"] > config["inference"]["max_model_len"]:
            raise ValueError(f"Prompt too long for {system_id}/{row['claim_id']}")
        prepared.append({
            "claim_id": row["claim_id"], "claim": row["claim"], "gold_label": row["label"],
            "document_id": doc_id, "sentence_count": len(document["abstract"]),
            "rendered_prompt": rendered, "prompt_tokens": prompt_tokens,
            "annotated_evidence": row["annotated_evidence"],
        })
    return prepared, tokenizer


async def _generate_one(client, semaphore, config, system, row):
    async with semaphore:
        started = time.perf_counter()
        response = await client.post("/v1/completions", json={
            "model": system["served_model_name"], "prompt": row["rendered_prompt"],
            "temperature": config["inference"]["temperature"],
            "repetition_penalty": config["inference"]["repetition_penalty"],
            "max_tokens": config["inference"]["max_new_tokens"],
            "seed": config["inference"]["seed"], "stream": False,
        })
        response.raise_for_status()
        payload = response.json()
        elapsed = (time.perf_counter() - started) * 1000
    raw = payload["choices"][0]["text"].strip()
    parsed = parse_output(raw, row["sentence_count"])
    value = parsed.get("value") if isinstance(parsed.get("value"), dict) else {}
    rationale = value.get("rationale_sentences") if isinstance(value.get("rationale_sentences"), list) else []
    usage = payload.get("usage") or {}
    return {
        "claim_id": row["claim_id"], "claim": row["claim"], "gold_label": row["gold_label"],
        "document_id": row["document_id"], "raw_generation": raw,
        "predicted_label": parsed["decision"] or "INVALID", "predicted_rationale_sentences": rationale,
        **{key: parsed[key] for key in ("json_valid", "schema_valid", "decision_valid", "rationale_index_valid", "errors")},
        "prompt_tokens": usage.get("prompt_tokens", row["prompt_tokens"]),
        "output_tokens": usage.get("completion_tokens", 0), "request_ms": elapsed,
    }


async def generate(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    system_id = args.system
    if system_id not in config["systems"]:
        raise ValueError(system_id)
    run_root = Path(config["run_root"])
    if not (run_root / "preflight.json").is_file():
        raise RuntimeError("Preflight is missing")
    output = run_root / "systems" / system_id
    if output.exists():
        raise FileExistsError(output)
    prepared, _ = make_inputs(config, system_id)
    system = config["systems"][system_id]
    output.mkdir(parents=True)
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=args.base_url, timeout=httpx.Timeout(300.0)) as client:
        response = await client.get("/health")
        response.raise_for_status()
        semaphore = asyncio.Semaphore(config["inference"]["request_concurrency"])
        records = await asyncio.gather(*[
            _generate_one(client, semaphore, config, system, row) for row in prepared
        ])
    wall = time.perf_counter() - started
    records.sort(key=lambda row: int(row["claim_id"]))
    with (output / "predictions.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    generation = {
        "system": system_id, "count": len(records), "wall_seconds": wall,
        "requests_per_second": len(records) / wall,
        "mean_request_ms": statistics.fmean(row["request_ms"] for row in records),
        "p50_request_ms": statistics.median(row["request_ms"] for row in records),
        "p95_request_ms": sorted(row["request_ms"] for row in records)[284],
        "mean_prompt_tokens": statistics.fmean(row["prompt_tokens"] for row in records),
        "max_prompt_tokens": max(row["prompt_tokens"] for row in records),
        "mean_output_tokens": statistics.fmean(row["output_tokens"] for row in records),
    }
    (output / "generation-summary.json").write_text(json.dumps(generation, indent=2) + "\n")
    print(json.dumps(generation, indent=2))


def score_continuations(model, tokenizer, prompts: list[str], labels: list[str], prefix: str, batch_size: int) -> list[dict]:
    device = next(model.parameters()).device
    work = []
    for row_index, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        for label in labels:
            full_ids = tokenizer.encode(prompt + prefix + label, add_special_tokens=False)
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("Restricted-choice prompt is not a token prefix")
            work.append((row_index, label, full_ids, len(prompt_ids)))
    scores = [dict() for _ in prompts]
    tokenizer.pad_token = tokenizer.eos_token
    for offset in range(0, len(work), batch_size):
        part = work[offset:offset + batch_size]
        width = max(len(item[2]) for item in part)
        input_ids, masks = [], []
        for _, _, ids, _ in part:
            pad = width - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id] * pad)
            masks.append([1] * len(ids) + [0] * pad)
        ids_tensor = torch.tensor(input_ids, device=device)
        mask_tensor = torch.tensor(masks, device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids_tensor, attention_mask=mask_tensor).logits.float()
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        for index, (row_index, label, ids, prompt_len) in enumerate(part):
            positions = torch.arange(prompt_len - 1, len(ids) - 1, device=device)
            targets = ids_tensor[index, prompt_len:len(ids)]
            value = log_probs[index, positions, targets].sum().item()
            scores[row_index][label] = value
        del logits, log_probs, ids_tensor, mask_tensor
    return scores


def calibrate(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    system_ids = args.system
    paths = {config["systems"][sid]["model_path"] for sid in system_ids}
    if len(paths) != 1:
        raise ValueError("A calibration invocation must share one checkpoint")
    model_path = paths.pop()
    for sid in system_ids:
        output = Path(config["run_root"]) / "systems" / sid
        if not (output / "predictions.jsonl").is_file() or (output / "calibration.jsonl").exists():
            raise RuntimeError(f"Predictions missing or calibration exists for {sid}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
    for sid in system_ids:
        prepared, _ = make_inputs(config, sid)
        scores = score_continuations(
            model, tokenizer, [row["rendered_prompt"] for row in prepared],
            config["calibration"]["labels"], config["calibration"]["continuation_prefix"], args.batch_size,
        )
        records = []
        for row, logits in zip(prepared, scores):
            values = np.array([logits[label] for label in LABELS], dtype=np.float64)
            probabilities = np.exp(values - values.max()); probabilities /= probabilities.sum()
            records.append({
                "claim_id": row["claim_id"], "gold_label": row["gold_label"],
                "sequence_log_likelihood": logits,
                "restricted_probabilities": {label: float(p) for label, p in zip(LABELS, probabilities)},
                "restricted_argmax": LABELS[int(probabilities.argmax())],
            })
        output = Path(config["run_root"]) / "systems" / sid
        with (output / "calibration.jsonl").open("w") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")
        print(f"calibrated {sid}: {len(records)}")
    del model
    torch.cuda.empty_cache()


def rationale_metrics(predictions: list[dict], mapping: dict) -> dict:
    values = []
    reasons = Counter()
    for row in predictions:
        source = mapping[row["claim_id"]]
        if source["label"] not in {"SUPPORT", "CONTRADICT"}:
            reasons["gold_insufficient"] += 1; continue
        candidates = [
            set(item["sentence_ids"])
            for item in source["annotated_evidence"].get(row["document_id"], [])
            if item["label"] == source["label"] and item["sentence_ids"]
        ]
        if not candidates:
            reasons["retrieved_document_without_valid_gold_rationale"] += 1; continue
        predicted = set(x for x in row["predicted_rationale_sentences"] if isinstance(x, int) and not isinstance(x, bool))
        alternatives = []
        for gold in candidates:
            tp = len(predicted & gold)
            precision = tp / len(predicted) if predicted else 0.0
            recall = tp / len(gold)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            alternatives.append((f1, precision, recall))
        best = max(alternatives)
        values.append({"claim_id": row["claim_id"], "f1": best[0], "precision": best[1], "recall": best[2]})
    return {
        "eligible_count": len(values), "excluded_count": len(predictions) - len(values),
        "exclusion_reasons": dict(reasons),
        "precision": statistics.fmean(x["precision"] for x in values) if values else None,
        "recall": statistics.fmean(x["recall"] for x in values) if values else None,
        "f1": statistics.fmean(x["f1"] for x in values) if values else None,
        "per_example": values,
    }


def calibration_metrics(calibration: list[dict], predictions: list[dict], bins: int) -> dict:
    by_id = {row["claim_id"]: row for row in predictions}
    confidences, correct, nll, brier = [], [], [], []
    agreement = 0
    for row in calibration:
        probs = row["restricted_probabilities"]
        conf = max(probs.values()); argmax = row["restricted_argmax"]
        confidences.append(conf); correct.append(argmax == row["gold_label"])
        nll.append(-math.log(max(probs[row["gold_label"]], 1e-300)))
        brier.append(sum((probs[label] - (label == row["gold_label"])) ** 2 for label in LABELS))
        agreement += argmax == by_id[row["claim_id"]]["predicted_label"]
    ece = 0.0
    details = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [i for i, value in enumerate(confidences) if (low <= value <= high if index == bins - 1 else low <= value < high)]
        if selected:
            accuracy = sum(correct[i] for i in selected) / len(selected)
            confidence = sum(confidences[i] for i in selected) / len(selected)
            ece += len(selected) / len(confidences) * abs(accuracy - confidence)
            details.append({"lower": low, "upper": high, "count": len(selected), "accuracy": accuracy, "confidence": confidence})
    return {
        "ece": ece, "brier_score": statistics.fmean(brier), "nll": statistics.fmean(nll),
        "restricted_generation_agreement_count": agreement,
        "restricted_generation_agreement_rate": agreement / len(calibration),
        "restricted_argmax_accuracy": sum(correct) / len(correct), "ece_bins": details,
    }


def holm(raw: list[float]) -> list[float]:
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    adjusted = [0.0] * len(raw); running = 0.0; count = len(raw)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * raw[index]))
        adjusted[index] = running
    return adjusted


def bootstrap_macro(gold, left, right, repeats, seed):
    rng = np.random.default_rng(seed); count = len(gold); values = np.empty(repeats)
    gold = np.asarray(gold); left = np.asarray(left); right = np.asarray(right)
    for index in range(repeats):
        take = rng.integers(0, count, count)
        values[index] = classification_metrics(gold[take].tolist(), left[take].tolist())["macro_f1"] - classification_metrics(gold[take].tolist(), right[take].tolist())["macro_f1"]
    return values


def finalize(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text()); run_root = Path(config["run_root"])
    rows, _, _ = load_inputs(config); mapping = {row["claim_id"]: row for row in rows}; expected = [row["claim_id"] for row in rows]
    summaries, predictions = {}, {}
    for sid in config["systems"]:
        output = run_root / "systems" / sid
        pred = read_jsonl(output / "predictions.jsonl"); cal = read_jsonl(output / "calibration.jsonl")
        if [row["claim_id"] for row in pred] != expected or [row["claim_id"] for row in cal] != expected:
            raise ValueError(f"{sid} order mismatch")
        predictions[sid] = pred
        metric = classification_metrics([row["gold_label"] for row in pred], [row["predicted_label"] for row in pred])
        metric.update({
            "true_label_distribution": dict(Counter(row["gold_label"] for row in pred)),
            "prediction_counts_including_invalid": dict(Counter(row["predicted_label"] for row in pred)),
            "json_valid_rate": sum(row["json_valid"] for row in pred) / len(pred),
            "schema_valid_rate": sum(row["schema_valid"] for row in pred) / len(pred),
            "decision_valid_rate": sum(row["decision_valid"] for row in pred) / len(pred),
            "rationale_index_valid_rate": sum(row["rationale_index_valid"] for row in pred) / len(pred),
            "invalid_output_rate": sum(not row["decision_valid"] for row in pred) / len(pred),
            "calibration": calibration_metrics(cal, pred, config["calibration"]["ece_equal_width_bins"]),
            "rationale": rationale_metrics(pred, mapping),
            "generation": json.loads((output / "generation-summary.json").read_text()),
        })
        summaries[sid] = metric
        (output / "metrics.json").write_text(json.dumps(metric, indent=2) + "\n")
    comparisons = []
    raw_p = []
    bootstrap_output = {}
    gold = [row["gold_label"] for row in predictions["B0"]]
    for comparison in config["statistics"]["comparisons"]:
        left_id, right_id = comparison["left"], comparison["right"]
        left = [row["predicted_label"] for row in predictions[left_id]]
        right = [row["predicted_label"] for row in predictions[right_id]]
        left_correct = [g == p for g, p in zip(gold, left)]; right_correct = [g == p for g, p in zip(gold, right)]
        left_only = sum(a and not b for a, b in zip(left_correct, right_correct))
        right_only = sum(b and not a for a, b in zip(left_correct, right_correct))
        discordant = left_only + right_only
        p = binomtest(left_only, discordant, 0.5, alternative="two-sided").pvalue if discordant else 1.0
        raw_p.append(p)
        boot = bootstrap_macro(gold, left, right, config["statistics"]["bootstrap_resamples"], config["statistics"]["bootstrap_seed"] + comparison["id"])
        bootstrap_output[str(comparison["id"])] = boot.tolist()
        comparisons.append({
            **comparison, "accuracy_difference": summaries[left_id]["accuracy"] - summaries[right_id]["accuracy"],
            "left_correct_right_wrong": left_only, "right_correct_left_wrong": right_only,
            "discordant": discordant, "mcnemar_exact_raw_p": p,
            "macro_f1_difference": summaries[left_id]["macro_f1"] - summaries[right_id]["macro_f1"],
            "macro_f1_bootstrap_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "bootstrap_resamples": len(boot),
        })
    for row, adjusted in zip(comparisons, holm(raw_p)):
        row["mcnemar_holm_adjusted_p"] = adjusted
    stats_dir = run_root / "statistics"; stats_dir.mkdir()
    (stats_dir / "bootstrap-distributions.json").write_text(json.dumps(bootstrap_output) + "\n")
    (stats_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    result = {
        "status": "complete", "benchmark_role": config["benchmark_role"],
        "primary_condition": "E1", "systems": summaries, "comparisons": comparisons,
        "integrity": {"claim_count": 300, "identical_claim_order": True, "main_run_count": 1, "config_sha256": digest(args.config)},
    }
    (run_root / "final-results.json").write_text(json.dumps(result, indent=2) + "\n")
    (ROOT / "docs/posttraining/milestone6-main-benchmark-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "systems": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for k, v in summaries.items()}, "comparisons": comparisons}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "generate", "calibrate", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--system", action="append", choices=("B0", "B1", "B2", "B3", "M1", "M2"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.phase in {"generate", "calibrate"} and not args.system:
        parser.error("--system is required")
    if args.phase == "generate" and len(args.system) != 1:
        parser.error("generate accepts exactly one --system")
    if args.phase == "generate":
        args.system = args.system[0]
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "preflight": preflight(args)
    elif args.phase == "generate": asyncio.run(generate(args))
    elif args.phase == "calibrate": calibrate(args)
    else: finalize(args)
