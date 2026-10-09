"""Frozen M3 CV/DEV aggregation utilities."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttraining.training.m1_gold import evaluate, read_jsonl

ROOT = Path(__file__).resolve().parents[2]
RUN = Path.home() / "rag/runs/posttraining/full-ft-v1"
CKPT = Path.home() / "rag/checkpoints/posttraining/m3"
CONFIG = ROOT / "configs/posttraining/train-m3-full-ft-v1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evaluate_checkpoint(model_path: Path, rows: list[dict], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).cuda()
    model.config.use_cache = True
    metrics, predictions = evaluate(model, tokenizer, rows, batch_size=8)
    output.mkdir(parents=True)
    with (output / "predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    metrics["count"] = len(predictions)
    metrics["model_path"] = str(model_path)
    metrics["weight_sha256"] = {
        p.name: sha256(p) for p in sorted(model_path.glob("*.safetensors"))
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def cv(args: argparse.Namespace) -> None:
    config = json.loads(CONFIG.read_text())
    rows = read_jsonl(Path(config["training_data"]["path"]))
    folds = json.loads((ROOT / config["cross_validation"]["fold_file"]).read_text())["folds"]
    results: dict[str, dict] = {"2": {}, "3": {}}
    for fold in folds:
        fold_id = fold["fold"]
        valid = set(fold["validation_claim_ids"])
        validation = [row for row in rows if row["claim_id"] in valid]
        for epoch in (2, 3):
            path = CKPT / f"cv/fold-{fold_id}/epoch-{epoch}/model"
            output = RUN / f"cv/fold-{fold_id}/epoch-{epoch}/evaluation"
            metrics = evaluate_checkpoint(path, validation, output)
            results[str(epoch)][str(fold_id)] = metrics
            print(json.dumps({"fold": fold_id, "epoch": epoch, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}))
    candidates = {}
    for epoch in (2, 3):
        values = list(results[str(epoch)].values())
        candidates[str(epoch)] = {
            "fold_accuracy": [x["accuracy"] for x in values],
            "mean_accuracy": statistics.fmean(x["accuracy"] for x in values),
            "fold_macro_f1": [x["macro_f1"] for x in values],
            "mean_macro_f1": statistics.fmean(x["macro_f1"] for x in values),
        }
    summary = {"status": "complete_before_DEV", "candidates": candidates, "finalists": [2, 3], "benchmark_loaded": False}
    (RUN / "cv-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finalist = {
        "status": "frozen_after_CV_before_DEV",
        "milestone": "M3_full_parameter_fine_tuning",
        "training_config": str(CONFIG.relative_to(ROOT)),
        "training_config_sha256": sha256(CONFIG),
        "cv_folds": config["cross_validation"]["fold_file"],
        "cv_folds_sha256": sha256(ROOT / config["cross_validation"]["fold_file"]),
        "training_data_sha256": config["training_data"]["sha256"],
        "cv_summary": str(RUN / "cv-summary.json"),
        "cv_summary_sha256": sha256(RUN / "cv-summary.json"),
        "finalists": [{"epoch": e, **candidates[str(e)]} for e in (2, 3)],
        "final_seeds": [0, 1, 2],
        "selection": config["final_training"]["selection"],
        "inferential_seed": 0,
        "fixed_benchmark_access_allowed": False,
    }
    path = ROOT / "configs/posttraining/m3-cv-finalists-v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(finalist, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def dev_summary(_: argparse.Namespace) -> None:
    all_results = {}
    for epoch in (2, 3):
        seeds = []
        for seed in (0, 1, 2):
            path = RUN / f"dev-finalists/seed-{seed}/epoch-{epoch}/dev-evaluation/metrics.json"
            value = json.loads(path.read_text())
            seeds.append({"seed": seed, **value})
        aggregate = {}
        for metric in ("accuracy", "macro_f1"):
            xs = [x[metric] for x in seeds]
            aggregate[metric] = {"mean": statistics.fmean(xs), "sample_sd": statistics.stdev(xs)}
        for label in ("SUPPORT", "CONTRADICT", "INSUFFICIENT"):
            for metric in ("f1", "recall"):
                xs = [x["per_label"][label][metric] for x in seeds]
                aggregate[f"{label.lower()}_{metric}"] = {"mean": statistics.fmean(xs), "sample_sd": statistics.stdev(xs)}
        all_results[str(epoch)] = {"seeds": seeds, "mean_sample_sd": aggregate}
    best = max((2, 3), key=lambda e: all_results[str(e)]["mean_sample_sd"]["macro_f1"]["mean"])
    other = 2 if best == 3 else 3
    difference = all_results[str(best)]["mean_sample_sd"]["macro_f1"]["mean"] - all_results[str(other)]["mean_sample_sd"]["macro_f1"]["mean"]
    if best == 3 and difference <= 0.005:
        best = 2
    payload = {"status": "complete", "results": all_results, "selected_epoch": best, "selection_rule": "highest three-seed mean DEV macro F1; differences <= 0.005 choose fewer epochs"}
    (RUN / "dev-summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    model = CKPT / f"dev-finalists/seed-0/epoch-{best}/model"
    finalist = {
        "status": "frozen_after_three_seed_DEV_selection_before_benchmark",
        "milestone": "M3_full_parameter_fine_tuning",
        "selected_epochs": best,
        "three_seed_dev": all_results,
        "selection_metric": "best three-seed mean DEV Macro F1",
        "tie_margin": 0.005,
        "tie_break": ["fewer_epochs"],
        "inferential_seed": 0,
        "inferential_checkpoint": str(model),
        "inferential_weight_sha256": {p.name: sha256(p) for p in sorted(model.glob("*.safetensors"))},
        "cv_finalists_config": "configs/posttraining/m3-cv-finalists-v1.json",
        "cv_finalists_config_sha256": sha256(ROOT / "configs/posttraining/m3-cv-finalists-v1.json"),
        "training_config_sha256": sha256(CONFIG),
        "mixture_data_sha256": json.loads(CONFIG.read_text())["training_data"]["sha256"],
        "fixed_benchmark_access_allowed": True,
    }
    path = ROOT / "configs/posttraining/m3-finalist-v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(finalist, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("cv", "dev-summary"))
    args = parser.parse_args()
    cv(args) if args.phase == "cv" else dev_summary(args)


if __name__ == "__main__":
    main()
