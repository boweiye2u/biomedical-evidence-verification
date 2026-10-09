"""Separate, one-shot frozen benchmark for M3 and comparison to saved M2."""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from posttraining.evaluation import benchmark
from retrieval.verification import classification_metrics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "configs/posttraining/m3-benchmark-v1.json"


def validate(config_path: Path, config: dict) -> tuple[list[dict], dict]:
    if config["status"] != "frozen_after_M3_DEV_selection_before_benchmark":
        raise ValueError("M3 benchmark was not frozen after DEV selection")
    if set(config["systems"]) != {"M3"}:
        raise ValueError("M3 benchmark runner is prohibited from generating any other system")
    finalist = ROOT / config["m3_finalist_config"]
    if benchmark.digest(finalist) != config["m3_finalist_config_sha256"]:
        raise ValueError("M3 finalist hash mismatch")
    frozen = json.loads(finalist.read_text())
    if frozen["status"] != "frozen_after_three_seed_DEV_selection_before_benchmark":
        raise ValueError("M3 selection is not frozen")
    m2_path = Path(config["saved_m2_predictions"])
    if benchmark.digest(m2_path) != config["saved_m2_predictions_sha256"]:
        raise ValueError("Saved M2 predictions changed")
    system = config["systems"]["M3"]
    if Path(system["model_path"]) != Path(frozen["inferential_checkpoint"]):
        raise ValueError("M3 checkpoint differs from frozen finalist")
    actual = {p.name: benchmark.digest(p) for p in sorted(Path(system["model_path"]).glob("*.safetensors"))}
    if actual != system["weight_sha256"] or actual != frozen["inferential_weight_sha256"]:
        raise ValueError("M3 checkpoint weight hash mismatch")
    for key in ("mapping", "rankings", "corpus"):
        if benchmark.digest(Path(config["inputs"][key])) != config["inputs"][f"{key}_sha256"]:
            raise ValueError(f"Frozen {key} hash mismatch")
    rows, _, _ = benchmark.load_inputs(config)
    m2 = benchmark.read_jsonl(m2_path)
    expected = [row["claim_id"] for row in rows]
    if [row["claim_id"] for row in m2] != expected:
        raise ValueError("Saved M2 order differs from benchmark")
    return rows, {row["claim_id"]: row for row in rows}


def preflight(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    run = Path(config["run_root"])
    if run.exists():
        raise FileExistsError(f"Refusing existing M3 benchmark run: {run}")
    rows, _ = validate(args.config, config)
    run.mkdir(parents=True)
    payload = {
        "status": "passed_before_M3_predictions", "count": len(rows),
        "generated_systems": ["M3"], "M2_predictions_reused": True,
        "main_benchmark_rerun": False, "config_sha256": benchmark.digest(args.config),
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
                        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
    }
    (run / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def finalize(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text())
    rows, mapping = validate(args.config, config)
    run = Path(config["run_root"]); output = run / "systems/M3"
    m3 = benchmark.read_jsonl(output / "predictions.jsonl")
    cal = benchmark.read_jsonl(output / "calibration.jsonl")
    m2 = benchmark.read_jsonl(Path(config["saved_m2_predictions"]))
    expected = [row["claim_id"] for row in rows]
    if [x["claim_id"] for x in m3] != expected or [x["claim_id"] for x in cal] != expected:
        raise ValueError("M3 record order mismatch")
    gold = [x["gold_label"] for x in m3]
    pred3 = [x["predicted_label"] for x in m3]; pred2 = [x["predicted_label"] for x in m2]
    metrics = classification_metrics(gold, pred3)
    metrics.update({
        "true_label_distribution": dict(Counter(gold)),
        "prediction_counts_including_invalid": dict(Counter(pred3)),
        "json_valid_rate": sum(x["json_valid"] for x in m3) / len(m3),
        "decision_valid_rate": sum(x["decision_valid"] for x in m3) / len(m3),
        "rationale_index_valid_rate": sum(x["rationale_index_valid"] for x in m3) / len(m3),
        "calibration": benchmark.calibration_metrics(cal, m3, config["calibration"]["ece_equal_width_bins"]),
        "rationale": benchmark.rationale_metrics(m3, mapping),
        "generation": json.loads((output / "generation-summary.json").read_text()),
    })
    m2_metrics = classification_metrics(gold, pred2)
    correct3 = [g == p for g, p in zip(gold, pred3)]; correct2 = [g == p for g, p in zip(gold, pred2)]
    m3_only = sum(a and not b for a, b in zip(correct3, correct2)); m2_only = sum(b and not a for a, b in zip(correct3, correct2))
    discordant = m3_only + m2_only
    p = binomtest(m3_only, discordant, 0.5, alternative="two-sided").pvalue if discordant else 1.0
    boot = benchmark.bootstrap_macro(gold, pred3, pred2, config["statistics"]["bootstrap_resamples"], config["statistics"]["bootstrap_seed"])
    comparison = {
        "id": 6, "left": "M3", "right": "M2", "outside_original_Holm_family": True,
        "accuracy_difference": metrics["accuracy"] - m2_metrics["accuracy"],
        "macro_f1_difference": metrics["macro_f1"] - m2_metrics["macro_f1"],
        "m3_correct_m2_wrong": m3_only, "m2_correct_m3_wrong": m2_only, "discordant": discordant,
        "mcnemar_exact_raw_p": p,
        "macro_f1_bootstrap_95_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "bootstrap_resamples": len(boot), "bootstrap_seed": config["statistics"]["bootstrap_seed"],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    stats = run / "statistics"; stats.mkdir()
    (stats / "test6.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (stats / "bootstrap-distribution.json").write_text(json.dumps(boot.tolist()) + "\n")
    result = {"status": "complete", "benchmark_role": config["benchmark_role"], "M3": metrics,
              "saved_M2": m2_metrics, "test6_M3_vs_M2": comparison,
              "integrity": {"claim_count": 300, "M2_regenerated": False, "main_benchmark_rerun": False,
                            "config_sha256": benchmark.digest(args.config)}}
    (run / "final-results.json").write_text(json.dumps(result, indent=2) + "\n")
    (ROOT / "docs/posttraining/milestone5-m3-benchmark-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"M3": {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, "test6": comparison}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("phase", choices=("preflight", "generate", "calibrate", "finalize"))
    p.add_argument("--config", type=Path, default=DEFAULT); p.add_argument("--base-url", default="http://127.0.0.1:8001"); p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "preflight": preflight(args)
    elif args.phase == "generate":
        args.system = "M3"; asyncio.run(benchmark.generate(args))
    elif args.phase == "calibrate":
        args.system = ["M3"]; benchmark.calibrate(args)
    else: finalize(args)
