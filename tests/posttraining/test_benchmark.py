import json
from pathlib import Path

import numpy as np
import pytest

from posttraining.evaluation.benchmark import (
    bootstrap_macro,
    calibration_metrics,
    holm,
    rationale_metrics,
)


def test_holm_is_monotone_in_sorted_p_values():
    raw = [0.04, 0.001, 0.03, 0.2, 0.02]
    adjusted = holm(raw)
    ordered = sorted(range(len(raw)), key=lambda i: raw[i])
    assert all(0 <= value <= 1 for value in adjusted)
    assert [adjusted[i] for i in ordered] == sorted(adjusted[i] for i in ordered)
    assert adjusted[1] == pytest.approx(0.005)


def test_calibration_is_restricted_choice_and_agreement_is_separate():
    cal = [
        {"claim_id": "1", "gold_label": "SUPPORT", "restricted_probabilities": {"SUPPORT": .8, "CONTRADICT": .1, "INSUFFICIENT": .1}, "restricted_argmax": "SUPPORT"},
        {"claim_id": "2", "gold_label": "CONTRADICT", "restricted_probabilities": {"SUPPORT": .1, "CONTRADICT": .7, "INSUFFICIENT": .2}, "restricted_argmax": "CONTRADICT"},
    ]
    pred = [{"claim_id": "1", "predicted_label": "SUPPORT"}, {"claim_id": "2", "predicted_label": "INSUFFICIENT"}]
    result = calibration_metrics(cal, pred, 10)
    assert result["restricted_argmax_accuracy"] == 1
    assert result["restricted_generation_agreement_rate"] == .5
    assert result["nll"] > 0 and result["brier_score"] > 0


def test_rationale_maximum_over_valid_sets_and_exclusions():
    mapping = {
        "1": {"label": "SUPPORT", "annotated_evidence": {"10": [{"label": "SUPPORT", "sentence_ids": [1, 2]}, {"label": "SUPPORT", "sentence_ids": [4]}]}},
        "2": {"label": "INSUFFICIENT", "annotated_evidence": {}},
        "3": {"label": "CONTRADICT", "annotated_evidence": {"11": [{"label": "CONTRADICT", "sentence_ids": [0]}]}},
    }
    pred = [
        {"claim_id": "1", "document_id": "10", "predicted_rationale_sentences": [4]},
        {"claim_id": "2", "document_id": "10", "predicted_rationale_sentences": []},
        {"claim_id": "3", "document_id": "99", "predicted_rationale_sentences": [0]},
    ]
    result = rationale_metrics(pred, mapping)
    assert result["eligible_count"] == 1
    assert result["f1"] == result["precision"] == result["recall"] == 1
    assert result["exclusion_reasons"] == {"gold_insufficient": 1, "retrieved_document_without_valid_gold_rationale": 1}


def test_paired_bootstrap_is_deterministic_and_zero_for_equal_predictions():
    gold = ["SUPPORT", "CONTRADICT", "INSUFFICIENT"] * 4
    pred = gold[:]
    first = bootstrap_macro(gold, pred, pred, 100, 123)
    second = bootstrap_macro(gold, pred, pred, 100, 123)
    assert np.array_equal(first, second)
    assert np.all(first == 0)


def test_final_eval_config_is_frozen_and_primary_is_e1():
    config = json.loads(Path("configs/posttraining/final-eval-v1.json").read_text())
    assert config["status"] == "frozen_before_main_benchmark_predictions"
    assert config["condition"]["id"] == "E1" and config["condition"]["primary"] is True
    assert set(config["systems"]) == {"B0", "B1", "B2", "B3", "M1", "M2"}
    assert config["statistics"]["bootstrap_resamples"] == 10_000
    assert [(row["left"], row["right"]) for row in config["statistics"]["comparisons"]] == [("M1", "B2"), ("M2", "B3"), ("M2", "M1"), ("M2", "B0"), ("M2", "B1")]
