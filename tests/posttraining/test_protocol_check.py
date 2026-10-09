
import pytest

from posttraining.evaluation.reproduce_b0 import evaluate
from posttraining.protocol_check import claim_label, inspect_claims


def test_claim_label_preserves_pair_semantics():
    claim = {
        "id": 1,
        "evidence": {
            "10": [{"label": "SUPPORT", "sentences": [0]}],
            "11": [{"label": "CONTRADICT", "sentences": [1]}],
        },
    }
    label, pairs = claim_label(claim)
    assert label == "MIXED"
    assert pairs == {"10": "SUPPORT", "11": "CONTRADICT"}


def test_empty_evidence_is_insufficient():
    assert claim_label({"id": 2, "evidence": {}}) == ("INSUFFICIENT", {})


def test_pair_cannot_have_conflicting_labels():
    claim = {
        "id": 3,
        "evidence": {"10": [{"label": "SUPPORT"}, {"label": "CONTRADICT"}]},
    }
    with pytest.raises(ValueError, match="Conflicting labels within pair"):
        claim_label(claim)


def test_inspect_claims_reports_mixed_without_overwriting_labels():
    records = [
        {"id": 1, "evidence": {"10": [{"label": "SUPPORT"}]}},
        {"id": 2, "evidence": {}},
    ]
    result = inspect_claims(records, ["1", "2"])
    assert result["label_counts"] == {"INSUFFICIENT": 1, "SUPPORT": 1}
    assert result["mixed_claim_count"] == 0


def test_b0_evaluation_keeps_invalid_outputs_wrong():
    records = [
        {
            "gold_label": "SUPPORT",
            "predicted_label": "SUPPORT",
            "valid": True,
            "retrieved_document_id": "1",
            "expected_document_id": "1",
        },
        {
            "gold_label": "INSUFFICIENT",
            "predicted_label": "INVALID",
            "valid": False,
            "retrieved_document_id": "2",
            "expected_document_id": "3",
        },
    ]
    result = evaluate(records)
    assert result["accuracy"] == 0.5
    assert result["invalid_output_rate"] == 0.5
    assert result["retrieval_top1_match_rate"] == 0.5
