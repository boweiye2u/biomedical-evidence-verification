import json
from pathlib import Path

import pytest

from posttraining.data.build_sft_data import (
    build_candidates,
    claim_gold_label,
    filter_accepts,
    random_negative_rows,
    select_rationale,
    validate_cross_condition_pairs,
    validate_rows,
    wilson_interval,
)


def document(doc_id):
    return {"doc_id": int(doc_id), "title": f"title {doc_id}", "abstract": ["a", "b", "c"], "structured": False}


def claim(claim_id="1", evidence=None, cited=None):
    return {"id": int(claim_id), "claim": f"claim {claim_id}", "evidence": evidence or {}, "cited_doc_ids": cited or []}


def test_claim_label_and_mixed_guard_path():
    assert claim_gold_label(claim()) == "INSUFFICIENT"
    assert claim_gold_label(claim(evidence={"10": [{"label": "SUPPORT", "sentences": [0]}]})) == "SUPPORT"
    mixed = claim(evidence={
        "10": [{"label": "SUPPORT", "sentences": [0]}],
        "11": [{"label": "CONTRADICT", "sentences": [0]}],
    })
    assert claim_gold_label(mixed) == "MIXED"


def test_deterministic_rationale_keeps_shortest_then_lexicographic():
    label, chosen, alternatives = select_rationale([
        {"label": "SUPPORT", "sentences": [2, 3]},
        {"label": "SUPPORT", "sentences": [1]},
        {"label": "SUPPORT", "sentences": [0]},
    ])
    assert label == "SUPPORT" and chosen == [0]
    assert alternatives == [[0], [1], [2, 3]]


def test_candidate_pools_exclude_exact_annotated_pairs():
    claims = {"1": claim(evidence={"10": [{"label": "SUPPORT", "sentences": [0]}]})}
    corpus = {str(i): document(str(i)) for i in range(10, 21)}
    ranking = {"1": [(str(i), 1.0 - i / 100) for i in range(10, 20)]}
    b, c = build_candidates(claims, corpus, ["1"], ranking)
    assert not b
    assert {row["doc_id"] for row in c} == {str(i) for i in range(11, 20)}
    assert all(row["context_label"] is None and row["pair_annotation"] is None for row in c)


def test_random_negatives_are_reproducible_and_exclude_related_docs():
    claims = {"1": claim(evidence={"10": [{"label": "SUPPORT", "sentences": [0]}]}, cited=[10, 11])}
    corpus = {str(i): document(str(i)) for i in range(10, 30)}
    extra = {"1": {"12", "13"}}
    first = random_negative_rows(claims, corpus, ["1"], 17, extra)
    second = random_negative_rows(claims, corpus, ["1"], 17, extra)
    assert first == second
    assert first[0]["doc_id"] not in {"10", "11", "12", "13"}
    assert first[0]["context_label"] == "INSUFFICIENT"


def test_pool_rows_require_filter_approval():
    row = {
        "claim_id": "1", "doc_id": "10", "condition": "B_retrieved_non_gold",
        "source_split": "TRAIN", "claim": "c", "evidence": {},
        "claim_gold_label": "SUPPORT", "pair_annotation": None,
        "context_label": "INSUFFICIENT", "rationale_sentences": [],
        "annotation_basis": "x", "retrieval_rank": 1, "retrieval_score": 0.8,
        "filter_version": None, "filter_status": "pending",
    }
    with pytest.raises(ValueError, match="approval"):
        validate_rows([row], {"1"}, "B")
    row.update(filter_version="pool-b-frozen-v1", filter_status="approved")
    validate_rows([row], {"1"}, "B")


def test_nontrain_and_duplicate_pairs_rejected():
    base = {
        "claim_id": "2", "doc_id": "10", "condition": "D_random_negative",
        "source_split": "TRAIN", "claim": "c", "evidence": {},
        "claim_gold_label": "INSUFFICIENT", "pair_annotation": None,
        "context_label": "INSUFFICIENT", "rationale_sentences": [],
        "annotation_basis": "random", "retrieval_rank": None, "retrieval_score": None,
        "filter_version": None,
    }
    with pytest.raises(ValueError, match="Non-TRAIN"):
        validate_rows([base], {"1"}, "D")
    base["claim_id"] = "1"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows([base, dict(base)], {"1"}, "D")


def test_filter_and_interval_utilities():
    assert filter_accepts({"retrieval_score": 0.7}, {"type": "absolute_score_max", "max_score_inclusive": 0.7})
    assert not filter_accepts({"retrieval_score": 0.8}, {"type": "absolute_score_max", "max_score_inclusive": 0.7})
    assert filter_accepts({}, {"type": "no_filter"})
    assert not filter_accepts({}, {"type": "drop_all"})
    low, high = wilson_interval(5, 50)
    assert low < 0.1 < high


def test_frozen_config_is_train_only_and_audits_are_independent():
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / "configs/posttraining/data-audit-v1.json").read_text())
    assert cfg["scope"]["source_split"] == "TRAIN_only"
    assert cfg["scope"]["dev_or_benchmark_access_allowed"] is False
    assert cfg["audit"]["pool_b"]["filter_development_seed"] != cfg["audit"]["pool_c"]["filter_development_seed"]
    assert cfg["audit"]["pool_b"]["frozen_filter_audit_seed"] != cfg["audit"]["pool_c"]["frozen_filter_audit_seed"]
    source = (root / "posttraining/data/build_sft_data.py").read_text()
    assert "claims_dev.jsonl" not in source and "qrels/test.tsv" not in source and "test-claim-mapping" not in source


def test_cross_condition_duplicate_rejected():
    row = {"claim_id": "1", "doc_id": "10"}
    with pytest.raises(ValueError, match="both A and D"):
        validate_cross_condition_pairs({"A": [row], "D": [dict(row)]})
