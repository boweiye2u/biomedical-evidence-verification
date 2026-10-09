import json

import pytest

from posttraining.evaluation.baselines import (
    LABELS,
    build_messages,
    parse_output,
    validate_config,
)


def base_config(fewshot=False):
    return {
        "scope": {"split": "frozen_DEV_only", "query_count": 162},
        "model": {"revision": "abc123", "tokenizer_revision": "def456"},
        "prompt": {
            "fewshot": fewshot,
            "sha256": __import__("posttraining.evaluation.baselines", fromlist=["prompt_sha256"]).prompt_sha256(),
        },
        "inference": {"max_model_len": 16384, "request_concurrency": 8},
    }


def split():
    return {"train_ids": ["1"] * 647, "dev_ids": ["2"] * 162}


def test_parser_separates_json_decision_and_rationale_validity():
    parsed = parse_output('{"decision":"SUPPORT","rationale_sentences":[9]}', 3)
    assert parsed["json_valid"] and parsed["schema_valid"] and parsed["decision_valid"]
    assert not parsed["rationale_index_valid"]
    assert parsed["decision"] == "SUPPORT"


@pytest.mark.parametrize("decision", LABELS)
def test_parser_accepts_all_decisions(decision):
    rationale = [] if decision == "INSUFFICIENT" else [0]
    parsed = parse_output(json.dumps({"decision": decision, "rationale_sentences": rationale}), 1)
    assert all(parsed[key] for key in ("json_valid", "schema_valid", "decision_valid", "rationale_index_valid"))


def test_invalid_json_has_invalid_decision():
    parsed = parse_output("not json", 3)
    assert not parsed["json_valid"] and not parsed["decision_valid"]


def test_fewshot_examples_must_be_train_only_and_cover_labels():
    cfg = base_config(True)
    frozen_split = {"train_ids": [str(i) for i in range(647)], "dev_ids": [str(i) for i in range(1000, 1162)]}
    examples = {
        "examples": [
            {"claim_id": "1", "decision": "SUPPORT"},
            {"claim_id": "2", "decision": "CONTRADICT"},
            {"claim_id": "3", "decision": "INSUFFICIENT"},
        ]
    }
    validate_config(cfg, frozen_split, examples)
    examples["examples"][2]["claim_id"] = "1000"
    with pytest.raises(ValueError, match="leak"):
        validate_config(cfg, frozen_split, examples)


def test_model_and_tokenizer_revisions_must_be_pinned():
    cfg = base_config()
    cfg["model"]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        validate_config(cfg, split(), None)


def test_build_messages_reuses_exact_frozen_examples():
    doc = {"doc_id": "10", "title": "T", "abstract": ["S"]}
    examples = [
        {"claim": "a", "document": doc, "decision": "SUPPORT", "rationale_sentences": [0]},
        {"claim": "b", "document": doc, "decision": "CONTRADICT", "rationale_sentences": [0]},
        {"claim": "c", "document": doc, "decision": "INSUFFICIENT", "rationale_sentences": []},
    ]
    messages = build_messages("target", doc, examples)
    assert [json.loads(row["content"])["decision"] for row in messages if row["role"] == "assistant"] == [
        "SUPPORT",
        "CONTRADICT",
        "INSUFFICIENT",
    ]


def test_frozen_repository_baseline_configs_share_exemplars_and_avoid_benchmark():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    split_payload = json.loads((root / "configs/splits/scifact_train_dev_v1.json").read_text())
    exemplar_payload = json.loads((root / "configs/posttraining/fewshot-exemplars-v1.json").read_text())
    train_ids = set(map(str, split_payload["train_ids"]))
    dev_ids = set(map(str, split_payload["dev_ids"]))
    exemplar_ids = {row["claim_id"] for row in exemplar_payload["examples"]}
    assert exemplar_ids <= train_ids
    assert not exemplar_ids & dev_ids
    assert exemplar_payload["benchmark_examples_used"] is False
    configs = {
        name: json.loads((root / "configs/posttraining" / name).read_text())
        for name in (
            "baseline-b0-7b-zero-dev-v2.json",
            "baseline-b1-7b-fewshot-v2.json",
            "baseline-b2-3b-zero-v2.json",
            "baseline-b3-3b-fewshot-v2.json",
        )
    }
    assert all(config["scope"] == {"split": "frozen_DEV_only", "query_count": 162, "benchmark_access_allowed": False} for config in configs.values())
    assert configs["baseline-b1-7b-fewshot-v2.json"]["prompt"]["exemplars"] == configs["baseline-b3-3b-fewshot-v2.json"]["prompt"]["exemplars"]
    assert all(len(config["model"]["revision"]) == 40 and len(config["model"]["tokenizer_revision"]) == 40 for config in configs.values())
