import hashlib
import json
from pathlib import Path
from fastapi.testclient import TestClient
from retrieval.verification import prompt_hash
from serving.app import create_app

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/serving-6b-v1.json").read_text())


class FakeIndex:
    ntotal = 5183


class FakeRetriever:
    index = FakeIndex()

    def __init__(self):
        self.calls = []

    def retrieve(self, claim, k=1):
        self.calls.append((claim, k))
        documents = [{
            "document_id": str(100 + i), "score": 0.9 - i / 10,
            "title": f"Title {i}", "abstract": [f"Sentence {i}."],
        } for i in range(k)]
        return documents, {"embedding": 1.0, "faiss": 0.1, "retrieval": 1.1}


class FakeVerifier:
    def __init__(self, valid=True, invented=False):
        self.valid = valid
        self.invented = invented

    async def reachable(self): return True
    async def close(self): pass

    def build_prompt(self, claim, document):
        return "frozen rendered prompt", {"prompt_tokens": 20, "evidence": {"truncated": False}}

    async def generate(self, prompt, allowed_ids):
        if self.valid:
            evidence = ["999"] if self.invented else allowed_ids
            return {
                "raw_output": "{}", "parsed": {
                    "label": "SUPPORT", "evidence_ids": evidence, "explanation": "Supported."
                }, "errors": [], "input_tokens": 20, "output_tokens": 10, "generation_ms": 2.0,
            }
        return {
            "raw_output": "not json", "parsed": None, "errors": ["invalid_json:JSONDecodeError"],
            "input_tokens": 20, "output_tokens": 2, "generation_ms": 2.0,
        }


class MemoryLogger:
    def __init__(self): self.records = []
    def write(self, record): self.records.append(record)


def client(valid=True):
    retriever, logger = FakeRetriever(), MemoryLogger()
    app = create_app(retriever, FakeVerifier(valid=valid), logger)
    return TestClient(app), retriever, logger


def test_frozen_identifiers_and_prompt_hash():
    assert CONFIG["retriever"]["revision"] == "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    assert CONFIG["generator"]["revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert prompt_hash("grounded-v2") == CONFIG["generator"]["prompt_sha256"]
    assert CONFIG["retriever"]["production_top_k"] == 1


def test_health_endpoint():
    api, _, _ = client()
    with api:
        body = api.get("/health").json()
    assert body["status"] == "ok" and body["faiss_loaded"] and body["generator_reachable"]


def test_retrieve_and_deterministic_order():
    api, retriever, _ = client()
    with api:
        first = api.post("/retrieve", json={"claim": "a claim", "k": 2}).json()
        second = api.post("/retrieve", json={"claim": "a claim", "k": 2}).json()
    assert [x["document_id"] for x in first["documents"]] == ["100", "101"]
    assert first["documents"] == second["documents"]
    assert retriever.calls == [("a claim", 2), ("a claim", 2)]


def test_empty_claim_and_k_validation():
    api, _, _ = client()
    with api:
        assert api.post("/verify", json={"claim": "   "}).status_code == 422
        assert api.post("/retrieve", json={"claim": "x", "k": 11}).status_code == 422


def test_verify_is_frozen_top_one_and_has_timings():
    api, retriever, logger = client()
    with api:
        response = api.post("/verify", json={"claim": "a claim"})
    body = response.json()
    assert response.status_code == 200 and body["valid"]
    assert body["label"] == "SUPPORT" and body["evidence_ids"] == ["100"]
    assert retriever.calls == [("a claim", 1)]
    assert {"embedding", "faiss", "retrieval", "prompt", "generation", "total"} <= set(body["timing_ms"])
    assert logger.records[0]["parse_valid"] is True


def test_malformed_generation_is_not_repaired():
    api, _, logger = client(valid=False)
    with api:
        body = api.post("/verify", json={"claim": "a claim"}).json()
    assert body["valid"] is False and body["label"] is None
    assert body["errors"] and logger.records[0]["raw_output"] == "not json"


def test_no_quality_search_space_in_serving_config():
    assert CONFIG["quality_changes_allowed"] is False
    assert CONFIG["test_evaluation_allowed"] is False
    assert "candidates" not in json.dumps(CONFIG).lower()
