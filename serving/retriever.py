from __future__ import annotations
import json
import time
from pathlib import Path
import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class FrozenRetriever:
    def __init__(self, config: dict, device: str = "cuda:0"):
        self.config = config
        self.device = device
        model_source = config["local_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_source, local_files_only=True).to(device).eval()
        cache = Path(config["embedding_cache"])
        metadata = json.loads((cache / "metadata.json").read_text())
        self.document_ids = list(map(str, metadata["doc_ids"]))
        if len(self.document_ids) != config["corpus_size"]:
            raise ValueError("cached document ID count does not match frozen corpus size")
        self.index = faiss.read_index(str(cache / "index.faiss"))
        if self.index.ntotal != config["corpus_size"]:
            raise ValueError("FAISS index size does not match frozen corpus size")
        records = [json.loads(line) for line in Path(config["corpus"]).read_text().splitlines() if line.strip()]
        self.corpus = {str(record["doc_id"]): record for record in records}
        if set(self.document_ids) != set(self.corpus):
            raise ValueError("corpus and cached index IDs differ")

    @torch.inference_mode()
    def _encode(self, claim: str) -> np.ndarray:
        text = self.config["query_prefix"] + claim
        tokens = self.tokenizer(
            [text], padding=True, truncation=True,
            max_length=self.config["max_length"], return_tensors="pt"
        ).to(self.device)
        cls = self.model(**tokens).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, p=2, dim=1).cpu().numpy().astype("float32")

    def retrieve(self, claim: str, k: int = 1) -> tuple[list[dict], dict[str, float]]:
        started = time.perf_counter()
        encode_started = time.perf_counter()
        query = self._encode(claim)
        encode_ms = (time.perf_counter() - encode_started) * 1000
        search_started = time.perf_counter()
        scores, indices = self.index.search(query, k)
        search_ms = (time.perf_counter() - search_started) * 1000
        documents = []
        for score, index in zip(scores[0], indices[0]):
            doc_id = self.document_ids[int(index)]
            record = self.corpus[doc_id]
            documents.append({
                "document_id": doc_id, "score": float(score),
                "title": record["title"], "abstract": record["abstract"],
            })
        return documents, {
            "embedding": encode_ms, "faiss": search_ms,
            "retrieval": (time.perf_counter() - started) * 1000,
        }
