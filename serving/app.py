from __future__ import annotations
import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from .logging_utils import JsonlRequestLogger
from .retriever import FrozenRetriever
from .schemas import ClaimRequest, RetrievalResponse, VerificationResponse, VerifyRequest
from .verifier import FrozenVerifier

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    path = Path(os.environ.get("SERVING_6B_CONFIG", ROOT / "configs/serving-6b-v1.json"))
    config = json.loads(path.read_text())
    artifact_root = Path(os.environ.get(config.get("artifact_root_env", "RAG_ROOT"), Path.home() / "rag"))
    for section, fields in {
        "retriever": ("local_path", "embedding_cache", "corpus"),
        "generator": ("local_path",),
        "service": ("request_log",),
    }.items():
        for field in fields:
            value = Path(config[section][field]).expanduser()
            config[section][field] = str(value if value.is_absolute() else artifact_root / value)
    return config


def create_app(retriever=None, verifier=None, request_logger=None) -> FastAPI:
    config = load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = retriever or FrozenRetriever(config["retriever"])
        app.state.verifier = verifier or FrozenVerifier(config["generator"])
        app.state.request_logger = request_logger or JsonlRequestLogger(config["service"]["request_log"])
        app.state.retrieval_lock = asyncio.Lock()
        yield
        close = getattr(app.state.verifier, "close", None)
        if close:
            await close()

    app = FastAPI(title="Frozen SciFact Verification Service", version="6B-v1", lifespan=lifespan)

    @app.get("/health")
    async def health():
        generator = await app.state.verifier.reachable()
        return {
            "status": "ok" if generator else "degraded",
            "retriever_loaded": app.state.retriever is not None,
            "faiss_loaded": getattr(app.state.retriever, "index", None) is not None,
            "generator_reachable": generator,
            "models": {
                "retriever": config["retriever"]["model_name"],
                "retriever_revision": config["retriever"]["revision"],
                "generator": config["generator"]["model_name"],
                "generator_revision": config["generator"]["revision"],
            },
        }

    @app.post("/retrieve", response_model=RetrievalResponse)
    async def retrieve(request: ClaimRequest):
        started = time.perf_counter()
        async with app.state.retrieval_lock:
            documents, timing = await asyncio.to_thread(app.state.retriever.retrieve, request.claim, request.k)
        timing["total"] = (time.perf_counter() - started) * 1000
        return {"claim": request.claim, "documents": documents, "timing_ms": timing}

    @app.post("/verify", response_model=VerificationResponse)
    async def verify(request: VerifyRequest):
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        async with app.state.retrieval_lock:
            documents, retrieval_timing = await asyncio.to_thread(app.state.retriever.retrieve, request.claim, 1)
        prompt_started = time.perf_counter()
        prompt, prompt_info = app.state.verifier.build_prompt(request.claim, documents[0])
        prompt_ms = (time.perf_counter() - prompt_started) * 1000
        generation = await app.state.verifier.generate(prompt, [documents[0]["document_id"]])
        total_ms = (time.perf_counter() - started) * 1000
        parsed = generation["parsed"] or {}
        timing = {
            **retrieval_timing, "prompt": prompt_ms,
            "generation": generation["generation_ms"], "total": total_ms,
        }
        result = {
            "label": parsed.get("label"), "evidence_ids": parsed.get("evidence_ids", []),
            "explanation": parsed.get("explanation"),
            "retrieval": {"document_id": documents[0]["document_id"], "score": documents[0]["score"]},
            "timing_ms": timing, "valid": generation["parsed"] is not None,
            "errors": generation["errors"], "request_id": request_id,
        }
        app.state.request_logger.write({
            "request_id": request_id, "endpoint": "/verify", "claim_char_count": len(request.claim),
            "input_token_count": generation["input_tokens"] or prompt_info["prompt_tokens"],
            "output_token_count": generation["output_tokens"],
            "retrieved_document_id": documents[0]["document_id"], "timing_ms": timing,
            "status": "ok" if result["valid"] else "invalid_generation",
            "parse_valid": result["valid"], "parse_errors": result["errors"],
            "raw_output": generation["raw_output"],
        })
        return result

    return app


app = create_app()
