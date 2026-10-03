from __future__ import annotations
import time
import httpx
from transformers import AutoTokenizer
from retrieval.verification import build_messages, format_evidence, parse_and_validate_output, prompt_hash


class FrozenVerifier:
    def __init__(self, config: dict, timeout: float = 180.0):
        self.config = config
        if prompt_hash(config["prompt_name"]) != config["prompt_sha256"]:
            raise ValueError("frozen prompt hash mismatch")
        self.tokenizer = AutoTokenizer.from_pretrained(config["local_path"], local_files_only=True)
        self.client = httpx.AsyncClient(base_url=config["base_url"], timeout=timeout)

    async def reachable(self) -> bool:
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def build_prompt(self, claim: str, document: dict) -> tuple[str, dict]:
        evidence_document = {
            "doc_id": document["document_id"], "title": document["title"],
            "abstract": document["abstract"],
        }
        evidence, token_info = format_evidence(
            [evidence_document], self.tokenizer, self.config["max_evidence_tokens"]
        )
        messages = build_messages(claim, evidence, self.config["prompt_name"])
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_tokens > self.config["max_input_tokens"]:
            raise ValueError("rendered prompt exceeds frozen input-token limit")
        return prompt, {"prompt_tokens": prompt_tokens, "evidence": token_info}

    async def generate(self, prompt: str, allowed_ids: list[str]) -> dict:
        started = time.perf_counter()
        response = await self.client.post("/v1/completions", json={
            "model": self.config["served_model_name"], "prompt": prompt,
            "temperature": self.config["temperature"], "repetition_penalty": self.config["repetition_penalty"],
            "max_tokens": self.config["max_new_tokens"],
            "seed": self.config["seed"], "stream": False,
        })
        response.raise_for_status()
        payload = response.json()
        raw = payload["choices"][0]["text"].strip()
        parsed, errors = parse_and_validate_output(raw, allowed_ids)
        usage = payload.get("usage") or {}
        return {
            "raw_output": raw, "parsed": parsed, "errors": errors,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "generation_ms": (time.perf_counter() - started) * 1000,
        }

    async def close(self):
        await self.client.aclose()
