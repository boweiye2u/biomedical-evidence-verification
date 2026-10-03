from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class ClaimRequest(BaseModel):
    claim: str = Field(max_length=20000)
    k: int = Field(default=1, ge=1, le=10)

    @field_validator("claim")
    @classmethod
    def nonempty_claim(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim must not be empty")
        return value


class VerifyRequest(BaseModel):
    claim: str = Field(max_length=20000)

    @field_validator("claim")
    @classmethod
    def nonempty_claim(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim must not be empty")
        return value


class RetrievedDocument(BaseModel):
    document_id: str
    score: float
    title: str
    abstract: list[str]


class RetrievalResponse(BaseModel):
    claim: str
    documents: list[RetrievedDocument]
    timing_ms: dict[str, float]


class VerificationResponse(BaseModel):
    label: Literal["SUPPORT", "CONTRADICT", "INSUFFICIENT"] | None
    evidence_ids: list[str]
    explanation: str | None
    retrieval: dict[str, Any]
    timing_ms: dict[str, float]
    valid: bool
    errors: list[str] = []
    request_id: str
