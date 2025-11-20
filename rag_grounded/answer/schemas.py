from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    n: int
    chunk_id: str
    source: str
    heading: str


class Answer(BaseModel):
    kind: Literal["answer"] = "answer"
    text: str
    citations: list[Citation]


class Refusal(BaseModel):
    kind: Literal["refusal"] = "refusal"
    message: str
    reason: Literal["no_context", "model_declined", "unsupported_claims"]
    nearest: list[Citation] = Field(default_factory=list)  # never a dead end
    unsupported_claims: list[str] = Field(default_factory=list)


class Trace(BaseModel):
    """Everything needed to reconstruct what a request saw."""
    query: str
    candidates: list[str] = Field(default_factory=list)
    reranked: list[tuple[str, float]] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    prompt_id: str = ""
    prompt_version: int = 0
    messages: list[dict] = Field(default_factory=list)
    raw_response: str = ""
    model: str = ""
    embed_model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)


class Result(BaseModel):
    output: Answer | Refusal
    trace: Trace

    @property
    def refused(self) -> bool:
        return self.output.kind == "refusal"
