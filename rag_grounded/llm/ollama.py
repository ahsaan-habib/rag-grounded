"""Thin Ollama client. No SDK, the HTTP API is small enough."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..config import settings


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class Ollama:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 timeout: float = 60.0):
        self.model = model or settings.llm_model
        self.http = httpx.Client(base_url=base_url or settings.ollama_url, timeout=timeout)

    def chat(self, messages: list[dict], temperature: float = 0.0,
             fmt: str | dict | None = None) -> Generation:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,  # qwen3: skip the <think> block, we want the answer
            "options": {"temperature": temperature},
        }
        if fmt is not None:
            body["format"] = fmt
        r = self.http.post("/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        return Generation(
            text=data["message"]["content"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            model=self.model,
        )
