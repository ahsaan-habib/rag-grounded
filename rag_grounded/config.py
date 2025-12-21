"""All knobs in one place. Override with env vars (RAG_*)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(f"RAG_{name}", default)


@dataclass
class Settings:
    index_dir: str = field(default_factory=lambda: _env("INDEX_DIR", ".chroma"))
    collection: str = field(default_factory=lambda: _env("COLLECTION", "docs"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-base"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3:4b-instruct"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "6")))


settings = Settings()
