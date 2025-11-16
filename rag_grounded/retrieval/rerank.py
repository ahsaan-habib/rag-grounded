"""Cross-encoder rerank. The expensive step, so it runs once on ~30 fused
candidates, never on the whole index."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from ..config import settings
from ..ingest.chunker import Chunk


@lru_cache(maxsize=1)
def _model(name: str) -> CrossEncoder:
    return CrossEncoder(name, max_length=512)


class Reranker:
    def __init__(self, model: str | None = None):
        self.name = model or settings.rerank_model

    def rank(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        scores = _model(self.name).predict([(query, c.text) for c in chunks])
        return sorted(zip(chunks, map(float, scores)), key=lambda x: x[1], reverse=True)
