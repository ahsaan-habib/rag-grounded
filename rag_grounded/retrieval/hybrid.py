from __future__ import annotations

from ..store.bm25 import BM25Index
from ..store.vector import VectorStore
from .fuse import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self, vectors: VectorStore | None = None, bm25: BM25Index | None = None,
                 per_retriever: int = 20, candidates: int = 30):
        self.vectors = vectors or VectorStore()
        self.bm25 = bm25 or BM25Index.load()
        self.per_retriever = per_retriever
        self.candidates = candidates

    def retrieve(self, query: str) -> list[str]:
        keyword = [cid for cid, _ in self.bm25.search(query, self.per_retriever)]
        dense = [cid for cid, _ in self.vectors.search(query, self.per_retriever)]
        return reciprocal_rank_fusion([keyword, dense])[: self.candidates]
