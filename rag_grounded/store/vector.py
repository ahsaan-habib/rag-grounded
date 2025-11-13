from __future__ import annotations

import chromadb

from ..config import settings
from ..embed import embed_docs, embed_query
from ..ingest.chunker import Chunk


class VectorStore:
    def __init__(self, path: str | None = None, collection: str | None = None):
        self.client = chromadb.PersistentClient(path=path or settings.index_dir)
        self.col = self.client.get_or_create_collection(
            collection or settings.collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], batch: int = 256) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            self.col.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                embeddings=embed_docs([c.text for c in part]),
                metadatas=[{"source": c.source, "heading": c.heading} for c in part],
            )

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        res = self.col.query(query_embeddings=[embed_query(query)], n_results=top_k)
        # chroma returns cosine *distance*
        return [(cid, 1.0 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]

    def get(self, ids: list[str]) -> list[Chunk]:
        res = self.col.get(ids=ids)
        by_id = {
            cid: Chunk(id=cid, source=m["source"], heading=m["heading"], text=doc)
            for cid, doc, m in zip(res["ids"], res["documents"], res["metadatas"])
        }
        return [by_id[i] for i in ids if i in by_id]

    def all(self) -> tuple[list[str], list[str]]:
        res = self.col.get(include=["documents"])
        return res["ids"], res["documents"]

    def count(self) -> int:
        return self.col.count()
