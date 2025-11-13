"""Keyword index. Dense vectors miss exact identifiers like `withoutGlobalScopes`;
BM25 doesn't."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..config import settings

# keep identifiers intact: withoutGlobalScopes, wire:model.live, php artisan make:model
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_:.\-]*|\d+")


def tokenize(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(text):
        low = tok.lower()
        out.append(low)
        # also index camelCase parts so "global scope" finds withoutGlobalScopes
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", tok)
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return out


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.ids[i], float(scores[i])) for i in best if scores[i] > 0]

    @staticmethod
    def path() -> Path:
        return Path(settings.index_dir) / "bm25.pkl"

    def save(self) -> None:
        self.path().parent.mkdir(parents=True, exist_ok=True)
        with open(self.path(), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls) -> "BM25Index":
        with open(cls.path(), "rb") as f:
            return pickle.load(f)
