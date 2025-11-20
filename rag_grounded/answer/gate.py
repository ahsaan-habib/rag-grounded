"""answer/gate.py — a refusal is a feature, not an error path.

A claim is supported when some sentence in some retrieved chunk is close to it
in embedding space. That's a blunt check: it misses paraphrase occasionally and
lets through claims about an *adjacent* feature. It is still far better than
trusting the model's own [n] markers, which it happily attaches to anything.
"""
from __future__ import annotations

import re

import numpy as np

from ..embed import embed_docs
from ..ingest.chunker import Chunk
from .claims import strip_citations

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(chunk: Chunk) -> list[str]:
    return [s for s in _SENT.split(chunk.text) if len(s.split()) >= 3]


class CitationGate:
    def __init__(self, threshold: float = 0.72):
        self.threshold = threshold

    def unsupported(self, claims: list[str], chunks: list[Chunk]) -> list[str]:
        if not claims:
            return []
        sents = [s for c in chunks for s in _sentences(c)]
        if not sents:
            return list(claims)
        claim_vecs = np.array(embed_docs([strip_citations(c) for c in claims]))
        sent_vecs = np.array(embed_docs(sents))
        best = (claim_vecs @ sent_vecs.T).max(axis=1)  # vectors are normalised
        return [c for c, score in zip(claims, best) if score < self.threshold]
