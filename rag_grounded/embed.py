from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import settings

# bge models want this prefix on the query side only
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=2)
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


CANARY = "Eloquent global scopes and withoutGlobalScopes in Laravel"


def fingerprint() -> str:
    """Name + a few rounded dimensions of a fixed sentence. If the model's
    weights change under the same name (unpinned dep, new revision), this
    changes too — the name alone wouldn't."""
    vec = embed_docs([CANARY])[0]
    return settings.embed_model + ":" + ",".join(f"{v:.3f}" for v in vec[:8])


def embed_docs(texts: list[str]) -> list[list[float]]:
    m = _model(settings.embed_model)
    return m.encode(texts, batch_size=32, normalize_embeddings=True,
                    show_progress_bar=len(texts) > 200).tolist()


def embed_query(text: str) -> list[float]:
    m = _model(settings.embed_model)
    return m.encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()
