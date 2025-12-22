import pytest

from rag_grounded.config import settings
from rag_grounded.ingest.chunker import Chunk
from rag_grounded.store.vector import IndexModelMismatch, VectorStore

CHUNKS = [
    Chunk("c1", "laravel/eloquent.md", "Global Scopes", "Remove a global scope with withoutGlobalScope on the query."),
    Chunk("c2", "laravel/queues.md", "Jobs", "Dispatch a job to the queue with the dispatch helper."),
]


def test_add_search_get_roundtrip(tmp_path):
    store = VectorStore(path=str(tmp_path), collection="test")
    store.add(CHUNKS)
    assert store.count() == 2
    top_id, score = store.search("remove global scope query", top_k=2)[0]
    assert top_id == "c1" and 0 < score <= 1.0001
    got = store.get(["c2", "missing", "c1"])
    assert [c.id for c in got] == ["c2", "c1"] and got[1].heading == "Global Scopes"
    ids, texts = store.all()
    assert sorted(ids) == ["c1", "c2"]


def test_reopening_with_a_different_embedding_model_fails_loudly(tmp_path, monkeypatch):
    VectorStore(path=str(tmp_path), collection="test").add(CHUNKS)
    VectorStore(path=str(tmp_path), collection="test")      # same model: fine
    monkeypatch.setattr(settings, "embed_model", "some/other-model")
    with pytest.raises(IndexModelMismatch):
        VectorStore(path=str(tmp_path), collection="test")
