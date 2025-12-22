from rag_grounded.retrieval.fuse import reciprocal_rank_fusion
from rag_grounded.store.bm25 import BM25Index, tokenize


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert set(fused[:2]) == {"a", "b"} and fused[-1] in {"c", "d"}


def test_tokenize_keeps_identifiers_and_camel_parts():
    toks = tokenize("Call withoutGlobalScopes and wire:model.live")
    assert "withoutglobalscopes" in toks
    assert {"without", "global", "scopes"} <= set(toks)
    assert "wire:model.live" in toks


def test_bm25_finds_exact_identifier(tmp_path, monkeypatch):
    from rag_grounded.config import settings

    monkeypatch.setattr(settings, "index_dir", str(tmp_path))
    idx = BM25Index(["1", "2", "3"], ["queues and jobs", "use withoutGlobalScopes to drop scopes", "routing basics"])
    idx.save()
    hits = BM25Index.load().search("global scope")
    assert hits[0][0] == "2"
    assert BM25Index.load().search("zzz nothing") == []
