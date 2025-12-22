import sys

from fastapi.testclient import TestClient

from rag_grounded import api, cli
from rag_grounded.answer.schemas import Answer, Citation, Result, Trace


class FakePipe:
    class llm:
        model = "fake-model"

    class prompt:
        id, version = "answer", 3

    class retriever:
        class vectors:
            @staticmethod
            def count():
                return 7

            @staticmethod
            def get(ids):
                return []

        @staticmethod
        def retrieve(q):
            return []

    class reranker:
        @staticmethod
        def rank(q, chunks):
            return []

    def ask(self, q):
        out = Answer(text="Use it [1].", citations=[Citation(n=1, chunk_id="c1", source="a.md", heading="H")])
        return Result(output=out, trace=Trace(query=q, prompt_version=3, timings_ms={"total": 1.0}))


def test_api_health_and_ask(monkeypatch):
    monkeypatch.setattr(api, "RAGPipeline", FakePipe)
    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"ok": True, "chunks": 7, "model": "fake-model", "prompt": "answer@v3"}
        r = client.post("/ask", json={"question": "q", "debug": True}).json()
        assert r["output"]["kind"] == "answer" and r["prompt_version"] == 3 and r["trace"]["query"] == "q"
        assert client.post("/ask", json={"question": "q"}).json()["trace"] is None


def test_cli_ingest_then_reset(tmp_path, monkeypatch, capsys):
    from rag_grounded.config import settings

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "eloquent.md").write_text("# Scopes\n\nRemove a global scope with withoutGlobalScope.\n")
    monkeypatch.setattr(settings, "index_dir", str(tmp_path / "idx"))
    for argv in (["rag", "ingest", str(corpus)], ["rag", "ingest", str(corpus), "--reset"]):
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
    out = capsys.readouterr().out
    assert "1 sections -> 1 chunks, index now 1" in out and "dropped collection" in out
    assert (tmp_path / "idx" / "bm25.pkl").exists()
