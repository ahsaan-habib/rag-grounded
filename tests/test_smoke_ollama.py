"""End to end against a real local model. Opt-in, because it needs Ollama
running with the model pulled:

    RUN_OLLAMA=1 pytest tests/test_smoke_ollama.py -s
"""
import os

import pytest

from rag_grounded.config import settings
from rag_grounded.llm.ollama import Ollama

pytestmark = pytest.mark.skipif(os.environ.get("RUN_OLLAMA") != "1", reason="set RUN_OLLAMA=1 to run")


def test_model_answers_without_a_think_block():
    gen = Ollama().chat([{"role": "user", "content": "Reply with exactly the word: pong"}])
    assert "pong" in gen.text.lower() and "<think>" not in gen.text
    assert gen.output_tokens < 50, "model is reasoning out loud; use a non-thinking model"


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    from rag_grounded.ingest.loaders import Section
    from rag_grounded.ingest.chunker import chunk_sections
    from rag_grounded.pipeline import RAGPipeline
    from rag_grounded.retrieval.hybrid import HybridRetriever
    from rag_grounded.store.bm25 import BM25Index
    from rag_grounded.store.vector import VectorStore

    monkeypatch.setattr(settings, "index_dir", str(tmp_path))
    chunks = chunk_sections([
        Section("laravel/eloquent.md", "Removing Global Scopes",
                "If you would like to remove a global scope for a given query, you may use the "
                "withoutGlobalScope method. The method accepts the class name of the global scope "
                "as its only argument."),
        Section("laravel/queues.md", "Dispatching Jobs",
                "Once you have written your job class, you may dispatch it using the dispatch "
                "method on the job itself."),
    ])
    store = VectorStore()
    store.add(chunks)
    BM25Index(*store.all()).save()
    pipe = RAGPipeline(retriever=HybridRetriever(vectors=store))

    res = pipe.ask("How do I remove a global scope for one query?")
    print("\nanswerable ->", res.output.model_dump_json())
    off = pipe.ask("What is the capital of Australia?")
    print("off-topic  ->", off.output.model_dump_json())
    assert off.refused, "answered a question the docs don't cover"
