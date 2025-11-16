from __future__ import annotations

from .config import settings
from .llm.ollama import Ollama
from .retrieval.hybrid import HybridRetriever
from .retrieval.rerank import Reranker

SYSTEM = """You answer questions about Laravel, Filament and Livewire using ONLY
the numbered context passages. Cite passages inline like [1] or [2][3] after
each sentence that uses them. If the passages do not contain the answer, say
you don't know."""


def ask(question: str) -> str:
    retriever = HybridRetriever()
    candidates = retriever.vectors.get(retriever.retrieve(question))
    chunks = [c for c, _ in Reranker().rank(question, candidates)[: settings.top_k]]

    context = "\n\n".join(
        f"[{i}] ({c.source} — {c.heading})\n{c.text}" for i, c in enumerate(chunks, 1)
    )
    gen = Ollama().chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ])
    sources = "\n".join(f"[{i}] {c.source} — {c.heading}" for i, c in enumerate(chunks, 1))
    return f"{gen.text}\n\nSources:\n{sources}"
