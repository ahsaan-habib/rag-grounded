from __future__ import annotations

from .config import settings
from .llm.ollama import Ollama
from .prompts import load as load_prompt
from .retrieval.hybrid import HybridRetriever
from .retrieval.rerank import Reranker

def ask(question: str) -> str:
    retriever = HybridRetriever()
    candidates = retriever.vectors.get(retriever.retrieve(question))
    chunks = [c for c, _ in Reranker().rank(question, candidates)[: settings.top_k]]

    context = "\n\n".join(
        f"[{i}] ({c.source} — {c.heading})\n{c.text}" for i, c in enumerate(chunks, 1)
    )
    gen = Ollama().chat(load_prompt("answer").messages(context=context, question=question))
    sources = "\n".join(f"[{i}] {c.source} — {c.heading}" for i, c in enumerate(chunks, 1))
    return f"{gen.text}\n\nSources:\n{sources}"
