from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable

from .answer.claims import cited_numbers, split_claims
from .answer.gate import CitationGate
from .answer.schemas import Answer, Citation, Refusal, Result, Trace
from .config import settings
from .ingest.chunker import Chunk
from .llm.ollama import Ollama
from .prompts import load as load_prompt
from .retrieval.hybrid import HybridRetriever
from .retrieval.rerank import Reranker

REFUSAL = "The documentation I have does not cover this."
NOT_IN_DOCS = "NOT_IN_DOCS"

# on_step(name, payload, duration_ms) — lets tracing hook in without this
# module knowing anything about the tracer.
StepHook = Callable[[str, dict, float], None]


def _citations(chunks: list[Chunk]) -> list[Citation]:
    return [Citation(n=i, chunk_id=c.id, source=c.source, heading=c.heading)
            for i, c in enumerate(chunks, 1)]


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever | None = None,
                 reranker: Reranker | None = None, llm: Ollama | None = None,
                 gate: CitationGate | None = None, prompt_id: str = "answer",
                 top_k: int | None = None, min_rerank_score: float = -2.0,
                 on_step: StepHook | None = None):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.llm = llm or Ollama()
        self.gate = gate or CitationGate()
        self.prompt = load_prompt(prompt_id)
        self.top_k = top_k or settings.top_k
        self.min_rerank_score = min_rerank_score
        self.on_step = on_step

    @contextmanager
    def _step(self, trace: Trace, name: str, payload: dict):
        t0 = time.perf_counter()
        yield payload
        ms = (time.perf_counter() - t0) * 1000
        trace.timings_ms[name] = round(ms, 1)
        if self.on_step:
            self.on_step(name, payload, ms)

    def ask(self, query: str) -> Result:
        trace = Trace(query=query, model=self.llm.model, embed_model=settings.embed_model,
                      prompt_id=self.prompt.id, prompt_version=self.prompt.version)
        t_start = time.perf_counter()
        try:
            output = self._run(query, trace)
        finally:
            trace.timings_ms["total"] = round((time.perf_counter() - t_start) * 1000, 1)
        return Result(output=output, trace=trace)

    def _run(self, query: str, trace: Trace) -> Answer | Refusal:
        with self._step(trace, "retrieve", {"query": query}) as p:
            ids = self.retriever.retrieve(query)
            candidates = self.retriever.vectors.get(ids)
            trace.candidates = ids
            p["candidates"] = ids

        with self._step(trace, "rerank", {}) as p:
            ranked = self.reranker.rank(query, candidates)[: self.top_k]
            ranked = [(c, s) for c, s in ranked if s >= self.min_rerank_score]
            trace.reranked = [(c.id, round(s, 4)) for c, s in ranked]
            p["ranked"] = trace.reranked

        chunks = [c for c, _ in ranked]
        trace.contexts = [c.text for c in chunks]
        if not chunks:
            return Refusal(message=REFUSAL, reason="no_context")

        context = "\n\n".join(f"[{i}] ({c.source} — {c.heading})\n{c.text}"
                              for i, c in enumerate(chunks, 1))
        messages = self.prompt.messages(context=context, question=query)
        trace.messages = messages

        with self._step(trace, "generate", {"messages": messages}) as p:
            gen = self.llm.chat(messages)
            trace.raw_response = gen.text
            trace.input_tokens, trace.output_tokens = gen.input_tokens, gen.output_tokens
            p.update(response=gen.text, input_tokens=gen.input_tokens,
                     output_tokens=gen.output_tokens)

        nearest = _citations(chunks[:2])
        if NOT_IN_DOCS in gen.text:
            return Refusal(message=REFUSAL, reason="model_declined", nearest=nearest)

        with self._step(trace, "gate", {}) as p:
            unsupported = self.gate.unsupported(split_claims(gen.text), chunks)
            p["unsupported"] = unsupported

        if unsupported:
            return Refusal(message=REFUSAL, reason="unsupported_claims",
                           nearest=nearest, unsupported_claims=unsupported)

        used = set(cited_numbers(gen.text))
        cites = [c for c in _citations(chunks) if c.n in used]
        return Answer(text=gen.text.strip(), citations=cites)


def ask(query: str) -> Result:
    return RAGPipeline().ask(query)
