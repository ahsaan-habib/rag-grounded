from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .answer.schemas import Answer, Refusal
from .pipeline import RAGPipeline

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipe = RAGPipeline()
    # A cold cross-encoder adds ~0.5s to the first request after startup —
    # the request someone is most likely to be judging. Pay it here instead.
    pipe.reranker.rank("warmup", pipe.retriever.vectors.get(pipe.retriever.retrieve("install"))[:2])
    state["pipe"] = pipe
    yield
    state.clear()


app = FastAPI(title="rag-grounded", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    debug: bool = False


class AskResponse(BaseModel):
    output: Answer | Refusal
    prompt_version: int
    timings_ms: dict[str, float]
    trace: dict | None = None


@app.get("/health")
def health() -> dict:
    pipe: RAGPipeline = state["pipe"]
    return {"ok": True, "chunks": pipe.retriever.vectors.count(),
            "model": pipe.llm.model, "prompt": f"{pipe.prompt.id}@v{pipe.prompt.version}"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    res = state["pipe"].ask(req.question)
    return AskResponse(
        output=res.output,
        prompt_version=res.trace.prompt_version,
        timings_ms=res.trace.timings_ms,
        trace=res.trace.model_dump() if req.debug else None,
    )
