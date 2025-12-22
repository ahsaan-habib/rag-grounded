from rag_grounded.ingest.chunker import Chunk
from rag_grounded.llm.ollama import Generation
from rag_grounded.pipeline import REFUSAL, RAGPipeline
from rag_grounded.prompts import load

CHUNKS = {
    "c1": Chunk("c1", "laravel/eloquent.md", "Removing Global Scopes",
                "Call withoutGlobalScope on the query builder to remove a global scope from one query."),
    "c2": Chunk("c2", "laravel/queues.md", "Dispatching Jobs", "Dispatch a job with the dispatch helper."),
}


class FakeVectors:
    def get(self, ids):
        return [CHUNKS[i] for i in ids if i in CHUNKS]


class FakeRetriever:
    def __init__(self, ids):
        self.ids, self.vectors = ids, FakeVectors()

    def retrieve(self, query):
        return list(self.ids)


class FakeReranker:
    def rank(self, query, chunks):
        return [(c, 5.0 - i) for i, c in enumerate(chunks)]


class FakeLLM:
    model = "fake"

    def __init__(self, text):
        self.text, self.seen = text, None

    def chat(self, messages, **kw):
        self.seen = messages
        return Generation(self.text, 120, 30, "fake")


def pipe(text, ids=("c1", "c2"), **kw):
    return RAGPipeline(retriever=FakeRetriever(ids), reranker=FakeReranker(), llm=FakeLLM(text), **kw)


def test_grounded_answer_keeps_only_used_citations():
    steps = []
    p = pipe("Call withoutGlobalScope on the query builder to remove a global scope [1].",
             on_step=lambda name, payload, ms: steps.append(name))
    res = p.ask("how do I drop a global scope for one query?")
    assert res.output.kind == "answer"
    assert [c.chunk_id for c in res.output.citations] == ["c1"]
    assert steps == ["retrieve", "rerank", "generate", "gate"]
    assert res.trace.input_tokens == 120 and "total" in res.trace.timings_ms
    assert "[1] (laravel/eloquent.md" in p.llm.seen[1]["content"]


def test_model_declining_is_a_refusal_with_nearest_chunks():
    res = pipe("NOT_IN_DOCS").ask("what is the airspeed of a swallow?")
    assert res.refused and res.output.reason == "model_declined"
    assert res.output.message == REFUSAL and len(res.output.nearest) == 2


def test_invented_claim_is_refused():
    res = pipe("Enable the eloquent.scopes_off config flag inside config/database.php today [1].").ask("q")
    assert res.refused and res.output.reason == "unsupported_claims"
    assert res.output.unsupported_claims


def test_no_candidates_means_no_model_call():
    p = pipe("should not be called", ids=())
    res = p.ask("q")
    assert res.output.reason == "no_context" and p.llm.seen is None


def test_low_rerank_scores_are_dropped():
    res = pipe("x", min_rerank_score=10.0).ask("q")
    assert res.output.reason == "no_context"


def test_prompt_loads_with_version():
    p = load("answer")
    assert p.version >= 1
    msgs = p.messages(context="[1] ctx", question="why?")
    assert msgs[0]["role"] == "system" and "why?" in msgs[1]["content"]
