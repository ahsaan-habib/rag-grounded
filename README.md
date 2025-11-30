# rag-grounded

Retrieval-augmented QA over the Laravel 12, Filament and Livewire docs that
**cites every claim and refuses when it can't**.

Most RAG demos stop at "retrieve top-5, stuff the prompt, ask for citations".
That version answers almost every question — including the ones the docs don't
cover — with something well-formatted, plausible and wrong. This repo is about
the part after that: making "the docs don't cover this" a first-class answer.

```
question
   │
   ├── BM25 (identifier-aware) ─┐
   │                            ├─ RRF fusion ─▶ 30 candidates ─▶ cross-encoder rerank ─▶ top 6
   └── dense (bge-small)  ──────┘
                                                                         │
                                       prompts/answer.yaml (versioned) ──▶ LLM (qwen3:4b via Ollama)
                                                                         │
                                                        citation gate: every claim must match a
                                                        sentence in a retrieved chunk
                                                                         │
                                                     Answer + citations   or   Refusal + nearest chunks
```

## Pieces

| Step | Where | Notes |
|---|---|---|
| Load | `rag_grounded/ingest/loaders.py` | Markdown (front matter stripped, split on headings), HTML, PDF |
| Chunk | `rag_grounded/ingest/chunker.py` | ~650 token target, 800 max, 100 overlap; fenced code is never split |
| Keyword | `rag_grounded/store/bm25.py` | Keeps `withoutGlobalScopes`, `wire:model.live` intact and also indexes camelCase parts |
| Dense | `rag_grounded/store/vector.py` | Chroma, cosine, `BAAI/bge-small-en-v1.5` |
| Fuse | `rag_grounded/retrieval/fuse.py` | Reciprocal Rank Fusion, k=60 — no score normalisation needed |
| Rerank | `rag_grounded/retrieval/rerank.py` | `BAAI/bge-reranker-base` over the 30 fused candidates |
| Prompt | `prompts/*.yaml` | Loaded by id; version is recorded on every answer |
| Gate | `rag_grounded/answer/gate.py` | Claim ↔ chunk-sentence similarity; any unsupported claim → refusal |

Refusals come in three flavours (`no_context`, `model_declined`,
`unsupported_claims`) and always carry the two nearest chunks so the user
isn't left at a dead end.

## Run it

Needs Python 3.10+ and [Ollama](https://ollama.com) with the model pulled.

```bash
ollama pull qwen3:4b
make install && source .venv/bin/activate
make corpus     # sparse-clones laravel/docs, filament docs, livewire docs
make ingest     # chunks + embeds + builds bm25 next to chroma
rag ask "How do I disable a global scope for one query?" -v
make serve      # POST /ask {"question": "...", "debug": true}
```

Everything is configurable through `RAG_*` env vars, see `.env.example`.

## Why a local 4B model

The point of the gate is that the model is *not* trusted to stay grounded. That
makes a small local model a reasonable default: it's cheap to run on every eval
pass and the gate catches what it invents. Swap `RAG_LLM_MODEL` for anything
Ollama serves.

## Known gaps

- The gate checks similarity, not entailment. A claim about an *adjacent*
  feature (right concept, wrong version or wrong method) can pass.
- Token counts in the chunker are approximate (words × 1.3).
- No conversation memory — single-turn QA by design.

Evaluation lives in [rag-eval-gate](https://github.com/ahsaan-habib/rag-eval-gate),
tracing in [rag-observability](https://github.com/ahsaan-habib/rag-observability).
