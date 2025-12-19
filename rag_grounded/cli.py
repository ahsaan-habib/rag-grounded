from __future__ import annotations

import argparse
from pathlib import Path


def cmd_ingest(args) -> None:
    from .ingest.chunker import chunk_sections
    from .ingest.loaders import load_corpus
    from .store.bm25 import BM25Index
    from .store.vector import VectorStore

    if args.reset:
        import chromadb

        from .config import settings

        client = chromadb.PersistentClient(path=settings.index_dir)
        if settings.collection in [c.name for c in client.list_collections()]:
            client.delete_collection(settings.collection)
            print(f"dropped collection {settings.collection!r}")

    sections = load_corpus(Path(args.path))
    chunks = chunk_sections(sections)
    store = VectorStore(check_model=not args.reset)
    store.add(chunks)
    # bm25 is rebuilt from the whole collection so it never drifts from chroma
    ids, texts = store.all()
    BM25Index(ids, texts).save()
    print(f"{len(sections)} sections -> {len(chunks)} chunks, index now {store.count()}")


def cmd_ask(args) -> None:
    from .pipeline import ask

    res = ask(args.question)
    out = res.output
    if out.kind == "refusal":
        print(out.message)
        if out.nearest:
            print("\nClosest things I found:")
            for c in out.nearest:
                print(f"  - {c.source} — {c.heading}")
    else:
        print(out.text)
        print("\nSources:")
        for c in out.citations:
            print(f"  [{c.n}] {c.source} — {c.heading}")
    if args.verbose:
        print("\n" + res.trace.model_dump_json(indent=2, exclude={"messages", "contexts"}))


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("rag_grounded.api:app", host=args.host, port=args.port)


def main() -> None:
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(required=True)

    ing = sub.add_parser("ingest", help="load, chunk, embed and index a docs folder")
    ing.add_argument("path")
    ing.add_argument("--reset", action="store_true", help="drop the collection first")
    ing.set_defaults(func=cmd_ingest)

    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(func=cmd_ask)

    srv = sub.add_parser("serve")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.set_defaults(func=cmd_serve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
