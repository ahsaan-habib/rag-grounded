from __future__ import annotations

import argparse
from pathlib import Path


def cmd_ingest(args) -> None:
    from .ingest.chunker import chunk_sections
    from .ingest.loaders import load_corpus
    from .store.bm25 import BM25Index
    from .store.vector import VectorStore

    sections = load_corpus(Path(args.path))
    chunks = chunk_sections(sections)
    store = VectorStore()
    store.add(chunks)
    # bm25 is rebuilt from the whole collection so it never drifts from chroma
    ids, texts = store.all()
    BM25Index(ids, texts).save()
    print(f"{len(sections)} sections -> {len(chunks)} chunks, index now {store.count()}")


def cmd_ask(args) -> None:
    from .pipeline import ask

    print(ask(args.question))


def main() -> None:
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(required=True)

    ing = sub.add_parser("ingest", help="load, chunk, embed and index a docs folder")
    ing.add_argument("path")
    ing.set_defaults(func=cmd_ingest)

    a = sub.add_parser("ask")
    a.add_argument("question")
    a.set_defaults(func=cmd_ask)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
