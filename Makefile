.PHONY: install corpus ingest reindex serve ask test clean

install:
	python -m venv .venv && .venv/bin/pip install -e .

corpus:
	./scripts/fetch_corpus.sh data/corpus

ingest:
	rag ingest data/corpus

reindex:
	rag ingest data/corpus --reset

serve:
	rag serve --port 8000

ask:
	@rag ask "$(Q)"

test:
	pytest -q

clean:
	rm -rf .chroma
