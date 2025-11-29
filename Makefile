.PHONY: install corpus ingest serve ask clean

install:
	python -m venv .venv && .venv/bin/pip install -e .

corpus:
	./scripts/fetch_corpus.sh data/corpus

ingest:
	rag ingest data/corpus

serve:
	rag serve --port 8000

ask:
	@rag ask "$(Q)"

clean:
	rm -rf .chroma
