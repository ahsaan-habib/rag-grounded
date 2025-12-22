"""Tests run offline: a fake embedder/reranker stands in for
sentence-transformers and the index lives in a temp dir. The one test that
talks to a real model is in test_smoke_ollama.py and is opt-in."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("RAG_INDEX_DIR", tempfile.mkdtemp(prefix="rag-test-"))

import fake_st  # noqa: E402

fake_st.install()
