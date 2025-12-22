"""rag-eval-gate, rag-observability and voice-agent-rt install this package
from git (not editable). The prompt YAML has to travel with the package or
every one of them fails on the first question."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_prompts_ship_inside_the_installed_package(tmp_path):
    try:
        import setuptools
    except ImportError:
        pytest.skip("setuptools not installed")
    if int(setuptools.__version__.split(".")[0]) < 68:
        pytest.skip("needs setuptools>=68 for an offline build")
    src = tmp_path / "src"   # build from a copy so build/ never lands in the repo
    shutil.copytree(ROOT, src, ignore=shutil.ignore_patterns(".git", ".venv", ".chroma", "data", "build", "*.egg-info"))
    target = tmp_path / "site"
    build = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation",
         "--target", str(target), str(src)],
        capture_output=True, text=True, cwd=tmp_path)
    assert build.returncode == 0, build.stderr
    probe = ("from rag_grounded.prompts import load, PROMPT_DIR; p = load('answer'); "
             "print(p.id, p.version, PROMPT_DIR)")
    env = {k: v for k, v in os.environ.items() if k != "RAG_PROMPT_DIR"}
    env["PYTHONPATH"] = str(target)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("answer ") and str(target) in out.stdout
