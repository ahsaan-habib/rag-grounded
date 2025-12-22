"""Prompts live next to this file as <id>.yaml and are loaded by id. They ship
inside the package, so a plain (non-editable) install finds them too.

A prompt edit is a behaviour change, so it goes through review like code and
the version travels with every answer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROMPT_DIR = Path(os.environ.get("RAG_PROMPT_DIR", Path(__file__).resolve().parent))


@dataclass(frozen=True)
class Prompt:
    id: str
    version: int
    system: str
    user: str

    def messages(self, **kw) -> list[dict]:
        return [
            {"role": "system", "content": self.system.strip()},
            {"role": "user", "content": self.user.format(**kw).strip()},
        ]


@lru_cache(maxsize=None)
def load(prompt_id: str) -> Prompt:
    data = yaml.safe_load((PROMPT_DIR / f"{prompt_id}.yaml").read_text())
    if data.get("id") != prompt_id:
        raise ValueError(f"{prompt_id}.yaml declares id={data.get('id')!r}")
    return Prompt(id=data["id"], version=int(data["version"]),
                  system=data["system"], user=data["user"])
