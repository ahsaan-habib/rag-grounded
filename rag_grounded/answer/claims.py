from __future__ import annotations

import re

_CITE = re.compile(r"\[(\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"(])")
_CODE = re.compile(r"```.*?```", re.S)


def cited_numbers(text: str) -> list[int]:
    return [int(n) for n in _CITE.findall(text)]


def strip_citations(text: str) -> str:
    return _CITE.sub("", text).strip()


def split_claims(text: str, min_words: int = 5) -> list[str]:
    """Sentences that assert something. Code blocks, headings and short
    connective sentences ("Here's how:") aren't claims."""
    text = _CODE.sub(" ", text)
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if not line or line.startswith("#"):
            continue
        for sent in _SENT.split(line):
            if len(strip_citations(sent).split()) >= min_words:
                out.append(sent.strip())
    return out
