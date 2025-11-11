"""Split sections into overlapping chunks.

Sizes are in *approximate* tokens (words * 1.3). Exact token counts depend on
the embedding model's tokenizer and the difference doesn't matter at this
granularity — what matters is never cutting a code block in half.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .loaders import Section

TOKENS_PER_WORD = 1.3


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str
    meta: dict = field(default_factory=dict)


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD)


def _blocks(text: str) -> list[str]:
    """Paragraphs, but a fenced code block always stays one block."""
    blocks, buf, in_code = [], [], False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        blocks.append("\n".join(buf))
    return blocks


def _tail(text: str, tokens: int) -> str:
    words = text.split()
    keep = int(tokens / TOKENS_PER_WORD)
    return " ".join(words[-keep:]) if keep else ""


def chunk_section(section: Section, target: int = 650, max_tokens: int = 800,
                  overlap: int = 100) -> list[Chunk]:
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for block in _blocks(section.text):
        n = approx_tokens(block)
        if current and size + n > max_tokens:
            pieces.append("\n\n".join(current))
            carry = _tail(pieces[-1], overlap)
            current, size = ([carry] if carry else []), approx_tokens(carry)
        current.append(block)
        size += n
        if size >= target:
            pieces.append("\n\n".join(current))
            carry = _tail(pieces[-1], overlap)
            current, size = ([carry] if carry else []), approx_tokens(carry)
    if current and (not pieces or approx_tokens("\n\n".join(current)) > overlap):
        pieces.append("\n\n".join(current))

    chunks = []
    for i, text in enumerate(pieces):
        # source must be in the id: "Installation" exists in ~40 files
        cid = hashlib.sha1(f"{section.source}:{section.heading}:{i}:{text[:200]}".encode()).hexdigest()[:16]
        chunks.append(Chunk(id=cid, source=section.source, heading=section.heading, text=text))
    return chunks


def chunk_sections(sections: list[Section], **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for s in sections:
        out.extend(chunk_section(s, **kw))
    return out


_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", text).strip()
