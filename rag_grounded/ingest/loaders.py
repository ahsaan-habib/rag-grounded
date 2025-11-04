"""Load source documents into a normalised (source, section, text) form.

Markdown is the main input (the Laravel/Filament/Livewire docs are all .md in
their repos). HTML and PDF are supported because people keep asking for them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Section:
    source: str      # path relative to the corpus root
    heading: str     # nearest heading, "" if none
    text: str


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FRONT_MATTER = re.compile(r"^---\n.*?\n---\n", re.S)


def load_markdown(path: Path, root: Path) -> list[Section]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = _FRONT_MATTER.sub("", raw)
    rel = str(path.relative_to(root))

    sections: list[Section] = []
    heading, buf = "", []
    in_code = False
    for line in raw.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else _HEADING.match(line)
        if m:
            if buf:
                sections.append(Section(rel, heading, "\n".join(buf).strip()))
            heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append(Section(rel, heading, "\n".join(buf).strip()))
    return [s for s in sections if s.text]


def load_html(path: Path, root: Path) -> list[Section]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    rel = str(path.relative_to(root))

    sections: list[Section] = []
    heading, buf = "", []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li", "td"]):
        if el.name in {"h1", "h2", "h3", "h4"}:
            if buf:
                sections.append(Section(rel, heading, "\n".join(buf)))
            heading, buf = el.get_text(" ", strip=True), []
        else:
            text = el.get_text(" ", strip=True)
            if text:
                buf.append(text)
    if buf:
        sections.append(Section(rel, heading, "\n".join(buf)))
    return sections


def load_pdf(path: Path, root: Path) -> list[Section]:
    from pypdf import PdfReader

    rel = str(path.relative_to(root))
    reader = PdfReader(str(path))
    return [
        Section(rel, f"page {i + 1}", text.strip())
        for i, page in enumerate(reader.pages)
        if (text := page.extract_text() or "").strip()
    ]


LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}


def load_corpus(root: Path) -> list[Section]:
    root = root.resolve()
    out: list[Section] = []
    for path in sorted(root.rglob("*")):
        loader = LOADERS.get(path.suffix.lower())
        if loader and path.is_file():
            out.extend(loader(path, root))
    return out
