from pathlib import Path

from rag_grounded.ingest.chunker import approx_tokens, chunk_section, chunk_sections
from rag_grounded.ingest.loaders import Section, load_corpus, load_markdown

MD = """---
title: Eloquent
---
# Eloquent

Intro paragraph about models.

## Global Scopes

Scopes add constraints to every query.

```php
# not a heading, it's inside code
User::withoutGlobalScopes()->get();
```

## Soft Deleting

Use the SoftDeletes trait.
"""


def test_markdown_sections_skip_front_matter_and_code_headings(tmp_path: Path):
    (tmp_path / "laravel").mkdir()
    f = tmp_path / "laravel" / "eloquent.md"
    f.write_text(MD)
    sections = load_markdown(f, tmp_path)
    assert [s.heading for s in sections] == ["Eloquent", "Global Scopes", "Soft Deleting"]
    assert all(s.source == "laravel/eloquent.md" for s in sections)
    assert "title:" not in sections[0].text
    assert "# not a heading" in sections[1].text


def test_load_corpus_picks_loaders_by_suffix(tmp_path: Path):
    (tmp_path / "a.md").write_text("# A\n\nalpha text here")
    (tmp_path / "b.html").write_text("<h2>B</h2><p>beta text</p><script>x()</script>")
    (tmp_path / "c.txt").write_text("ignored")
    sections = load_corpus(tmp_path)
    assert {s.source for s in sections} == {"a.md", "b.html"}
    html = next(s for s in sections if s.source == "b.html")
    assert html.heading == "B" and "x()" not in html.text


def test_chunker_never_splits_a_code_block():
    code = "```php\n" + "\n".join(f"$x{i} = {i};" for i in range(400)) + "\n```"
    text = "intro " * 300 + "\n\n" + code + "\n\n" + "outro " * 300
    chunks = chunk_section(Section("a.md", "H", text), target=200, max_tokens=300, overlap=20)
    with_code = [c for c in chunks if "```php" in c.text]
    assert len(with_code) == 1 and with_code[0].text.count("```") == 2


def test_chunk_ids_include_source():
    s1, s2 = Section("a.md", "Installation", "same text " * 10), Section("b.md", "Installation", "same text " * 10)
    a, b = chunk_sections([s1, s2])
    assert a.id != b.id


def test_approx_tokens():
    assert approx_tokens("one two three four five six seven eight nine ten") == 13
