from rag_grounded.answer.claims import cited_numbers, split_claims, strip_citations
from rag_grounded.answer.gate import CitationGate
from rag_grounded.ingest.chunker import Chunk


def test_claims_skip_code_headings_and_short_sentences():
    text = ("## Answer\nHere's how:\nCall withoutGlobalScope on the query builder to remove it [1].\n"
            "```php\nUser::withoutGlobalScopes()->get(); // this is a long code comment line\n```\n"
            "- You can also pass an array of scope classes to remove several [2].")
    claims = split_claims(text)
    assert claims == ["Call withoutGlobalScope on the query builder to remove it [1].",
                      "You can also pass an array of scope classes to remove several [2]."]
    assert cited_numbers(text) == [1, 2]
    assert strip_citations("Done [1][2]") == "Done"


def test_gate_flags_only_the_invented_claim():
    chunk = Chunk("c1", "a.md", "Scopes",
                  "Call withoutGlobalScope on the query builder to remove a scope. Pass an array to remove several.")
    gate = CitationGate()
    supported = "Call withoutGlobalScope on the query builder to remove a scope [1]."
    invented = "Set the config key eloquent.disable_all_scopes to true in config/app.php [1]."
    assert gate.unsupported([supported, invented], [chunk]) == [invented]
    assert gate.unsupported([], [chunk]) == []
    assert gate.unsupported([supported], []) == [supported]
