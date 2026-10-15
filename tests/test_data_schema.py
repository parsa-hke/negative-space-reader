"""data/schema.py: records, character spans and jsonl I/O (plan §2.3)."""

import pytest

from data.schema import (Assertion, Document, ParsedDocument, RevisionPair, Section, Sentence, read_jsonl,
                         with_char_spans, write_jsonl)


def test_char_spans_index_the_joined_text():
    secs = with_char_spans([Section("d", 5, "A", "first"), Section("d", 9, None, "second part")])
    pd = ParsedDocument(Document(id="d", genre="g", source="s", entity_id="e"), secs)
    assert [s.idx for s in secs] == [0, 1]
    for s in secs:
        assert pd.text()[s.char_span[0]:s.char_span[1]] == s.text


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_jsonl_round_trip(tmp_path, suffix):
    records = {
        Document: [Document(id="d1", genre="privacy", source="s", entity_id="e.com", version="2019B", date="2019-07-01",
                            category="shopping", title="T", meta={"categories": ["a", "b"]})],
        Section: with_char_spans([Section("d1", 0, "H", "Text with ünïcödé.")]),
        Sentence: [Sentence("d1", 0, 0, "Text with ünïcödé.")],
        Assertion: [Assertion("d1", 0, "retention_period", 0.93, [0])],
        RevisionPair: [RevisionPair("e.com", "privacy", "d0", "d1", "2018-01-01", "2019-07-01", 0.21)],
    }
    for cls, recs in records.items():
        path = tmp_path / f"{cls.__name__}{suffix}"
        assert write_jsonl(path, recs) == len(recs)
        assert list(read_jsonl(path, cls)) == recs
    assert next(read_jsonl(tmp_path / f"Document{suffix}"))["meta"] == {"categories": ["a", "b"]}
