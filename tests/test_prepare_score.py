"""engine/pipeline.py prepare / score (plan 7.1): prepare() does the slow part once per upload (segment, embed,
presence), score() the fast part (cohort, expectation, rank, evidence, explanation), so the app's cohort controls
re-rank without re-embedding. analyse() is score(prepare(..)): the two must give the same Report."""

import pytest

from test_pipeline import DOC, KEYS, pack  # noqa: F401  (the small pack fixture)


def corpus_markdown():
    from data.schema import Document, Section
    from engine.pipeline import document_markdown

    d = Document(id="q1", genre="g", source="s", entity_id="own", title="Our policy")
    secs = [Section(doc_id="q1", idx=0, heading="Sharing", text="We share your data with partners when you ask us to."),
            Section(doc_id="q1", idx=1, heading="Contact", text="Write to us at the address below for anything at all.")]
    return document_markdown(d, secs)


def without_timings(r):
    d = r.to_dict()
    d.pop("timings")
    return d


def test_prepare_then_score_equals_analyse_on_a_corpus_document(pack):
    from engine.pipeline import analyse, prepare, score

    md = corpus_markdown()
    whole = analyse(md, pack, entity_id="own")
    split = score(prepare(md, pack), pack, entity_id="own")
    assert without_timings(split) == without_timings(whole)
    assert {"segment", "embed", "presence", "cohort", "rank", "evidence", "total"} <= set(split.timings)
    assert [s[0] for s in prepare(md, pack).sections] == ["Sharing", "Contact"]


def test_score_reranks_with_other_cohort_settings_without_embedding_the_upload_again(pack):
    from engine.pipeline import prepare, score

    upload_calls = []
    encode = pack.embedder.encode

    def counting(texts, **kw):
        if any("share your data" in t for t in texts):
            upload_calls.append(len(texts))
        return encode(texts, **kw)

    pack.embedder.encode = counting
    prep = prepare(DOC, pack)
    a = score(prep, pack, k=10)
    b = score(prep, pack, k=30)
    assert len(upload_calls) == 1
    assert a.cohort["n"] == 10 and b.cohort["n"] == 30 and a.absences[0]["n"] == 10
    c = score(prep, pack, category_filter=True, category="nope")  # no peer has the category: falls back, flagged
    assert c.cohort["fallback"] is True


def test_score_applies_a_date_window(pack):
    from engine.pipeline import prepare, score

    prep = prepare(DOC, pack)
    assert score(prep, pack, date_window=("2021-01-01", None)).cohort["n"] == 0
    assert score(prep, pack, date_window=("2019-01-01", "2020-12-31")).cohort["n"] == 20


def test_prepare_refuses_empty_documents(pack):
    from engine.pipeline import prepare

    with pytest.raises(ValueError):
        prepare("  \n ", pack)
