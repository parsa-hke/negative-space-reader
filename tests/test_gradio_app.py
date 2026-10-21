"""app/gradio_app.py (plan 7.1): the Blocks app is built (not launched) on the small pack, and its handlers are
called directly: analyse with a named pack and with "auto", re-rank without embedding again, the evidence panel (from
the dropdown and from a ghost click), the chat-model comparison, examples, uploads (deleted once read), and the
wording rule of plan §0.2 #21."""

import json

import pytest

from test_pipeline import DOC, pack  # noqa: F401  (the small pack fixture)

ARGS = dict(k=20, cat_on=False, category="none", y0="any", y1="any", teach=False)


def app(pack, **kw):
    from app.gradio_app import build_app

    return build_app({"g": pack}, pack.cfg, **kw)


def run(demo, text=DOC, pack_name="g", meta=None, compare=False, **over):
    a = {**ARGS, **over}
    return demo.nsr_handlers["analyse"](text, pack_name, meta, compare, a["k"], a["cat_on"], a["category"], a["y0"], a["y1"], a["teach"])


def test_the_app_builds_and_analyses_with_ghosts_after_their_sections(pack):
    import gradio as gr

    from app.gradio_app import FORBIDDEN

    record = []
    demo = app(pack, record=record)
    assert isinstance(demo, gr.Blocks)
    st, doc, status, cohort, present, absence, evidence, *_ = run(demo)
    absences = st["report"]["absences"]
    assert doc.count("class='nsr-ghost'") == len(absences) > 0
    assert doc.index("<h3>Contact</h3>") < doc.index("nsr-ghost")  # section_hint "middle": no heading, so the last part
    assert "of peers like this one state" in doc and "this document does not" in doc and "data-type='retention'" in doc
    assert "closest peers among" in status and "kinds of statement" in status
    assert absence["choices"][0].startswith("#1 retention") and "keep" in evidence and "nsr-ring" in evidence
    assert "nearest peers" in cohort and "sharing" in present
    for page in (doc, status, cohort, evidence):
        assert not any(w in page.lower() for w in FORBIDDEN)
    assert record and record[0]["pack"] == "g" and set(record[0]) == {"pack", "seconds", "sections"}


def test_auto_detect_reports_the_scores_and_rerank_does_not_embed_again(pack):
    calls = []
    encode = pack.embedder.encode
    pack.embedder.encode = lambda texts, **kw: (calls.append(1) if any("share your data" in t for t in texts) else None) \
        or encode(texts, **kw)
    demo = app(pack)
    out = run(demo, pack_name="auto")
    assert out[0]["pack"] == "g" and "Auto-detect" in out[3] and len(calls) == 1
    before = len(calls)
    re = demo.nsr_handlers["rerank"](out[0], False, 10, False, "none", "any", "any", False)
    assert len(calls) == before and re[0]["report"]["cohort"]["n"] == 10 and "Re-ranked" in re[2]


def test_the_evidence_panel_follows_the_selected_absence(pack):
    demo = app(pack)
    st, *_ = run(demo)
    labels = [f"#{i + 1} x" for i in range(len(st["report"]["absences"]))]
    html = demo.nsr_handlers["absence"](st, labels[-1])
    last = st["report"]["absences"][-1]
    assert last["type"].replace("_", " ") in html and "peers</b>" in html
    assert "Analyse a document" in demo.nsr_handlers["absence"](None, None)
    label, html, tabs = demo.nsr_handlers["ghost"](st, f"{last['type']}|123")  # a click on the last ghost
    assert label["value"].endswith("of peers)") and last["type"].replace("_", " ") in html and tabs["selected"] == "evidence"


def test_examples_carry_their_entity_so_its_documents_are_not_peers(pack):
    ex = [{"name": "g: Our policy (q1)", "kind": "corpus", "pack": "g", "markdown": DOC, "entity_id": "own", "category": None}]
    demo = app(pack, examples=ex)
    text, meta, pack_name, _ = demo.nsr_handlers["example"]("g: Our policy (q1)")
    assert text == DOC and meta["entity_id"] == "own" and pack_name == "g" and "markdown" not in meta
    st, *_ = run(demo, text, pack_name, meta)
    assert st["entity_id"] == "own"
    assert all(n["doc_id"] not in ("p0", "p1", "p2") for n in st["report"]["cohort"]["nearest"])


def test_uploads_are_read_then_deleted_and_storing_them_is_refused(pack, tmp_path):
    f = tmp_path / "upload.html"
    f.write_text("<h1>Policy</h1><p>We share data with partners when asked.</p>")
    demo = app(pack)
    text, meta = demo.nsr_handlers["upload"](str(f))
    assert "We share data" in text and meta is None and not f.exists()
    pack.cfg["serve"]["store_uploads"] = True
    with pytest.raises(ValueError):
        app(pack)


def test_the_teacher_is_never_called_unless_allowed(pack):
    calls = []
    demo = app(pack, teacher_factory=lambda: calls.append(1))
    st, *_ = run(demo, teach=True)
    assert calls == [] and st["report"]["absences"][0]["explanation_mode"] == "template"


def test_a_category_of_another_pack_does_not_carry_over(pack):
    demo = app(pack)
    st, *_ = run(demo, cat_on=True, category="informationtech")  # left over from a privacy example
    assert st["report"]["cohort"]["fallback"] is False and st["report"]["cohort"]["filters"]["category"] is None


def test_the_comparison_shows_cached_chat_lists_for_a_benchmark_example_only(pack):
    from app.gradio_app import FORBIDDEN

    ex = [{"name": "Benchmark · g: q1", "kind": "benchmark", "pack": "g", "markdown": DOC, "entity_id": "own", "category": None,
           "instance_id": "single_type:q1", "removed": ["retention"],
           "chat": {"Opus": {"ranked": ["cookies", "retention"], "rank_first": 2}, "Qwen": {"ranked": ["cookies"], "rank_first": None}}}]
    demo = app(pack, examples=ex)
    text, meta, pack_name, _ = demo.nsr_handlers["example"]("Benchmark · g: q1")
    out = run(demo, text, pack_name, meta, compare=True)
    comp = out[-1]
    assert comp["visible"] is True
    h = comp["value"]
    assert h.count("nsr-col") >= 3 and "Opus" in h and "Qwen" in h and "first deleted type at rank 2" in h
    assert "none of the 1 deleted types" in h and "nsr-col ours" in h and "class='hit'" in h
    assert not any(w in h.lower() for w in FORBIDDEN)
    plain = run(demo, DOC, "g", None, compare=True)[-1]["value"]
    assert "benchmark examples" in plain and "off on this host" in plain
    doc_off, status_off, off = demo.nsr_handlers["compare"](out[0], False)
    assert off["visible"] is False and "Challenge." in status_off and "nsr-found" not in doc_off
    doc_on, status_on, on = demo.nsr_handlers["compare"](out[0], True)
    assert on["visible"] is True and "Reveal." in status_on and doc_on.count("nsr-found") == 1  # retention is ranked, and was removed
    assert demo.nsr_handlers["compare"](None, True)[2]["visible"] is True


def test_zerogpu_wraps_the_analyse_handler_when_asked(pack, monkeypatch):
    import sys
    import types

    calls = []

    def GPU(duration):
        def wrap(fn):
            calls.append(duration)
            return fn
        return wrap

    monkeypatch.setitem(sys.modules, "spaces", types.SimpleNamespace(GPU=GPU))
    pack.cfg["serve"].update(zerogpu=True, zerogpu_seconds=45)
    app(pack)
    assert calls == [45]
    monkeypatch.delitem(sys.modules, "spaces")
    monkeypatch.setattr("builtins.__import__", __import__)
