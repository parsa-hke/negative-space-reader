"""packs/schema_induction.py step 1 and experiments/induce.py eye-test stage (plan §3.1)."""

import json

import pytest

from data.schema import Section, with_char_spans, write_jsonl
from experiments import induce
from experiments.config import load_config
from models.teacher import PROMPTS, Teacher
from packs.schema_induction import (ExtractedAssertion, SectionExtraction, extract, extraction_checks,
                                    render_extract_prompt, select_eye_test, summarise_checks)


class FakeTeacher(Teacher):
    """Echoes the first sentence of the section as one grounded assertion, plus one ungrounded one."""

    backend = "fake"

    def __init__(self):
        super().__init__("fake-model", max_retries=0)
        self.prompts = []

    def _complete(self, conversations, *, json_schema, max_tokens):
        outs = []
        for conv in conversations:
            prompt = conv[-1]["content"]
            self.prompts.append(prompt)
            text = prompt.split('Section text:\n"""\n', 1)[1].split('\n"""', 1)[0]
            first = text.split(". ")[0].rstrip(".") + "."
            outs.append(json.dumps({"assertions": [
                {"claim": "The company states a practice.", "type_guess": "stated_practice", "span": first},
                {"claim": "We invent something.", "type_guess": "Bad Label", "span": "not in the text"}]}))
            self._served()
        return outs


def sec(doc, idx, text, heading=None):
    return Section(doc_id=doc, idx=idx, heading=heading, text=text)


TEXT = "We keep your data for two years after closing the account. We delete it after that period ends."


def test_prompt_renders_for_every_configured_genre():
    cfg = load_config("configs/induce.yaml")
    for genre, info in cfg["genre_info"].items():
        p = render_extract_prompt(sec("d", 0, TEXT, "Retention"), info, max_assertions=7)
        assert info["document_kind"] in p and info["author_role"] in p and TEXT in p and "At most 7" in p
        assert "{{" not in p


def test_extract_truncates_and_keeps_order():
    t = FakeTeacher()
    out = extract(t, [sec("d", i, f"Point {i} is here. More.") for i in range(5)],
                  {"document_kind": "policy", "author_role": "the company"}, max_assertions=1, max_tokens=50, batch_size=2)
    assert [len(x.assertions) for x in out] == [1] * 5
    assert [x.assertions[0].span for x in out] == [f"Point {i} is here." for i in range(5)]


def test_extraction_checks():
    ex = SectionExtraction(assertions=[
        ExtractedAssertion(claim="The company keeps data for a fixed period.", type_guess="retention_period",
                           span="keep   your data for two years"),
        ExtractedAssertion(claim="We delete data.", type_guess="Deletion", span="we erase everything"),
        ExtractedAssertion(claim="The company keeps data for a fixed period.", type_guess="retention_period",
                           span="We keep your data")])
    c = extraction_checks(TEXT, ex)
    assert c["grounded"] == [True, False, True] and c["grounded_spans"] == 2
    assert (c["snake_case_types"], c["first_person_claims"], c["duplicate_claims"]) == (2, 1, 1)
    assert (c["sentences"], c["sentences_covered"]) == (2, 1)
    s = summarise_checks([c, extraction_checks("Nothing.", SectionExtraction(assertions=[]))])
    assert s["empty_sections"] == 1 and s["span_grounding_rate"] == round(2 / 3, 3) and s["sentence_coverage"] == 0.333


def test_select_eye_test_is_deterministic_balanced_and_one_per_document():
    sections = [sec(f"doc{d}", i, "x" * (150 + 40 * i)) for d in range(30) for i in range(9)]
    dev = {f"doc{d}" for d in range(25)}
    a = select_eye_test(sections, dev, "g", n=12, seed=1, min_chars=200, max_chars=400)
    b = select_eye_test(sections, dev, "g", n=12, seed=1, min_chars=200, max_chars=400)
    assert a == b and len(a) == 12
    assert len({r.doc_id for r in a}) == 12 and {r.doc_id for r in a} <= dev
    assert [r.position for r in a].count("first") == 4 and [r.position for r in a].count("last") == 4
    by_key = {(s.doc_id, s.idx): s for s in sections}
    assert all(200 <= len(by_key[(r.doc_id, r.section_idx)].text) <= 400 for r in a)


@pytest.fixture
def corpus(tmp_path):
    processed = tmp_path / "processed"
    for genre in ("privacy", "papers"):
        secs = []
        for d in range(6):
            secs += with_char_spans([sec(f"{genre}:{d}", 0, f"Doc {d} first section. " + TEXT, "A"),
                                     sec(f"{genre}:{d}", 0, f"Doc {d} middle section. " + TEXT),
                                     sec(f"{genre}:{d}", 0, f"Doc {d} last section. " + TEXT, "C")])
        write_jsonl(processed / genre / "sections.jsonl", secs)
        manifest = {"doc_ids": {"train": [], "dev": [f"{genre}:{d}" for d in range(6)], "test": []}}
        (tmp_path / "splits").mkdir(exist_ok=True)
        (tmp_path / "splits" / f"{genre}.json").write_text(json.dumps(manifest))
    cfg = load_config("configs/induce.yaml")
    cfg["paths"].update(data_processed=str(processed), splits=str(tmp_path / "splits"))
    cfg["induce"]["eye_test"].update(sections_per_genre=4, selection=str(tmp_path / "sel.json"), min_chars=50)
    cfg["run"]["seed"] = 0
    return cfg, tmp_path


def test_eye_test_end_to_end_with_comparison(corpus, monkeypatch):
    cfg, tmp = corpus
    teacher = FakeTeacher()
    monkeypatch.setattr("models.teacher.load_teacher", lambda *a, **k: teacher)
    PROMPTS.reset_usage()
    run1 = tmp / "run1"
    run1.mkdir()
    m = induce.main(cfg, run1)
    assert m["model"] == cfg["models"]["teacher"]["id"] and m["prompt"].startswith("induce_extract v")
    assert m["privacy/sections"] == 4 and m["privacy/span_grounding_rate"] == 0.5
    assert PROMPTS.usage()["induce_extract"] == PROMPTS.get("induce_extract").tag
    rows = [json.loads(l) for l in open(run1 / "eye_test_papers.jsonl")]
    assert len(rows) == 4 and rows[0]["assertions"][0]["type_guess"] == "stated_practice"
    sel = json.loads((tmp / "sel.json").read_text())
    assert {r["genre"] for r in sel} == {"privacy", "papers"} and len(sel) == 8
    page = (run1 / "eye_test.html").read_text()
    assert page.count("<mark>") == 8 and "not in the text" in page

    (run1 / "metrics.json").write_text(json.dumps({"metrics": {"model": "small-model"}}))
    run2 = tmp / "run2"
    run2.mkdir()
    cfg["induce"]["eye_test"]["compare_with"] = str(run1)
    m2 = induce.main(cfg, run2)
    assert m2["compared_with"] == str(run1)
    assert json.loads((tmp / "sel.json").read_text()) == sel  # the same sections, not a new selection
    page2 = (run2 / "eye_test.html").read_text()
    assert "compared with: <b>small-model</b>" in page2 and page2.count("<table>") == 16


def test_eye_test_refuses_a_changed_corpus(corpus, monkeypatch):
    cfg, tmp = corpus
    monkeypatch.setattr("models.teacher.load_teacher", lambda *a, **k: FakeTeacher())
    (tmp / "r").mkdir()
    induce.main(cfg, tmp / "r")
    path = tmp / "processed" / "privacy" / "sections.jsonl"
    path.write_text(path.read_text().replace("Doc ", "Document "))
    with pytest.raises(RuntimeError, match="changed"):
        induce.eye_test_sections(cfg, "privacy")


class FlakyTeacher(FakeTeacher):
    """Returns invalid JSON for any section mentioning 'broken', so whole batches fail validation."""

    def _complete(self, conversations, *, json_schema, max_tokens):
        good = super()._complete(conversations, json_schema=json_schema, max_tokens=max_tokens)
        return ["{not json" if "broken" in c[-1]["content"] else g for c, g in zip(conversations, good)]


def test_extract_skips_failures_and_streams_batches():
    t = FlakyTeacher()
    secs = [sec("d", i, f"Point {i} is here. More." if i != 3 else "This one is broken. Sadly.") for i in range(6)]
    seen = []
    out = extract(t, secs, {"document_kind": "policy", "author_role": "the company"}, max_assertions=5, max_tokens=50,
                  batch_size=4, skip_failures=True, on_batch=lambda start, res: seen.append((start, len(res))))
    assert [x is None for x in out] == [False, False, False, True, False, False]
    assert seen == [(0, 4), (4, 2)]
    with pytest.raises(Exception):
        extract(FlakyTeacher(), secs, {"document_kind": "p", "author_role": "c"}, max_assertions=5, max_tokens=50)


def test_extract_stage_end_to_end(corpus, monkeypatch):
    from data.schema import Document

    cfg, tmp = corpus
    for genre in ("privacy", "papers"):
        write_jsonl(tmp / "processed" / genre / "documents.jsonl",
                    [Document(id=f"{genre}:{d}", genre=genre, source="s", entity_id=f"e{d}", date=f"20{10 + d}-01-01",
                              category="c") for d in range(6)])
    cfg["induce"]["stage"] = "extract"
    cfg["induce"]["extract"]["documents_per_genre"] = 4
    cfg["induce"]["batch_size"] = 5
    monkeypatch.setattr("models.teacher.load_teacher", lambda *a, **k: FakeTeacher())
    run = tmp / "x"
    run.mkdir()
    m = induce.main(cfg, run)
    assert m["privacy/documents"] == 4 and m["privacy/sections"] == 12 and m["privacy/claims"] == 24
    assert m["privacy/span_grounding_rate"] == 0.5 and m["privacy/failed_sections"] == 0 and m["route"] == "fake:fake-model"
    rows = [json.loads(l) for l in open(run / "claims_privacy.jsonl")]
    assert len(rows) == 24 and rows[0]["claim_id"].endswith("#0#0") and rows[0]["teacher"] == "fake:fake-model"
    assert rows[0]["prompt"].startswith("induce_extract v")
    assert json.loads((run / "sample_privacy.json").read_text()) == induce.sample_documents(cfg, "privacy")
    assert len([l for l in open(run / "sections_papers.jsonl")]) == 12
