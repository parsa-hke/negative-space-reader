"""experiments/analyse.py (task analyse) and engine/explain.py unsupported_words: analyse named documents with a pack,
one Report JSON each in the run directory; in teacher mode, list the words of the teacher's text that neither the
quotes nor the type card contain (the check of plan 5.8: the wording stays within the quoted peer sentences)."""

import json

from test_pipeline import pack  # noqa: F401  (the small pack fixture)

from engine.explain import unsupported_words


def test_unsupported_words_flags_only_words_absent_from_the_sources():
    sources = ["We keep logs for 90 days.", "Data is deleted after one year."]
    assert unsupported_words("Peers keep logs and delete data after a year.", sources) == []  # inflections tolerated
    assert unsupported_words("We keep [data] for [period].", sources) == []                    # placeholders ignored
    assert unsupported_words("We keep logs as required by the GDPR.", sources) == ["required", "gdpr"]


def test_analyse_task_writes_reports_and_the_teacher_check(pack, tmp_path, monkeypatch):  # noqa: F811
    from data.schema import Document, Section, write_jsonl
    from experiments import analyse
    from packs.spec import GenrePack

    processed = tmp_path / "processed" / "g"
    processed.mkdir(parents=True)
    write_jsonl(processed / "documents.jsonl", [Document(id="q:1", genre="g", source="s", entity_id="qe", title="Policy")])
    write_jsonl(processed / "sections.jsonl", [Section(doc_id="q:1", idx=0, heading="Data",
                                                       text="We share your data with partners when you ask.")])

    class T:
        def generate_structured_batch(self, prompts, model, **kw):
            return [model(rationale="Peers say they keep things here.", suggested_wording="We keep [data] as the GDPR requires.")
                    for _ in prompts]

    import models.teacher
    monkeypatch.setattr(GenrePack, "load", classmethod(lambda cls, cfg, genre, **kw: pack))
    monkeypatch.setattr(models.teacher, "load_teacher", lambda tcfg: T())
    cfg = pack.cfg
    cfg["paths"]["data_processed"] = str(tmp_path / "processed")
    cfg["genre_info"]["g"] = {"document_kind": "test policy", "author_role": "the company"}
    cfg.update(pack="g", analyse={"doc_ids": ["q:1"]})
    cfg["explain"] = {"mode": "teacher", "top_n": 1, "max_tokens": 64}
    m = analyse.main(cfg, tmp_path / "run")

    rep = json.loads((tmp_path / "run" / "report_q_1.json").read_text())
    assert rep["absences"][0]["explanation_mode"] == "teacher" and rep["absences"][1]["explanation_mode"] == "template"
    assert m["documents"] == 1 and m["explain_mode"] == "teacher" and m["teacher_explanations"] == 1
    assert m["unsupported_words"] == {"q:1": {rep["absences"][0]["type"]: ["gdpr", "requires"]}}
    assert m["max_seconds"] > 0
