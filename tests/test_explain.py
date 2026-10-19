"""engine/explain.py (plan 5.8): template mode and the opt-in teacher mode fill the same Report fields."""

import json

from engine.explain import EXPLAIN_FIELDS, explain_absences

CARD = {"id": "states_retention_period", "name": "states_retention_period", "definition": "States how long data is kept."}
ABS = {"type": "states_retention_period", "n": 200, "count": 130.0, "expected": 0.65,
       "evidence": [{"text": "We keep logs for 90 days.", "doc_id": "p1", "source": "cohort"},
                    {"text": "Data is deleted after one year.", "doc_id": "p2", "source": "cohort"},
                    {"text": "Prototype.", "doc_id": "lab", "source": "prototype"}]}
GENRE = {"document_kind": "privacy policy", "author_role": "the company"}


class FakeTeacher:
    def __init__(self):
        self.prompts = []

    def generate_structured_batch(self, prompts, model, **kw):
        self.prompts += prompts
        return [model(rationale="Peers state how long they keep data.", suggested_wording="We keep [data] for [period].")
                for _ in prompts]


def test_template_mode_fills_every_field():
    out = explain_absences([dict(ABS)], {CARD["id"]: CARD}, mode="template")
    a = out[0]
    assert set(EXPLAIN_FIELDS) <= set(a) and a["rationale"] is None and a["suggested_wording"] is None
    assert a["explanation"].startswith("65% of peers like this one (130 of 200)") and a["explanation_mode"] == "template"
    assert "“We keep logs for 90 days.” (p1)" in a["explanation"] and "Prototype" not in a["explanation"]


def test_teacher_mode_uses_only_the_quotes_for_the_top_n():
    t = FakeTeacher()
    absences = [dict(ABS, type=CARD["id"]) for _ in range(3)]
    out = explain_absences(absences, {CARD["id"]: CARD}, mode="teacher", teacher=t, genre=GENRE, top_n=2)
    assert len(t.prompts) == 2 and "We keep logs for 90 days. (p1)" in t.prompts[0] and "Prototype." not in t.prompts[0]
    assert "130 (65%)" in t.prompts[0] and "privacy policy" in t.prompts[0]
    assert out[0]["suggested_wording"] == "We keep [data] for [period]." and out[0]["explanation_mode"] == "teacher"
    assert out[2]["rationale"] is None and out[2]["explanation_mode"] == "template"  # beyond top_n: template only
    assert all(a["explanation"].startswith("65% of peers") for a in out)  # the evidence sentence stays the template
    json.dumps(out)


def test_teacher_mode_without_cohort_quotes_falls_back_to_template():
    t = FakeTeacher()
    a = dict(ABS, evidence=[ABS["evidence"][2]])  # prototypes only: nothing from peers to build on
    out = explain_absences([a], {CARD["id"]: CARD}, mode="teacher", teacher=t, genre=GENRE)
    assert t.prompts == [] and out[0]["explanation_mode"] == "template"
