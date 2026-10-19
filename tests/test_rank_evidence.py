"""engine/rank.py and engine/evidence.py (plan 5.5; §0.2 #1, #13, #14)."""

import numpy as np

from engine.evidence import PeerEvidence, best_sections
from engine.rank import rank_absences

TYPES = ["a", "b", "c", "d"]


def test_score_is_expected_times_absent_times_weight_and_skips_present_types():
    expected = np.array([0.9, 0.8, 0.8, 0.3])
    p_present = np.array([0.95, 0.1, 0.1, 0.0])  # a is present in the document
    bits = np.array([1, 0, 0, 0])
    weights = np.array([1.0, 1.0, 1.5, 1.0])
    out = rank_absences(TYPES, expected, p_present, bits, weights=weights, top_k=10)
    assert [r["type"] for r in out] == ["c", "b", "d"]
    assert np.isclose(out[0]["score"], 0.8 * 0.9 * 1.5) and out[0]["weight"] == 1.5
    assert "a" not in [r["type"] for r in out]


def test_ties_broken_by_prevalence_threshold_and_top_k():
    expected = np.array([0.5, 0.5, 0.2, 0.1])
    out = rank_absences(TYPES, expected, np.zeros(4), np.zeros(4, dtype=int), prevalence=np.array([0.4, 0.6, 0.2, 0.1]), top_k=2)
    assert [r["type"] for r in out] == ["b", "a"]
    out = rank_absences(TYPES, expected, np.zeros(4), np.zeros(4, dtype=int), top_k=None, threshold=0.15)
    assert [r["type"] for r in out] == ["a", "b", "c"]


def test_best_sections_per_document_and_type():
    import pandas as pd

    sp = pd.DataFrame({"doc_id": ["p1", "p1", "p2"], "section_idx": [0, 1, 0],
                       "prob_a": [0.2, 0.9, 0.7], "prob_b": [0.6, 0.1, 0.3]})
    best = best_sections(sp, ["a", "b"])
    assert best["p1"].tolist() == [1, 0] and best["p2"].tolist() == [0, 0]


def test_peer_evidence_quotes_cohort_sentences_and_falls_back_to_prototypes():
    best = {"p1": np.array([1, 0, 0, 0]), "p2": np.array([0, 0, 0, 0]), "p3": np.array([0, 0, 0, 0])}
    texts = {("p1", 1): "Intro words. We keep data for 30 days. Contact us.", ("p2", 0): "Nothing here. Data is kept 1 year.",
             ("p3", 0): "We sell nothing. Ever, at all."}
    calls = []

    def score(groups, t):  # stands in for the presence model on sentences; one call per type (batched)
        calls.append(len(groups))
        return [np.array([0.9 if ("kept" in s or "keep" in s) else 0.1 for s in g]) for g in groups]

    protos = {"a": [{"text": "We retain data for a set time.", "doc_id": "opp:1", "section_idx": 2}] * 3}
    ev = PeerEvidence(best, lambda d, i: texts.get((d, i)), score, protos, TYPES)
    out = ev.for_type("a", ["p1", "p3", "p2"], k=3, min_prob=0.5)
    assert out[0] == {"doc_id": "p1", "section_idx": 1, "text": "We keep data for 30 days.", "source": "cohort", "p": 0.9}
    assert out[1]["text"] == "Data is kept 1 year." and out[1]["source"] == "cohort"  # p3 says nothing of the kind: skipped
    assert out[2]["source"] == "prototype" and out[2]["doc_id"] == "opp:1"  # filled up from the labelled set
    assert calls == [3]  # every candidate scored in one call
    assert len(ev.for_type("a", [], k=2)) == 2  # no peer states it: prototypes only


def test_stored_evidence_is_looked_up_thresholded_and_deduplicated(tmp_path):
    import pandas as pd

    from engine.evidence import EvidenceStore

    (tmp_path / "evidence").mkdir()
    pd.DataFrame([
        {"doc_id": "p1", "type": "a", "section_idx": 2, "sentence_idx": 0, "p": 0.9, "text": "We keep data 30 days."},
        {"doc_id": "p2", "type": "a", "section_idx": 1, "sentence_idx": 3, "p": 0.8, "text": "We keep data 30 days."},  # boilerplate
        {"doc_id": "p3", "type": "a", "section_idx": 0, "sentence_idx": 1, "p": 0.3, "text": "Weak sentence."},
        {"doc_id": "p4", "type": "a", "section_idx": 0, "sentence_idx": 2, "p": 0.7, "text": "Data is deleted after a year."},
        {"doc_id": "p1", "type": "b", "section_idx": 0, "sentence_idx": 0, "p": 0.6, "text": "We share data."},
    ]).to_parquet(tmp_path / "evidence" / "part-00000.parquet", index=False)
    store = EvidenceStore.load(tmp_path / "evidence", {"a": 0.5, "b": 0.7, "c": 0.5, "d": 0.5})
    assert store.get("a", "p3") is None and store.get("b", "p1") is None  # below each type's threshold
    protos = {"a": [{"text": "Prototype a.", "doc_id": "lab:1", "section_idx": 0}]}
    ev = PeerEvidence({}, lambda d, i: None, None, protos, TYPES, store=store)
    out = ev.for_type("a", ["p1", "p2", "p3", "p4"], k=3)
    assert [o["text"] for o in out] == ["We keep data 30 days.", "Data is deleted after a year.", "Prototype a."]
    assert out[0] == {"doc_id": "p1", "section_idx": 2, "sentence_idx": 0, "text": "We keep data 30 days.",
                      "source": "cohort", "p": 0.9}
