"""bench/synth.py: controlled deletions for OmitBench-synthetic (plan 6.1, §0.2 #15), verified by document presence.

The fake presence model: a section's probability of a type is the best probability among the sentences still in its
text, and a document states a type when its best section does (>= 0.5)."""

import numpy as np
import pytest

from bench.synth import BenchDoc, BenchSection, build_bench, delete_sentences, make_instance

TYPES = ["a", "b", "c", "d"]
THR = np.array([0.5, 0.5, 0.5, 0.5])


def section(idx, sents, S, heading=None):
    return BenchSection(idx=idx, heading=heading, text=" ".join(sents), sentences=sents, S=np.array(S, float),
                        P=np.array(S, float).max(0), V=np.ones(3) * (idx + 1))


def doc(doc_id="d1", entity="e1"):
    return BenchDoc(id=doc_id, entity=entity, title="T", category=None, date=None, title_vector=None, sections=[
        section(0, ["Alpha states a clearly here.", "Filler sentence number one.", "Alpha weakly hints at a."],
                [[0.9, 0, 0, 0.4], [0, 0, 0, 0], [0.55, 0, 0, 0]]),
        section(1, ["Beta states b and also d.", "Gamma states c in this sentence."], [[0, 0.8, 0, 0.7], [0, 0, 0.9, 0]]),
        section(2, ["Closing remarks that state nothing.", "Another sentence of filler text.",
                    "A third sentence of filler text.", "And a fourth one to close."], [[0, 0, 0, 0]] * 4),
    ])


def fake_scoring(d):
    sent_p = {s: row for sec in d.sections for s, row in zip(sec.sentences, sec.S)}

    def rescore(texts):
        P = np.zeros((len(texts), len(TYPES)))
        for i, t in enumerate(texts):
            rows = [p for s, p in sent_p.items() if s in t]
            P[i] = np.max(rows, axis=0) if rows else 0
        return P, np.ones((len(texts), 3))

    return rescore, lambda P: np.asarray(P).max(0)


def test_delete_sentences_keeps_the_rest_verbatim_and_tolerates_whitespace():
    text = "First one is here.\nSecond  one\nwraps lines. Third one stays."
    sents = ["First one is here.", "Second one wraps lines.", "Third one stays."]
    assert delete_sentences(text, sents, {1}) == "First one is here.\nThird one stays."
    assert delete_sentences(text, sents, {0, 2}) == "Second  one\nwraps lines."
    assert delete_sentences(text, sents, set()) == text
    # a sentence that cannot be located: the section is rebuilt from the kept sentences
    assert delete_sentences("Something else entirely.", ["Not there at all.", "Kept one."], {0}) == "Kept one."


def test_single_type_deletes_carrying_sentences_and_records_collateral():
    d = doc()
    rescore, doc_prob = fake_scoring(d)
    rng = np.random.default_rng(0)
    inst, reason = make_instance(d, "single_type", rng, types=TYPES, thresholds=THR, rescore=rescore,
                                 doc_prob=doc_prob, targets=["b"], min_section_chars=5)
    assert reason == "ok"
    assert inst["targets"] == ["b"] and set(inst["ground_truth"]) == {"b", "d"}  # d lived in the same sentence
    assert inst["collateral"] == ["d"] and inst["deleted"] == [[1, 0]] and inst["rounds"] == 0
    assert inst["deleted_sections"] == []
    texts = [s["text"] for s in inst["sections"]]
    assert "Beta states b" not in " ".join(texts) and "Gamma states c" in " ".join(texts)
    assert inst["p_after"]["b"] < 0.5 <= inst["p_before"]["b"]
    assert inst["P"].shape == (3, 4) and inst["doc_vector"].shape == (3,)


def test_escalates_until_the_type_is_absent_or_gives_up():
    d = doc()
    rescore, doc_prob = fake_scoring(d)
    # the weak sentence (0.55) keeps `a` present after the carrying one (0.9) goes: the threshold is halved once ...
    thr = np.array([0.6, 0.5, 0.5, 0.5])
    kw = dict(types=TYPES, thresholds=thr, rescore=rescore, doc_prob=doc_prob, targets=["a"], min_section_chars=5)
    inst, reason = make_instance(d, "single_type", np.random.default_rng(0), **kw)
    assert reason == "ok" and inst["deleted"] == [[0, 0], [0, 2]] and inst["rounds"] == 1
    assert inst["escalation"] == {"sentence_rounds": 1, "section_rounds": 0}
    # ... with no sentence rounds the section that still scores `a` highest is dropped whole ...
    inst, reason = make_instance(d, "single_type", np.random.default_rng(0), sentence_rounds=0, **kw)
    assert reason == "ok" and inst["deleted_sections"] == [0] and inst["escalation"]["section_rounds"] == 1
    # ... and with no escalation at all the instance is rejected
    inst, reason = make_instance(d, "single_type", np.random.default_rng(0), sentence_rounds=0, section_rounds=0, **kw)
    assert inst is None and reason == "still_present"


def test_rejects_edits_that_remove_too_much():
    d = doc()
    rescore, doc_prob = fake_scoring(d)
    inst, reason = make_instance(d, "whole_section", np.random.default_rng(0), types=TYPES, thresholds=THR,
                                 rescore=rescore, doc_prob=doc_prob, targets=["a"], max_removed_share=0.2)
    assert inst is None and reason == "too_much_removed"


def test_whole_section_drops_the_sections_carrying_the_type():
    d = doc()
    rescore, doc_prob = fake_scoring(d)
    inst, reason = make_instance(d, "whole_section", np.random.default_rng(0), types=TYPES, thresholds=THR,
                                 rescore=rescore, doc_prob=doc_prob, targets=["c"])
    assert reason == "ok" and inst["deleted_sections"] == [1] and inst["deleted"] == []
    assert set(inst["ground_truth"]) == {"b", "c", "d"} and [s["section_idx"] for s in inst["sections"]] == [0, 2]


def test_three_types_needs_three_present_types_and_samples_from_them():
    d = doc()
    rescore, doc_prob = fake_scoring(d)
    inst, reason = make_instance(d, "three_types", np.random.default_rng(3), types=TYPES, thresholds=THR,
                                 rescore=rescore, doc_prob=doc_prob, min_section_chars=5)
    assert reason == "ok" and len(inst["targets"]) == 3 and set(inst["targets"]) <= set(inst["ground_truth"])
    d.sections = d.sections[2:]  # nothing present
    inst, reason = make_instance(d, "three_types", np.random.default_rng(3), types=TYPES, thresholds=THR,
                                 rescore=rescore, doc_prob=doc_prob)
    assert inst is None and reason == "too_few_types"


def test_build_bench_draws_in_order_one_document_per_entity_and_is_reproducible():
    docs = [doc(f"d{i}", entity=f"e{i // 2}") for i in range(8)]  # two snapshots per entity
    rescore, doc_prob = fake_scoring(docs[0])
    kw = dict(types=TYPES, thresholds=THR, rescore=rescore, doc_prob=doc_prob, seed=5, min_section_chars=5)
    out, stats = build_bench(docs, ["single_type", "whole_section"], 3, **kw)
    for regime in ("single_type", "whole_section"):
        ids = [i["doc_id"] for i in out[regime]]
        assert ids == ["d0", "d2", "d4"]  # the second snapshot of an entity is skipped
        assert stats[regime]["ok"] == 3 and stats[regime]["same_entity"] >= 2
    again, _ = build_bench(docs, ["single_type", "whole_section"], 3, **kw)
    assert [i["targets"] for i in again["single_type"]] == [i["targets"] for i in out["single_type"]]
    with pytest.raises(ValueError, match="regime"):
        build_bench(docs, ["nope"], 1, **kw)


def test_rejected_draws_are_retried_with_new_targets():
    docs = [doc(f"d{i}", entity=f"e{i}") for i in range(12)]
    rescore, doc_prob = fake_scoring(docs[0])
    kw = dict(types=TYPES, thresholds=THR, rescore=rescore, doc_prob=doc_prob, seed=1)
    _, stats = build_bench(docs[:2], ["whole_section"], 5, max_removed_share=0.1, attempts_per_document=3, **kw)
    assert stats["whole_section"]["too_much_removed"] == 6  # every draw fails: three attempts per document
    # deleting section 0 (type a) removes ~30%, section 1 (b, c, d) ~21%: under a 25% cap only a's draws fail
    once, s1 = build_bench(docs, ["whole_section"], 12, max_removed_share=0.25, attempts_per_document=1, **kw)
    thrice, s3 = build_bench(docs, ["whole_section"], 12, max_removed_share=0.25, attempts_per_document=3, **kw)
    assert s1["whole_section"]["too_much_removed"] > 0 and len(thrice["whole_section"]) > len(once["whole_section"])
    assert all(i["targets"] != ["a"] for i in thrice["whole_section"])
