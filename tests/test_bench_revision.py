"""bench/revision.py: OmitBench-revision instances from version pairs (plan 6.3b, §0.2 #16, #19)."""

import numpy as np

from bench.revision import added_types, revision_instances

TYPES = ["a", "b", "c", "d"]


def test_added_types_are_present_in_the_new_version_and_absent_in_the_old():
    old, new = np.array([0.9, 0.2, 0.4, 0.1]), np.array([0.8, 0.7, 0.45, 0.6])
    assert added_types(old, new) == [1, 3]
    assert added_types(new, old) == []


def test_pairs_qualify_by_the_prevalence_of_an_added_type_among_earlier_peers():
    pairs = [{"entity_id": "e1", "old_doc_id": "o1", "new_doc_id": "n1", "old_date": "2017-03-01", "new_date": "2018-06-01"},
             {"entity_id": "e2", "old_doc_id": "o2", "new_doc_id": "n2", "old_date": "2019-01-01", "new_date": "2019-09-01"},
             {"entity_id": "e3", "old_doc_id": "o3", "new_doc_id": "missing", "old_date": "2019-01-01", "new_date": None}]
    p = {"o1": np.array([0.9, 0.1, 0.1, 0.1]), "n1": np.array([0.9, 0.8, 0.7, 0.1]),
         "o2": np.array([0.9, 0.1, 0.1, 0.1]), "n2": np.array([0.9, 0.6, 0.1, 0.1]), "o3": np.zeros(4)}
    prevalence = {"o1": np.array([0.9, 0.7, 0.3, 0.2]), "o2": np.array([0.9, 0.4, 0.2, 0.2])}
    calls = []

    def expected(pair):
        calls.append(pair["old_doc_id"])
        return prevalence[pair["old_doc_id"]]

    inst, stats = revision_instances(pairs, p, expected, TYPES, min_prevalence=0.5)
    assert [i["doc_id"] for i in inst] == ["o1"] and inst[0]["ground_truth"] == ["b"]
    assert inst[0]["added"] == ["b", "c"] and inst[0]["added_prevalence"] == {"b": 0.7, "c": 0.3}
    assert inst[0]["year"] == 2017 and calls == ["o1", "o2"]
    assert stats == {"pairs": 3, "missing_version": 1, "no_added_type": 0, "below_prevalence": 1, "instances": 1}
    # the system-neutral variant keeps every pair that adds a type
    inst, _ = revision_instances(pairs, p, expected, TYPES, min_prevalence=None)
    assert [i["doc_id"] for i in inst] == ["o1", "o2"] and inst[0]["ground_truth"] == ["b", "c"]
