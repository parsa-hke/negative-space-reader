"""packs/prototypes.py: prototype selection per type (plan 4.4)."""

import numpy as np

from packs.prototypes import n_prototypes, select_prototypes, span_candidates


def test_n_prototypes_stays_in_the_plan_range():
    assert [n_prototypes(n, 10, 30) for n in (3, 10, 25, 100, 225, 5000)] == [3, 10, 10, 20, 30, 30]


def test_select_prototypes_picks_one_per_cluster():
    rng = np.random.default_rng(0)
    centres = np.eye(4)
    V = np.concatenate([c + 0.05 * rng.normal(size=(30, 4)) for c in centres])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    picked = select_prototypes(V, k=4, seed=0)
    assert len(picked) == 4 and len({i // 30 for i in picked}) == 4  # one from each true cluster
    assert select_prototypes(V[:3], k=10, seed=0) == [0, 1, 2]


def test_span_candidates_are_grounded_unique_and_from_training_sections():
    rows = {("d", 0): {"evidence": {"a": "Alpha here.", "b": "Beta here."}, "grounded": {"a": True, "b": False}},
            ("e", 1): {"evidence": {"a": "alpha  here."}, "grounded": {"a": True}},
            ("f", 2): {"evidence": {"a": "Another alpha."}, "grounded": {"a": True}}}
    cands = span_candidates(rows, ["a", "b"], keys={("d", 0), ("e", 1)})
    assert [c["text"] for c in cands["a"]] == ["Alpha here."] and cands["b"] == []
    assert cands["a"][0]["doc_id"] == "d"
