"""experiments/bench.py eval helpers: which expectation each system ranks by (plan 6.3)."""

import numpy as np

from experiments.bench import _expectations, _ranked

TYPES = ["a", "b", "c", "d"]
UNIFORM, COHORT = np.array([0.9, 0.2, 0.5, 0.4]), np.array([0.1, 0.8, 0.5, 0.3])
P_DOC, BITS = np.array([0.1, 0.2, 0.9, 0.0]), np.array([0, 0, 1, 0])  # c is present: never a candidate


def rank(system, seed=0):
    e, with_presence = _expectations(system, uniform=UNIFORM, cohort=COHORT, n_types=4, rng=np.random.default_rng(seed))
    return _ranked(TYPES, e, P_DOC, BITS, np.ones(4), with_presence)


def test_systems_rank_absent_types_by_their_own_expectation():
    assert rank("genre_uniform") == ["a", "d", "b"]   # 0.9 x 0.9, 0.4 x 1.0, 0.2 x 0.8
    assert rank("level1") == ["b", "d", "a"]          # 0.8 x 0.8, 0.3 x 1.0, 0.1 x 0.9
    assert rank("absence_only") == ["d", "a", "b"]    # 1 - P(present)
    assert rank("random", seed=3) == rank("random", seed=3) and sorted(rank("random")) == ["a", "b", "d"]


def test_severity_weights_reorder_candidates():
    w = np.array([1.0, 1.0, 1.0, 3.0])  # d: 0.4 x 1.0 x 3 overtakes a
    e, with_presence = _expectations("genre_uniform", uniform=UNIFORM, cohort=COHORT, n_types=4,
                                     rng=np.random.default_rng(0))
    assert _ranked(TYPES, e, P_DOC, BITS, w, with_presence) == ["d", "a", "b"]


def test_compare_to_reference_pairs_instances_per_system():
    from experiments.bench import _compare_to_reference

    def row(i, s, r5):
        return {"instance_id": f"i{i}", "regime": "single_type", "system": s, "recall@5": r5, "mrr": r5}

    ref = [row(i, s, 0.0) for i in range(4) for s in ("level1", "genre_uniform")]
    new = [row(i, "level1", 1.0 if i < 2 else 0.0) for i in range(4)] + [row(i, "genre_uniform", 0.0) for i in range(3)]
    out = _compare_to_reference(new, ref, "g", ["single_type"], ["recall@5"], dict(n_boot=50, ci=0.95, seed=0))
    by = {r["system"]: r for r in out}
    assert by["level1 - reference"]["value"] == 0.5 and by["level1 - reference"]["n"] == 4
    assert by["genre_uniform - reference"]["value"] == 0.0 and by["genre_uniform - reference"]["n"] == 3
    assert all(r["metric"] == "recall@5" and r["regime"] == "single_type" for r in out)
