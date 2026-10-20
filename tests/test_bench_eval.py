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
