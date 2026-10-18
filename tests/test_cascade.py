"""engine/presence.py: the presence cascade (plan §0.2 #8, task 4.5)."""

import numpy as np
import pytest

from engine.presence import Cascade, band_thresholds, rerank_pairs, type_query


def test_band_thresholds_cover_the_overlap():
    pos, neg = np.array([0.5, 0.6, 0.7, 0.8, 0.9]), np.array([0.1, 0.2, 0.3, 0.4, 0.55])
    low, high = band_thresholds(pos, neg, q_low=0.0, q_high=1.0)
    assert (low, high) == (0.5, 0.55)
    low, high = band_thresholds(np.array([0.9]), np.array([0.1]), q_low=0.0, q_high=1.0)
    assert low == high  # no overlap: an empty band, nothing goes to the reranker


def test_rerank_pairs_only_in_the_band():
    proto = np.array([[0.2, 0.5, 0.9], [0.45, 0.6, 0.3]])
    pairs = rerank_pairs(proto, low=0.4, high=0.7)
    assert pairs == [(0, 1), (1, 0), (1, 1)]


def test_cascade_combines_and_thresholds_per_type(tmp_path):
    rng = np.random.default_rng(0)
    n, T = 400, 3
    Y = (rng.random((n, T)) < 0.3).astype(int)
    clf = np.clip(0.25 + 0.5 * Y + 0.2 * rng.normal(size=(n, T)), 0.01, 0.99)
    proto = 0.3 + 0.3 * Y + 0.1 * rng.normal(size=(n, T))
    rr = np.where((proto > 0.4) & (proto < 0.6), np.clip(0.2 + 0.6 * Y + 0.1 * rng.normal(size=(n, T)), 0, 1), np.nan)
    c = Cascade(types=["a", "b", "c"], low=0.4, high=0.6)
    c.fit(clf[:200], proto[:200], rr[:200], Y[:200])
    P = c.predict(clf[200:], proto[200:], rr[200:])
    assert P.shape == (200, 3) and ((P >= c.threshold) == Y[200:]).mean() > 0.9
    c.save(tmp_path / "c.json")
    again = Cascade.load(tmp_path / "c.json")
    assert np.allclose(again.predict(clf[:5], proto[:5], rr[:5]), c.predict(clf[:5], proto[:5], rr[:5]))


def test_type_query_reads_like_a_definition():
    q = type_query({"id": "x", "definition": "States x.", "positive_examples": ["X is said.", "X again."]})
    assert q.startswith("States x.") and "X is said." in q


@pytest.mark.parametrize("bad", [np.array([[0.1, 0.2]]), np.array([[0.1], [0.2]])])
def test_cascade_refuses_shape_mismatch(bad):
    c = Cascade(types=["a", "b"], low=0.4, high=0.6)
    with pytest.raises(ValueError):
        c.fit(bad, np.zeros((2, 2)), np.full((2, 2), np.nan), np.zeros((2, 2), int))
