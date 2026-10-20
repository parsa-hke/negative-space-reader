"""Helpers of the cohort diagnosis (bench.stage: diagnose; decision record 007)."""

import numpy as np
import pytest

from bench.metrics import within_document_auc
from engine.embed import document_vector, section_mean_vector
from experiments.bench import one_per_entity


@pytest.mark.parametrize("w", [0.25, 0.5, 0.8])
def test_the_section_mean_is_recovered_from_a_document_vector(w):
    rng = np.random.default_rng(0)
    S, t = rng.normal(size=(7, 16)), rng.normal(size=16)
    v = document_vector(S, t, title_abstract_weight=w)
    want = document_vector(S, None)
    assert float(section_mean_vector(v, t, title_abstract_weight=w) @ want) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(section_mean_vector(v, None, title_abstract_weight=w), v / np.linalg.norm(v))


def test_within_document_auc_by_hand():
    E = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.8]])
    Y = np.array([[1, 0, 0], [1, 0, 1]])
    # document 1: the present type outranks both absent ones (1.0); document 2: 0.2 vs 0.8 loses, 0.8 vs 0.8 ties (0.25)
    assert within_document_auc(E, Y) == pytest.approx((1.0 + 0.25) / 2)
    assert within_document_auc(np.ones((1, 3)), np.array([[1, 1, 1]])) is None  # no absent type: no document scored


def test_one_snapshot_per_entity_keeps_the_nearest_and_the_order():
    peers = ["a1", "b1", "a2", "c1", "b2", "d1"]
    entity = {"a1": "a", "a2": "a", "b1": "b", "b2": "b", "c1": "c", "d1": "d"}
    assert one_per_entity(peers, entity, 3) == ["a1", "b1", "c1"]
    assert one_per_entity(peers, entity, 10) == ["a1", "b1", "c1", "d1"]
