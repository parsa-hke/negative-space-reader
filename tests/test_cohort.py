"""engine/cohort.py (plan 5.3) and the document vector of §0.2 #9 (engine/embed.document_vector)."""

import numpy as np
import pytest

from engine.cohort import CohortRetriever
from engine.embed import document_vector
from engine.index import make_index


def test_document_vector_is_the_normalised_mean_of_sections_plus_title():
    S = np.array([[1.0, 0, 0], [0, 2.0, 0]])  # section vectors need not be unit length
    v = document_vector(S, np.array([0, 0, 3.0]))
    expect = np.array([0.5, 0.5, 0]) / np.linalg.norm([0.5, 0.5, 0]) + np.array([0, 0, 1.0])
    assert np.allclose(v, expect / np.linalg.norm(expect)) and abs(np.linalg.norm(v) - 1) < 1e-6
    assert np.allclose(document_vector(S, None), np.array([1, 1, 0]) / np.sqrt(2))  # no title: sections only
    w = document_vector(S, np.array([0, 0, 1.0]), title_abstract_weight=0.0)
    assert np.allclose(w, np.array([1, 1, 0]) / np.sqrt(2))


@pytest.fixture
def index():
    rng = np.random.default_rng(1)
    idx = make_index("faiss", 16)
    ids, X, payloads = [], [], []
    for i in range(300):
        ent = f"e{i // 3}"  # three yearly snapshots per entity, nearly identical vectors
        base = rng.normal(size=16) if i % 3 == 0 else X[-1]
        X.append(base + rng.normal(scale=0.01, size=16))
        ids.append(f"d{i}")
        payloads.append({"genre": "g", "entity_id": ent, "category": "x" if i < 240 else "y",
                         "date": f"{2015 + i % 9}-06-01", "split": "train"})
    idx.add(ids, np.array(X), payloads)
    return idx, np.array(X), payloads


def test_never_returns_the_query_entity(index):
    idx, X, payloads = index
    r = CohortRetriever(idx, k=20, category_filter=False)
    c = r.retrieve(X[3], entity_id="e1", doc_id="d3")
    assert len(c.peers) == 20 and all(h.payload["entity_id"] != "e1" for h in c.peers)
    assert "d4" not in [h.id for h in c.peers]  # the same site's other snapshot would otherwise be the top peer
    assert c.stats["entities"] == len({h.payload["entity_id"] for h in c.peers})


def test_category_filter_and_fallback(index):
    idx, X, _ = index
    r = CohortRetriever(idx, k=30, category_filter=True)
    c = r.retrieve(X[0], entity_id="e0", category="x")
    assert all(h.payload["category"] == "x" for h in c.peers) and not c.fallback
    small = r.retrieve(X[250], entity_id="e83", category="y")  # 60 'y' documents minus 3 of e83: 57 >= 30
    assert not small.fallback and all(h.payload["category"] == "y" for h in small.peers)
    r40 = CohortRetriever(idx, k=80, category_filter=True)
    fb = r40.retrieve(X[250], entity_id="e83", category="y")  # pool of 57 < 80: unfiltered, flagged
    assert fb.fallback and len(fb.peers) == 80 and fb.filters_used["category"] is None


def test_date_window_and_unknown_category(index):
    idx, X, _ = index
    r = CohortRetriever(idx, k=10, category_filter=True, date_window=("2017-01-01", "2019-12-31"))
    c = r.retrieve(X[0], entity_id="e0", category=None)  # no category metadata: no category filter
    assert all("2017" <= h.payload["date"][:4] <= "2019" for h in c.peers) and c.filters_used["category"] is None
    assert 0 < c.stats["mean_similarity"] <= 1 and c.stats["min_similarity"] <= c.stats["max_similarity"]
