"""engine/index.py (plan 5.2): one Index interface, FAISS (development) and Qdrant (production) backends, payload filters
for genre, category, date window and excluded entities; both backends agree on exact search."""

import numpy as np
import pytest

from engine.index import Filters, load_index, make_index

N, DIM = 1000, 32


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, DIM)).astype(np.float32)
    ids = [f"doc{i}" for i in range(N)]
    payloads = [{"genre": "g", "category": "abc"[i % 3], "date": f"{2016 + i % 8}-0{1 + i % 9}-15",
                 "entity_id": f"site{i // 4}", "split": "train"} for i in range(N)]
    Q = rng.normal(size=(20, DIM)).astype(np.float32)
    return ids, X, payloads, Q


def build(backend, corpus, **kw):
    ids, X, payloads, _ = corpus
    idx = make_index(backend, DIM, **kw)
    idx.add(ids, X, payloads)
    return idx


@pytest.mark.parametrize("filters", [None,
                                     Filters(category=["a"]),
                                     Filters(category=["b", "c"], date_from="2018-01-01", date_to="2020-12-31"),
                                     Filters(exclude_entities=["site0", "site1", "site2"], genre="g")])
def test_faiss_and_qdrant_agree_on_exact_search(corpus, filters):
    ids, X, payloads, Q = corpus
    f, q = build("faiss", corpus), build("qdrant", corpus)
    assert len(f) == len(q) == N
    for v in Q:
        a = f.search(v, 10, filters, exact=True)
        b = q.search(v, 10, filters, exact=True)
        assert [h.id for h in a] == [h.id for h in b]
        assert np.allclose([h.score for h in a], [h.score for h in b], atol=1e-4)
        for h in a:
            p = payloads[ids.index(h.id)]
            if filters and filters.category:
                assert p["category"] in filters.category
            if filters and filters.date_from:
                assert filters.date_from <= p["date"] <= filters.date_to
            if filters and filters.exclude_entities:
                assert p["entity_id"] not in filters.exclude_entities
            assert h.payload["entity_id"] == p["entity_id"]


def test_scores_are_cosine_and_sorted(corpus):
    ids, X, _, Q = corpus
    hits = build("faiss", corpus).search(Q[0], 5, exact=True)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    qn = Q[0] / np.linalg.norm(Q[0])
    best = np.argsort(-(Xn @ qn))[:5]
    assert [h.id for h in hits] == [ids[i] for i in best]
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:])) and abs(hits[0].score - float(Xn[best[0]] @ qn)) < 1e-5


def test_hnsw_recall_against_exact(corpus):
    f = build("faiss", corpus)
    assert f.hnsw is None  # FAISS itself is loaded only for approximate search
    _, _, _, Q = corpus
    rec = np.mean([len({h.id for h in f.search(v, 10)} & {h.id for h in f.search(v, 10, exact=True)}) / 10 for v in Q])
    assert rec >= 0.9


@pytest.mark.parametrize("backend", ["faiss", "qdrant"])
def test_save_and_load_round_trip(corpus, tmp_path, backend):
    _, _, _, Q = corpus
    idx = build(backend, corpus)
    idx.save(tmp_path / backend, meta={"model": "test-embedder"})
    back = load_index(tmp_path / backend)
    assert type(back) is type(idx) and len(back) == N and back.meta["model"] == "test-embedder"
    flt = Filters(category=["a"], date_from="2017-01-01")
    assert [h.id for h in back.search(Q[1], 10, flt, exact=True)] == [h.id for h in idx.search(Q[1], 10, flt, exact=True)]


def test_ids_are_unique_and_dimension_checked(corpus):
    ids, X, payloads, _ = corpus
    idx = make_index("faiss", DIM)
    idx.add(ids[:5], X[:5], payloads[:5])
    with pytest.raises(ValueError):
        idx.add(ids[:1], X[:1], payloads[:1])
    with pytest.raises(ValueError):
        idx.search(np.zeros(DIM + 1), 3)


def test_fewer_matches_than_k(corpus):
    q = build("qdrant", corpus)
    f = build("faiss", corpus)
    flt = Filters(exclude_entities=[f"site{i}" for i in range(249)])  # 4 documents left
    assert len(q.search(corpus[3][0], 10, flt, exact=True)) == len(f.search(corpus[3][0], 10, flt)) == 4


def test_filtered_search_never_loads_faiss_with_torch_present():
    import subprocess
    import sys

    code = ("import torch, sys, numpy as np\n"
            "from engine.index import make_index, Filters\n"
            "i = make_index('faiss', 8); i.add([f'd{k}' for k in range(50)], np.random.rand(50, 8), [{'category': 'a'}] * 50)\n"
            "assert len(i.search(np.random.rand(8), 5, Filters(category=['a']))) == 5\n"
            "assert len(i.search(np.random.rand(8), 5)) == 5 and 'faiss' not in sys.modules or sys.platform != 'darwin'\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "ok", out.stderr[-500:]
