"""Index interface with FAISS and Qdrant backends (plan §0.2 #10, §5.2).

    idx = make_index("faiss", dim)                    # or "qdrant" (in memory, or a server via url=...)
    idx.add(ids, vectors, payloads)                   # payload: genre, category, date (ISO), entity_id, split, ...
    hits = idx.search(vector, k, Filters(category=["cs.CL"], date_from="2019-01-01", exclude_entities=["x.com"]))
    idx.save(path, meta={...}); idx = load_index(path)

Similarity is cosine (vectors are L2-normalised on the way in). Both backends store the same vectors and payloads and
save to the same portable format (index.json, vectors.npy, payloads.jsonl), so an index built with one loads into the
other. FAISS (development): HNSW for unfiltered approximate search, built on first use; exact or filtered search is a
brute-force pass over the matching documents (a pack holds tens of thousands of documents: milliseconds). Qdrant
(production): payload filters inside the engine; `exact=True` asks it for exact search.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class Filters:
    """Restrictions on the documents a search may return; None means no restriction. Dates are ISO strings (a
    date_to of "2020-12-31" includes that day); exclude_entities drops every document of those entities."""
    genre: str | None = None
    category: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    splits: list[str] | None = None
    exclude_entities: list[str] | None = None
    exclude_ids: list[str] | None = None


@dataclass
class Hit:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


def _date_num(date: str | None, *, end: bool = False) -> int | None:
    """'2019-05-01' -> 20190501; a bare year or year-month is padded to its first (or, with end, last) day."""
    if not date:
        return None
    parts = (str(date)[:10].split("-") + ["", ""])[:3]
    y, m, d = parts[0], parts[1] or ("12" if end else "01"), parts[2] or ("31" if end else "01")
    return int(f"{int(y):04d}{int(m):02d}{int(d):02d}")


class Index(ABC):
    backend = "abstract"

    def __init__(self, dim: int, meta: dict[str, Any] | None = None):
        self.dim = dim
        self.meta = dict(meta or {})
        self.ids: list[str] = []
        self.pos: dict[str, int] = {}
        self.payloads: list[dict[str, Any]] = []
        self.X = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def _check(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        if v.shape[-1] != self.dim:
            raise ValueError(f"vector dimension {v.shape[-1]} != index dimension {self.dim}")
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    def add(self, ids: Sequence[str], vectors: np.ndarray, payloads: Sequence[dict[str, Any]]) -> None:
        if len(ids) != len(vectors) or len(ids) != len(payloads):
            raise ValueError("ids, vectors and payloads differ in length")
        dup = [i for i in ids if i in self.pos] or ([i for i in set(ids) if list(ids).count(i) > 1] if len(set(ids)) < len(ids) else [])
        if dup:
            raise ValueError(f"ids already in the index: {dup[:3]}")
        V = self._check(np.atleast_2d(vectors))
        start = len(self.ids)
        for k, i in enumerate(ids):
            self.pos[i] = start + k
        self.ids += list(ids)
        self.payloads += [dict(p) for p in payloads]
        self.X = np.vstack([self.X, V])
        self._added(start, V)

    @abstractmethod
    def _added(self, start: int, V: np.ndarray) -> None:
        """Backend hook: rows start.. of self.X were added."""

    @abstractmethod
    def search(self, vector: np.ndarray, k: int, filters: Filters | None = None, *, exact: bool = False) -> list[Hit]:
        ...

    def _mask(self, f: Filters | None) -> np.ndarray:
        """Boolean mask of the documents a filter admits (shared by the brute-force paths)."""
        m = np.ones(len(self.ids), dtype=bool)
        if f is None:
            return m
        lo, hi = _date_num(f.date_from), _date_num(f.date_to, end=True)
        for i, p in enumerate(self.payloads):
            if (f.genre and p.get("genre") != f.genre) or (f.category and p.get("category") not in f.category) \
                    or (f.splits and p.get("split") not in f.splits) \
                    or (f.exclude_entities and p.get("entity_id") in f.exclude_entities) \
                    or (f.exclude_ids and self.ids[i] in f.exclude_ids):
                m[i] = False
                continue
            if lo is not None or hi is not None:
                d = _date_num(p.get("date"))
                if d is None or (lo is not None and d < lo) or (hi is not None and d > hi):
                    m[i] = False
        return m

    def _brute(self, q: np.ndarray, k: int, mask: np.ndarray) -> list[Hit]:
        rows = np.flatnonzero(mask)
        if not len(rows):
            return []
        s = self.X[rows] @ q
        top = rows[np.argsort(-s, kind="stable")[:k]]
        return [Hit(self.ids[r], float(self.X[r] @ q), self.payloads[r]) for r in top]

    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.meta.update(meta or {})
        np.save(path / "vectors.npy", self.X)
        with open(path / "payloads.jsonl", "w", encoding="utf-8") as fh:
            for i, p in zip(self.ids, self.payloads):
                fh.write(json.dumps({"id": i, **p}, ensure_ascii=False) + "\n")
        (path / "index.json").write_text(json.dumps({"backend": self.backend, "dim": self.dim, "count": len(self),
                                                     "metric": "cosine", "meta": self.meta}, indent=1) + "\n")


class FaissIndex(Index):
    """Exact and filtered searches are a NumPy pass over the matching rows (what cohort retrieval uses), so the FAISS
    library is loaded only when an unfiltered approximate search is first asked for, and the HNSW graph is built
    then. On macOS, FAISS and PyTorch each bundle a different build of libomp and a process that loads both aborts
    (checked 2026-10-19: faiss-cpu 1.15.1, torch 2.13.0); there, in a process that has already loaded PyTorch, an
    approximate search is answered exactly instead (`self.approximate` says which)."""

    backend = "faiss"

    def __init__(self, dim: int, *, m: int = 32, ef_construction: int = 80, ef_search: int = 128,
                 meta: dict[str, Any] | None = None):
        super().__init__(dim, meta)
        self.hnsw_params = (m, ef_construction, ef_search)
        self.hnsw: Any = None
        self.approximate: bool | None = None

    def _added(self, start: int, V: np.ndarray) -> None:
        self.hnsw = None  # rebuilt on the next approximate search

    def _ensure_hnsw(self) -> bool:
        import sys

        if self.hnsw is not None:
            return True
        if sys.platform == "darwin" and "torch" in sys.modules:
            self.approximate = False
            return False
        import faiss

        m, efc, efs = self.hnsw_params
        self.hnsw = faiss.IndexHNSWFlat(self.dim, m, faiss.METRIC_INNER_PRODUCT)
        self.hnsw.hnsw.efConstruction, self.hnsw.hnsw.efSearch = efc, efs
        self.hnsw.add(self.X)
        self.approximate = True
        return True

    def search(self, vector, k, filters=None, *, exact=False):
        q = self._check(vector).reshape(-1)
        if exact or filters is not None or not self._ensure_hnsw():
            return self._brute(q, k, self._mask(filters))
        _, I = self.hnsw.search(q[None, :], min(k, len(self)))
        return [Hit(self.ids[r], float(self.X[r] @ q), self.payloads[r]) for r in I[0] if r >= 0]


class QdrantIndex(Index):
    backend = "qdrant"
    COLLECTION = "documents"

    def __init__(self, dim: int, *, url: str | None = None, location: str = ":memory:",
                 meta: dict[str, Any] | None = None, batch: int = 512):
        from qdrant_client import QdrantClient, models

        super().__init__(dim, meta)
        self.m, self.batch, self.local = models, batch, not url  # local mode is always exact (brute force)
        self.client = QdrantClient(url=url) if url else QdrantClient(location=location)
        if self.client.collection_exists(self.COLLECTION):
            self.client.delete_collection(self.COLLECTION)
        self.client.create_collection(self.COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    def _added(self, start: int, V: np.ndarray) -> None:
        pts = [self.m.PointStruct(id=start + k, vector=v.tolist(),
                                  payload={**self.payloads[start + k], "doc_id": self.ids[start + k],
                                           "date_num": _date_num(self.payloads[start + k].get("date"))})
               for k, v in enumerate(V)]
        for b in range(0, len(pts), self.batch):
            self.client.upsert(self.COLLECTION, points=pts[b:b + self.batch])

    def _filter(self, f: Filters | None) -> Any:
        if f is None:
            return None
        m, must, must_not = self.m, [], []
        if f.genre:
            must.append(m.FieldCondition(key="genre", match=m.MatchValue(value=f.genre)))
        if f.category:
            must.append(m.FieldCondition(key="category", match=m.MatchAny(any=list(f.category))))
        if f.splits:
            must.append(m.FieldCondition(key="split", match=m.MatchAny(any=list(f.splits))))
        lo, hi = _date_num(f.date_from), _date_num(f.date_to, end=True)
        if lo is not None or hi is not None:
            must.append(m.FieldCondition(key="date_num", range=m.Range(gte=lo, lte=hi)))
        if f.exclude_entities:
            must_not.append(m.FieldCondition(key="entity_id", match=m.MatchAny(any=list(f.exclude_entities))))
        if f.exclude_ids:
            must_not.append(m.FieldCondition(key="doc_id", match=m.MatchAny(any=list(f.exclude_ids))))
        return m.Filter(must=must or None, must_not=must_not or None)

    def search(self, vector, k, filters=None, *, exact=False):
        q = self._check(vector).reshape(-1)
        res = self.client.query_points(self.COLLECTION, query=q.tolist(), limit=k, query_filter=self._filter(filters),
                                       search_params=None if self.local else self.m.SearchParams(exact=exact),
                                       with_payload=False)
        return [Hit(self.ids[p.id], float(p.score), self.payloads[p.id]) for p in res.points]


def make_index(backend: str, dim: int, **kw: Any) -> Index:
    if backend == "faiss":
        return FaissIndex(dim, **kw)
    if backend == "qdrant":
        return QdrantIndex(dim, **kw)
    raise ValueError(f"unknown index backend {backend!r} (faiss | qdrant)")


def load_index(path: str | Path, backend: str | None = None, **kw: Any) -> Index:
    """Load a saved index; `backend` overrides the one it was saved with (the format is shared)."""
    path = Path(path)
    info = json.loads((path / "index.json").read_text())
    idx = make_index(backend or info["backend"], info["dim"], meta=info.get("meta"), **kw)
    rows = [json.loads(line) for line in open(path / "payloads.jsonl", encoding="utf-8")]
    idx.add([r.pop("id") for r in rows], np.load(path / "vectors.npy"), rows)
    return idx
