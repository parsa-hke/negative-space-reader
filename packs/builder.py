"""Pack builder (plan §0.2 #3, #25; §5.7): the stages that turn a corpus and a frozen schema into a pack.

Stages written so far (experiments/build_pack.py runs build.stages in order, per genre):
  index         document vectors (§0.2 #9: mean of the stored section vectors and the title vector, models.embedder.title_abstract_weight) for the indexed
                split, with payloads for the cohort filters (genre, category, date, entity, split, source), saved as
                packs/<genre>/index/ (engine/index.py format; the manifest records the embedder).
  evidence      the best section per indexed document and type (section_probs) -> index/best_sections.npz
  cohort_report cohorts of a sample of dev documents: similarity, entity diversity, fallback rate (plan 5.3).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def document_vectors(cfg: dict[str, Any], genre: str, splits: list[str]) -> tuple[list[str], np.ndarray,
                                                                                list[dict[str, Any]], int]:
    """(ids, vectors, payloads, documents with a title vector) for the documents of the given splits."""
    from data.schema import Document, read_jsonl
    from data.splits import load_manifest
    from engine.embed import EmbeddingStore, document_vector, embedding_slug

    split = load_manifest(_path(cfg["paths"]["splits"]) / f"{genre}.json")
    docs = [d for d in read_jsonl(_path(cfg["paths"]["data_processed"]) / genre / "documents.jsonl", Document)
            if split.get(d.id) in splits]
    emb = _path(cfg["paths"]["embeddings"]) / embedding_slug(cfg["models"]["embedder"]) / genre
    S, T = EmbeddingStore.load(emb / "sections"), EmbeddingStore.load(emb / "titles")
    for store in (S, T):
        if store.meta.get("model") != cfg["models"]["embedder"]["id"]:
            raise ValueError(f"{genre}: stored vectors come from {store.meta.get('model')}, the pack's embedder is "
                             f"{cfg['models']['embedder']['id']}")
    rows: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(S.ids):
        rows[r[0]].append(i)
    title_row = {d: i for i, d in enumerate(T.ids)}
    w = cfg["models"]["embedder"].get("title_abstract_weight", 0.5)
    ids, X, payloads, missing = [], [], [], []
    for d in docs:
        if not rows.get(d.id):
            missing.append(d.id)
            continue
        t = T.vectors[title_row[d.id]] if d.id in title_row and d.title else None
        ids.append(d.id)
        X.append(document_vector(S.vectors[rows[d.id]], t, title_abstract_weight=w))
        payloads.append({"genre": genre, "category": d.category, "date": d.date, "entity_id": d.entity_id,
                         "split": split[d.id], "source": d.source})
    if missing:
        raise ValueError(f"{genre}: {len(missing)} documents have no stored section vectors, e.g. {missing[:3]}")
    return ids, np.asarray(X), payloads, sum(1 for d in docs if d.id in title_row and d.title)


def build_index(cfg: dict[str, Any], genre: str) -> dict[str, Any]:
    import time

    from engine.index import make_index

    t0 = time.perf_counter()
    splits = (cfg.get("index") or {}).get("splits", ["train"])
    w = cfg["models"]["embedder"].get("title_abstract_weight", 0.5)
    ids, X, payloads, with_title = document_vectors(cfg, genre, splits)
    icfg = cfg.get("index") or {}
    backend = icfg.get("backend", "faiss")
    kw = {}
    if backend == "faiss":
        f = icfg.get("faiss") or {}
        kw = {"m": f.get("M", 32), "ef_construction": f.get("ef_construction", 80), "ef_search": f.get("ef_search", 128)}
    idx = make_index(backend, int(np.asarray(X).shape[1]), **kw)
    idx.add(ids, np.asarray(X), payloads)
    out = _path(cfg["paths"]["packs"]) / genre / "index"
    n_offsets = _section_offsets(_path(cfg["paths"]["data_processed"]) / genre / "sections.jsonl", set(ids),
                                 out / "sections.npz")
    idx.save(out, meta={"embedder": cfg["models"]["embedder"]["id"], "title_abstract_weight": w, "splits": splits,
                        "document_vector": "(1 - w) x normalised mean of section vectors + w x title vector"})
    return {"documents": len(ids), "dim": idx.dim, "backend": backend, "with_title": with_title, "sections": n_offsets,
            "seconds": round(time.perf_counter() - t0, 1)}


def _section_offsets(sections_path: Path, keep: set[str], out: Path) -> int:
    """Byte offset of every section of the indexed documents in sections.jsonl, so evidence reads single lines."""
    import re

    head_re = re.compile(rb'^\{"doc_id": "((?:[^"\\]|\\.)*)", "idx": (\d+)')
    docs, idxs, offs = [], [], []
    with open(sections_path, "rb") as fh:
        pos = 0
        for line in fh:
            m = head_re.match(line)
            d, i = (m.group(1).decode("utf-8"), int(m.group(2))) if m else (None, None)
            if m is None and line.strip():
                r = json.loads(line)
                d, i = r["doc_id"], r["idx"]
            if d in keep:
                docs.append(d)
                idxs.append(i)
                offs.append(pos)
            pos += len(line)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, doc_id=np.array(docs), idx=np.array(idxs, dtype=np.int32), offset=np.array(offs, dtype=np.int64),
             size=np.array([sections_path.stat().st_size]))
    return len(docs)


class SectionTexts:
    """section_text(doc_id, idx) for the indexed documents, read by byte offset from sections.jsonl."""

    def __init__(self, offsets: dict[tuple[str, int], int], sections_path: Path):
        self.offsets, self.path = offsets, Path(sections_path)

    @classmethod
    def load(cls, index_dir: str | Path, sections_path: str | Path) -> "SectionTexts":
        z = np.load(Path(index_dir) / "sections.npz")
        if int(z["size"][0]) != Path(sections_path).stat().st_size:
            raise ValueError(f"{sections_path} changed since the index was built; rebuild the index stage")
        return cls({(d, int(i)): int(o) for d, i, o in zip(z["doc_id"].tolist(), z["idx"], z["offset"])}, Path(sections_path))

    def __call__(self, doc_id: str, idx: int) -> str | None:
        off = self.offsets.get((doc_id, int(idx)))
        if off is None:
            return None
        with open(self.path, "rb") as fh:
            fh.seek(off)
            r = json.loads(fh.readline())
        return r["text"] if r["doc_id"] == doc_id and r["idx"] == idx else None


def cohort_report(cfg: dict[str, Any], genre: str) -> dict[str, Any]:
    """Cohorts of a seeded sample of dev documents (never indexed) against the pack index: similarity of the k
    peers, distinct entities per cohort, how often the category filter fell back, and a check that no cohort
    contains the query's own entity. Table reports/tables/cohort_similarity_<genre>.csv, figure
    reports/figures/cohort_similarity_<genre>.png."""
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from engine.cohort import CohortRetriever
    from engine.index import load_index

    rc = (cfg.get("build") or {}).get("cohort_report") or {}
    idx = load_index(_path(cfg["paths"]["packs"]) / genre / "index")
    ids, X, payloads, _ = document_vectors(cfg, genre, ["dev"])
    rng = np.random.default_rng(cfg.get("splits", {}).get("seed", 0))
    pick = sorted(rng.choice(len(ids), size=min(rc.get("queries", 500), len(ids)), replace=False))
    ccfg = dict(cfg.get("cohort") or {})
    if rc.get("k"):
        ccfg["k"] = rc["k"]
    retr = CohortRetriever.from_config(idx, ccfg)
    rows = []
    for i in pick:
        p = payloads[i]
        c = retr.retrieve(X[i], entity_id=p["entity_id"], doc_id=ids[i], category=p["category"])
        own = sum(h.payload.get("entity_id") == p["entity_id"] for h in c.peers)
        rows.append({"doc_id": ids[i], "category": p["category"], **c.stats, "fallback": int(c.fallback),
                     "own_entity_in_cohort": own})
    tables = _path(cfg["paths"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / f"cohort_similarity_{genre}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mean = np.array([r["mean_similarity"] for r in rows], dtype=float)
    ents = np.array([r["entities"] / max(1, r["n"]) for r in rows], dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].hist([r["max_similarity"] for r in rows], bins=30, alpha=0.6, label="nearest peer")
    ax[0].hist(mean, bins=30, alpha=0.6, label=f"mean of the {ccfg.get('k', 200)} peers")
    ax[0].hist([r["min_similarity"] for r in rows], bins=30, alpha=0.6, label="farthest peer")
    ax[0].set_xlabel("cosine similarity to the query document")
    ax[0].set_ylabel("query documents")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].hist(ents, bins=20, color="#3a7d5c")
    ax[1].set_xlabel("distinct entities / peers in the cohort")
    ax[1].set_title(f"fallback (category pool < k): {np.mean([r['fallback'] for r in rows]):.0%}", fontsize=9)
    fig.suptitle(f"{genre}: cohorts of {len(rows)} dev documents", fontsize=10)
    fig.tight_layout()
    figs = _path(cfg["paths"]["figures"])
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"cohort_similarity_{genre}.png", dpi=150)
    plt.close(fig)
    return {"queries": len(rows), "k": ccfg.get("k", 200), "mean_similarity": round(float(mean.mean()), 4),
            "median_nearest_similarity": round(float(np.median([r["max_similarity"] for r in rows])), 4),
            "entity_share_mean": round(float(ents.mean()), 4),
            "fallback_rate": round(float(np.mean([r["fallback"] for r in rows])), 4),
            "own_entity_in_cohort": int(sum(r["own_entity_in_cohort"] for r in rows))}


def build_evidence(cfg: dict[str, Any], genre: str) -> dict[str, Any]:
    """For every indexed document and type, the section with the highest stored probability (section_probs):
    where a peer states the type most clearly, quoted as evidence (engine/evidence.py). -> index/best_sections.npz"""
    import yaml

    from engine.evidence import best_sections
    from packs.profiles import read_profiles

    pack = _path(cfg["paths"]["packs"]) / genre
    types = [t["id"] for t in yaml.safe_load((pack / "schema.yaml").read_text(encoding="utf-8"))["types"]]
    splits = (cfg.get("index") or {}).get("splits", ["train"])
    sp = read_profiles(pack / "section_probs")
    best = best_sections(sp[sp.split.isin(splits)], types)
    docs = sorted(best)
    (pack / "index").mkdir(parents=True, exist_ok=True)
    np.savez(pack / "index" / "best_sections.npz", doc_id=np.array(docs), types=np.array(types),
             best=np.stack([best[d] for d in docs]).astype(np.int32) if docs else np.zeros((0, len(types)), np.int32))
    return {"documents": len(docs), "types": len(types)}


def load_best_sections(index_dir: str | Path) -> dict[str, np.ndarray]:
    z = np.load(Path(index_dir) / "best_sections.npz")
    return dict(zip(z["doc_id"].tolist(), z["best"]))


STAGES = {"index": build_index, "evidence": build_evidence, "cohort_report": cohort_report}
