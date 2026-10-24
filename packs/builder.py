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


def analyse_gate(cfg: dict[str, Any], genre: str, *, pack: Any = None) -> dict[str, Any]:
    """Gate 5.9: analyse() end to end on held-out dev documents (never indexed; rebuilt as Markdown from their
    sections), template explanations, after one warm-up call. Pass when every document takes under max_seconds.
    -> reports/tables/analyse_gate_<genre>.csv (per document: sections, every step's time)."""
    import csv
    import platform

    from data.schema import Document, Section, read_jsonl
    from data.splits import load_manifest
    from engine.pipeline import analyse, document_markdown
    from packs.spec import GenrePack

    g = (cfg.get("build") or {}).get("analyse_gate") or {}
    n, limit = g.get("documents", 5), g.get("max_seconds", 10)
    pack = pack or GenrePack.load(cfg, genre)
    split = load_manifest(_path(cfg["paths"]["splits"]) / f"{genre}.json")
    processed = _path(cfg["paths"]["data_processed"]) / genre
    labelled = set()
    ls = _path(cfg["paths"]["packs"]) / genre / "label_sets.json"
    if ls.exists():
        labelled = {r["doc_id"] for refs in json.loads(ls.read_text()).values() if isinstance(refs, list) for r in refs}
    indexed_sources = {p.get("source") for p in getattr(pack.index, "payloads", [])} - {None}
    docs = [d for d in read_jsonl(processed / "documents.jsonl", Document) if split.get(d.id) == "dev"
            and d.id not in labelled and (not indexed_sources or d.source in indexed_sources)]
    rng = np.random.default_rng(cfg.get("splits", {}).get("seed", 0))
    docs = [docs[i] for i in sorted(rng.choice(len(docs), size=min(n, len(docs)), replace=False))]
    keep = {d.id for d in docs}
    secs: dict[str, list[Any]] = defaultdict(list)
    for s in read_jsonl(processed / "sections.jsonl", Section):
        if s.doc_id in keep:
            secs[s.doc_id].append(s)
    texts = {d.id: document_markdown(d, secs[d.id]) for d in docs}
    analyse(texts[docs[0].id], pack, top_k=10, explain_mode="template")  # warm-up: first calls load kernels
    rows = []
    for d in docs:
        r = analyse(texts[d.id], pack, entity_id=d.entity_id, top_k=10, explain_mode="template")
        rows.append({"doc_id": d.id, "sections": r.sections, "absences": len(r.absences), **r.timings})
    tables = _path(cfg["paths"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(tables / f"analyse_gate_{genre}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    totals = np.array([r["total"] for r in rows])
    steps = [k for k in rows[0] if k not in ("doc_id", "sections", "absences", "total")]
    return {"documents": len(rows), "max_seconds": round(float(totals.max()), 2), "median_seconds": round(float(np.median(totals)), 2),
            "max_sections": int(max(r["sections"] for r in rows)), "gate": "pass" if totals.max() < limit else "not met",
            "limit_seconds": limit, "median_timings": {k: round(float(np.median([r[k] for r in rows])), 3) for k in steps},
            "machine": f"{platform.machine()} {platform.system()} {platform.processor()}".strip(),
            "device": (getattr(pack, "info", {}) or {}).get("device", "?"),
            "evidence": (getattr(pack, "info", {}) or {}).get("evidence"), "presence": (getattr(pack, "info", {}) or {}).get("presence")}


# ---------------------------------------------------------------------------- the builder (plan 5.7)
from dataclasses import dataclass, field  # noqa: E402
from typing import Callable  # noqa: E402


class GpuStageNeeded(RuntimeError):
    """A GPU stage is stale and this machine has no CUDA: run the named Colab job (configs/colab_jobs.yaml)."""


@dataclass
class Stage:
    name: str
    run: Callable[[dict[str, Any], str, Path], dict[str, Any]]
    inputs: Callable[[dict[str, Any], str], list[Path]]
    outputs: Callable[[dict[str, Any], str], list[Path]]
    config_keys: list[str] = field(default_factory=list)
    gpu: bool = False
    colab_job: str | None = None


def _get(cfg: dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def _file_fp(p: Path) -> str:
    import hashlib

    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(p, "rb") as fh:
        if size <= 64 * 1024 ** 2:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        else:  # large files: size + head + tail (a rerun that changes them changes these)
            h.update(fh.read(4 << 20))
            fh.seek(-(4 << 20), 2)
            h.update(fh.read())
    return h.hexdigest()[:16]


def _key(p: Path) -> str:
    """Repository-relative path when inside the repository (manifests then compare across machines)."""
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def fingerprint(paths: list[Path]) -> dict[str, str]:
    """{path: content fingerprint} for files and every file under directories; missing paths are 'missing'."""
    out: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        files = sorted(f for f in p.rglob("*") if f.is_file() and not f.name.startswith(".")) if p.is_dir() else [p]
        if not p.exists() or (p.is_dir() and not files):
            out[_key(p)] = "missing"
            continue
        for f in files:
            out[_key(f)] = _file_fp(f)
    return out


def _digest(obj: Any) -> str:
    import hashlib

    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build(cfg: dict[str, Any], genre: str, stages: list[Stage], *, manifest_path: Path, run_dir: Path,
          adopt: bool = False, force: list[str] | None = None) -> dict[str, Any]:
    """Run the stages in order, skipping a stage whose input fingerprints, settings (its config_keys and the model
    ids) and outputs are those recorded in the manifest. adopt: a stage with no record whose outputs exist is
    recorded as is (first run over artefacts built before the builder). A stale GPU stage on a machine without CUDA
    raises GpuStageNeeded with its Colab job instead of running for hours on the CPU."""
    import datetime

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)  # a pack built from nothing (fresh clone, smoke; 7.6)
    man = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    man.setdefault("stages", {})
    models = {k: v.get("id") for k, v in (cfg.get("models") or {}).items() if isinstance(v, dict) and v.get("id")}
    schema = _path(cfg["paths"]["packs"]) / genre / "schema.yaml" if cfg.get("paths") else None
    man.update(genre=genre, tier=cfg.get("tier", "lean"), models=models)
    if schema is not None and schema.exists():
        import yaml

        man["schema_version"] = yaml.safe_load(schema.read_text(encoding="utf-8")).get("version")
    report: dict[str, Any] = {"ran": [], "skipped": [], "adopted": []}
    device = str((cfg.get("run") or {}).get("device", ""))
    for st in stages:
        settings = _digest({"config": {k: _get(cfg, k) for k in st.config_keys}, "models": models if st.config_keys else {}})
        ins, outs = fingerprint(st.inputs(cfg, genre)), st.outputs(cfg, genre)
        out_fp = fingerprint(outs)
        have = all(v != "missing" for v in out_fp.values())
        rec = man["stages"].get(st.name)
        if (rec and have and not (force and st.name in force) and rec["inputs"] == _digest(ins)
                and rec["settings"] == settings and rec["outputs"] == out_fp):
            report["skipped"].append(st.name)
            continue
        if adopt and rec is None and have and not (force and st.name in force):
            man["stages"][st.name] = {"inputs": _digest(ins), "settings": settings, "outputs": out_fp, "adopted": True,
                                      "at": datetime.datetime.now().isoformat(timespec="seconds")}
            report["adopted"].append(st.name)
        else:
            if st.gpu and not device.startswith("cuda") and not (cfg.get("build") or {}).get("allow_gpu_stages_on_cpu"):
                manifest_path.write_text(json.dumps(man, indent=1) + "\n")
                raise GpuStageNeeded(f"{genre}: stage {st.name} is out of date and needs a GPU; run the Colab job "
                                     f"{st.colab_job} (configs/colab_jobs.yaml), pull, and build again")
            metrics = st.run(cfg, genre, run_dir)
            man["stages"][st.name] = {"inputs": _digest(ins), "settings": settings, "outputs": fingerprint(outs),
                                      "adopted": False, "metrics": metrics,
                                      "at": datetime.datetime.now().isoformat(timespec="seconds")}
            report["ran"].append(st.name)
        man["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(man, indent=1) + "\n")
    return report


def _with(cfg: dict[str, Any], **sets: Any) -> dict[str, Any]:
    """A deep copy of cfg with dotted keys set."""
    import copy

    c = copy.deepcopy(cfg)
    for dotted, v in sets.items():
        cur = c
        *head, last = dotted.split("__")
        for k in head:
            cur = cur.setdefault(k, {})
        cur[last] = v
    return c


def _distil(stage: str, **sets: Any) -> Callable[[dict[str, Any], str, Path], dict[str, Any]]:
    def run(cfg: dict[str, Any], genre: str, run_dir: Path) -> dict[str, Any]:
        from experiments import distil

        return distil.main(_with(cfg, pack=genre, distil__stage=stage, **sets), run_dir)
    return run


def _embed(cfg: dict[str, Any], genre: str, run_dir: Path) -> dict[str, Any]:
    from experiments import embed

    return embed.main(_with(cfg, pack=genre), run_dir)


def pack_stages(cfg: dict[str, Any]) -> list[Stage]:
    """The lean pack, in order (the schema must exist and be frozen). GPU stages name their Colab job."""
    from engine.embed import embedding_slug

    P = lambda g: _path(cfg["paths"]["packs"]) / g  # noqa: E731
    D = lambda g: _path(cfg["paths"]["data_processed"]) / g  # noqa: E731
    E = lambda g: _path(cfg["paths"]["embeddings"]) / embedding_slug(cfg["models"]["embedder"]) / g  # noqa: E731
    S = lambda g: _path(cfg["paths"]["splits"]) / f"{g}.json"  # noqa: E731
    lab = lambda key: (lambda g: [_path(cfg["presence"]["labels"][key].format(genre=g))])  # noqa: E731
    mode = cfg["presence"].get("train_labels", "teacher")
    model = lambda g: [P(g) / "classifier" / f"linear_{mode}", P(g) / "prototypes", P(g) / "cascade.json"]  # noqa: E731
    tables, figures = _path(cfg["paths"]["tables"]), _path(cfg["paths"]["figures"])
    return [
        Stage("embed", _embed, inputs=lambda c, g: [D(g) / "sections.jsonl", D(g) / "documents.jsonl"],
              outputs=lambda c, g: [E(g) / "sections", E(g) / "titles"], config_keys=["models.embedder"], gpu=True,
              colab_job="an embed job (configs/embed.yaml --overlay colab)"),
        Stage("heads", _distil("heads"), inputs=lambda c, g: [E(g) / "sections", P(g) / "schema.yaml", P(g) / "label_sets.json"]
              + lab("teacher")(g) + lab("reference")(g), outputs=lambda c, g: [P(g) / "classifier" / f"linear_{mode}"],
              config_keys=["presence.head", "presence.train_labels", "presence.reference_split_seed", "distil.heads"]),
        Stage("prototypes", _distil("prototypes"), inputs=lambda c, g: [P(g) / "schema.yaml"] + lab("teacher")(g),
              outputs=lambda c, g: [P(g) / "prototypes"], config_keys=["presence.prototypes_per_type", "models.late_interaction"]),
        Stage("cascade", _distil("cascade"), inputs=lambda c, g: [P(g) / "classifier" / f"linear_{mode}", P(g) / "prototypes",
                                                                 E(g) / "sections"] + lab("reference")(g),
              outputs=lambda c, g: [P(g) / "cascade.json"], config_keys=["presence.cascade", "presence.band"]),
        Stage("section_probs", _distil("section_probs"), inputs=lambda c, g: model(g) + [D(g) / "sections.jsonl", E(g) / "sections", S(g)],
              outputs=lambda c, g: [P(g) / "section_probs"], config_keys=["distil.section_probs"], gpu=True,
              colab_job="section_probs_v1"),
        Stage("sentence_evidence", _distil("sentence_evidence"), inputs=lambda c, g: model(g) + [D(g) / "sentences.jsonl", S(g)],
              outputs=lambda c, g: [P(g) / "evidence"], config_keys=["distil.sentence_evidence"], gpu=True,
              colab_job="sentence_evidence_v1"),
        Stage("doc_presence", _distil("doc_presence", distil__doc_presence__write_profiles=True),
              inputs=lambda c, g: [P(g) / "section_probs", P(g) / "label_sets.json", P(g) / "cascade.json"] + lab("documents")(g),
              outputs=lambda c, g: [P(g) / "doc_presence.json", P(g) / "profiles"],
              config_keys=["distil.doc_presence.min_positives", "distil.doc_presence.C_grid", "presence.labels.documents"]),
        Stage("index", lambda c, g, r: build_index(c, g), inputs=lambda c, g: [E(g) / "sections", E(g) / "titles",
                                                                              D(g) / "documents.jsonl", D(g) / "sections.jsonl", S(g)],
              outputs=lambda c, g: [P(g) / "index" / n for n in ("vectors.npy", "payloads.jsonl", "index.json", "sections.npz")],
              config_keys=["index", "models.embedder.title_abstract_weight"]),
        Stage("evidence", lambda c, g, r: build_evidence(c, g), inputs=lambda c, g: [P(g) / "section_probs"],
              outputs=lambda c, g: [P(g) / "index" / "best_sections.npz"], config_keys=["index.splits"]),
        Stage("cohort_report", lambda c, g, r: cohort_report(c, g), inputs=lambda c, g: [P(g) / "index" / "vectors.npy", S(g)],
              outputs=lambda c, g: [tables / f"cohort_similarity_{g}.csv", figures / f"cohort_similarity_{g}.png"],
              config_keys=["cohort", "build.cohort_report"]),
        Stage("analyse_gate", lambda c, g, r: analyse_gate(c, g),
              inputs=lambda c, g: [P(g) / "doc_presence.json", P(g) / "profiles", P(g) / "evidence", P(g) / "index" / "vectors.npy"],
              outputs=lambda c, g: [tables / f"analyse_gate_{g}.csv"],
              config_keys=["build.analyse_gate", "cohort", "rank", "evidence", "explain.top_n"]),
    ]
