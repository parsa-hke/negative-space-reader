"""Embed every section and every title of the processed corpus with models.embedder (plan §5.1, run in the lab).

Writes data/embeddings/<model slug>/<genre>/{sections,titles}/ (EmbeddingStore: float16 vectors, ids, text sha1).
Document vectors (mean of sections + title/abstract, plan §0.2 #9) are derived from these at home, so the GPU work
is done once. Devices: run.devices (several -> one worker per GPU), else run.device. embed.reuse: true keeps every row
whose text sha1 is already in the store and encodes only the rest (after a corpus rebuild).

embed.mode (default encode):
  export   no model: the texts the store lacks (by sha1, each once) -> runs/<run>/colab_bundle.zip with manifest.json
           (model, Hugging Face snapshot, settings, library versions) and experiments/colab_embed.py, to encode on
           another machine (notebooks/colab_embed.ipynb on a free Colab T4)
  import   embed.import_from (that machine's colab_out.zip or folder) + the existing store -> the complete store for
           the current corpus, by sha1; refuses if any text has no vector; embed.import_check re-encodes that many
           imported texts locally and refuses if a cosine falls below embed.import_min_cosine
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data.schema import Document, Section, read_jsonl
from engine.embed import Embedder, embed_and_store, embedding_slug

REPO = Path(__file__).resolve().parent.parent


def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def _units(cfg: dict[str, Any], genre: str) -> list[tuple[str, list[Any], list[str], dict[str, Any]]]:
    """(unit, ids, texts, meta) for every stored unit of a genre: sections and titles."""
    processed = _path(cfg["paths"]["data_processed"]) / genre
    limit = cfg.get("embed", {}).get("limit")
    sections = list(read_jsonl(processed / "sections.jsonl", Section))[:limit]
    titled = [d for d in list(read_jsonl(processed / "documents.jsonl", Document))[:limit] if d.title]
    return [("sections", [[s.doc_id, s.idx] for s in sections], [s.text for s in sections],
             {"genre": genre, "unit": "section"}),
            ("titles", [d.id for d in titled], [d.title for d in titled], {"genre": genre, "unit": "title"})]


def _store_root(cfg: dict[str, Any]) -> Path:
    return _path(cfg["paths"].get("embeddings", "data/embeddings")) / embedding_slug(cfg["models"]["embedder"])


def _snapshot(model_id: str, revision: str | None) -> str | None:
    """The commit of the locally cached model (so the other machine loads the same weights), or the pinned one."""
    if revision:
        return revision
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True)).name
    except Exception:
        return None


def run_export(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import gzip
    import json
    import shutil

    from engine.embed import EmbeddingStore, text_sha1

    ecfg = cfg["models"]["embedder"]
    bundle = run_dir / "colab_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    versions = {}
    for mod in ("sentence_transformers", "transformers", "torch"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    manifest: dict[str, Any] = {"model": ecfg["id"], "revision": _snapshot(ecfg["id"], ecfg.get("revision")),
                                "max_length": ecfg.get("max_length", 1024), "normalize": ecfg.get("normalize", True),
                                "truncate_dim": ecfg.get("truncate_dim"), "versions": versions, "files": {}}
    metrics: dict[str, Any] = {"model": ecfg["id"], "revision": manifest["revision"]}
    genres = [cfg["pack"]] if cfg.get("pack") else list(cfg["genres"])
    for genre in genres:
        for unit, _, texts, _ in _units(cfg, genre):
            path = _store_root(cfg) / genre / unit
            have = set(EmbeddingStore.load(path).sha1) if (path / "rows.jsonl").exists() else set()
            missing: dict[str, str] = {}
            for t in texts:
                h = text_sha1(t)
                if h not in have and h not in missing:
                    missing[h] = t
            metrics[f"{genre}/{unit}_missing"] = len(missing)
            if not missing:
                continue
            name = f"{genre}_{unit}"
            with gzip.open(bundle / f"{name}.jsonl.gz", "wt", encoding="utf-8") as f:
                for h, t in missing.items():
                    f.write(json.dumps({"sha1": h, "text": t}, ensure_ascii=False) + "\n")
            manifest["files"][name] = {"file": f"{name}.jsonl.gz", "rows": len(missing),
                                       "chars": sum(len(t) for t in missing.values())}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.copy(REPO / "experiments" / "colab_embed.py", bundle / "colab_embed.py")
    shutil.make_archive(str(run_dir / "colab_bundle"), "zip", bundle)
    metrics["bundle"] = str(run_dir / "colab_bundle.zip")
    print(f"[embed export] {sum(v['rows'] for v in manifest['files'].values())} texts -> {metrics['bundle']}",
          flush=True)
    return metrics


def run_import(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import json
    import tempfile
    import zipfile

    import numpy as np

    from engine.embed import EmbeddingStore, text_sha1

    ecfg, emb = cfg["models"]["embedder"], cfg["embed"]
    src = _path(emb["import_from"])
    tmp = None
    if src.suffix == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp.name)
        src = Path(tmp.name)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    imported: dict[str, Any] = {}
    for name in meta.get("files", {}):
        rows = [json.loads(line)["sha1"] for line in open(src / f"rows_{name}.jsonl", encoding="utf-8")]
        X = np.load(src / f"vectors_{name}.f16.npy").astype(np.float32)
        if len(rows) != len(X):
            raise ValueError(f"{name}: {len(rows)} rows but {len(X)} vectors")
        imported.update(zip(rows, X))
    provenance = {k: meta.get(k) for k in ("device", "gpu", "dtype", "torch", "transformers", "sentence_transformers")}
    provenance["files"] = meta.get("files")
    metrics: dict[str, Any] = {"model": ecfg["id"], "imported_from": str(emb["import_from"]), **{
        f"import_{k}": v for k, v in provenance.items() if k != "files"}}
    check_texts: list[tuple[str, Any]] = []
    genres = [cfg["pack"]] if cfg.get("pack") else list(cfg["genres"])
    for genre in genres:
        for unit, ids, texts, umeta in _units(cfg, genre):
            path = _store_root(cfg) / genre / unit
            old = EmbeddingStore.load(path) if (path / "rows.jsonl").exists() else None
            found = old.lookup(texts) if old is not None else [None] * len(texts)
            vectors, n_imported, missing = [], 0, 0
            for t, v in zip(texts, found):
                if v is None and (h := text_sha1(t)) in imported:
                    v = imported[h]
                    n_imported += 1
                    if len(check_texts) < (emb.get("import_check") or 0):
                        check_texts.append((t, v))
                missing += v is None
                vectors.append(v)
            if missing:
                raise ValueError(f"{genre}/{unit}: {missing} texts have no vector in the store or the import")
            if old is None and not texts:
                continue
            store_meta = {**(old.meta if old is not None else {}), **umeta, "reused_rows": len(texts) - n_imported,
                          "imported_rows": n_imported, "imported_from": provenance}
            for k in ("rows", "dim"):
                store_meta.pop(k, None)
            if n_imported or old is None or len(old.ids) != len(texts):
                EmbeddingStore.save(path, ids, texts, np.asarray(vectors, dtype=np.float32), store_meta)
            metrics.update({f"{genre}/{unit}_rows": len(texts), f"{genre}/{unit}_imported_rows": n_imported})
    if check_texts:  # the imported vectors must agree with what this machine's encoder gives for the same texts
        from experiments.run import resolve_device

        local = Embedder(ecfg["id"], devices=[resolve_device(cfg["run"].get("device", "auto"))],
                         dtype=ecfg.get("dtype", "float16"), max_length=ecfg.get("max_length", 1024),
                         batch_size=ecfg.get("batch_size", 32), normalize=ecfg.get("normalize", True),
                         truncate_dim=ecfg.get("truncate_dim"), revision=ecfg.get("revision"))
        L = local.encode([t for t, _ in check_texts])
        R = np.asarray([v for _, v in check_texts], dtype=np.float32)
        cos = (L * R).sum(1) / (np.linalg.norm(L, axis=1) * np.linalg.norm(R, axis=1))
        metrics.update({"import_check_rows": len(cos), "import_check_min_cosine": round(float(cos.min()), 5),
                        "import_check_mean_cosine": round(float(cos.mean()), 5)})
        if cos.min() < emb.get("import_min_cosine", 0.99):
            raise ValueError(f"imported vectors disagree with the local encoder: min cosine {cos.min():.4f}")
    if tmp is not None:
        tmp.cleanup()
    print(f"[embed import] {metrics}", flush=True)
    return metrics


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from experiments.run import resolve_device

    mode = cfg.get("embed", {}).get("mode") or "encode"
    if mode == "export":
        return run_export(cfg, run_dir)
    if mode == "import":
        return run_import(cfg, run_dir)

    ecfg = cfg["models"]["embedder"]
    devices = cfg["run"].get("devices") or [resolve_device(cfg["run"].get("device", "auto"))]
    embedder = Embedder(ecfg["id"], devices=devices, dtype=ecfg.get("dtype", "float16"),
                        max_length=ecfg.get("max_length", 1024), batch_size=ecfg.get("batch_size", 32),
                        normalize=ecfg.get("normalize", True), truncate_dim=ecfg.get("truncate_dim"),
                        revision=ecfg.get("revision"))
    out_root = _path(cfg["paths"].get("embeddings", "data/embeddings")) / embedding_slug(ecfg)
    limit = cfg.get("embed", {}).get("limit")
    reuse = bool(cfg.get("embed", {}).get("reuse"))
    metrics: dict[str, Any] = {"model": ecfg["id"], "devices": devices}
    genres = [cfg["pack"]] if cfg.get("pack") else list(cfg["genres"])
    for genre in genres:
        processed = _path(cfg["paths"]["data_processed"]) / genre
        sections = list(read_jsonl(processed / "sections.jsonl", Section))[:limit]
        stats = embed_and_store(embedder, [[s.doc_id, s.idx] for s in sections], [s.text for s in sections],
                                out_root / genre / "sections", {"genre": genre, "unit": "section"}, reuse=reuse)
        metrics.update({f"{genre}/sections_{k}": v for k, v in stats.items()})
        docs = list(read_jsonl(processed / "documents.jsonl", Document))[:limit]
        titled = [d for d in docs if d.title]
        stats = embed_and_store(embedder, [d.id for d in titled], [d.title for d in titled],
                                out_root / genre / "titles", {"genre": genre, "unit": "title"}, reuse=reuse)
        metrics.update({f"{genre}/titles_{k}": v for k, v in stats.items()})
        print(f"[embed {genre}] {metrics[f'{genre}/sections_rows']} sections in {metrics[f'{genre}/sections_seconds']}s",
              flush=True)
    metrics["output"] = str(out_root)
    return metrics
