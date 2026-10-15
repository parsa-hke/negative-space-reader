"""Embed every section and every title of the processed corpus with models.embedder (plan §5.1, run in the lab).

Writes data/embeddings/<model slug>/<genre>/{sections,titles}/ (EmbeddingStore: float16 vectors, ids, text sha1).
Document vectors (mean of sections + title/abstract, plan §0.2 #9) are derived from these at home, so the GPU work
is done once. Devices: run.devices (several -> one worker per GPU), else run.device.
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


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from experiments.run import resolve_device

    ecfg = cfg["models"]["embedder"]
    devices = cfg["run"].get("devices") or [resolve_device(cfg["run"].get("device", "auto"))]
    embedder = Embedder(ecfg["id"], devices=devices, dtype=ecfg.get("dtype", "float16"),
                        max_length=ecfg.get("max_length", 1024), batch_size=ecfg.get("batch_size", 32),
                        normalize=ecfg.get("normalize", True), truncate_dim=ecfg.get("truncate_dim"),
                        revision=ecfg.get("revision"))
    out_root = _path(cfg["paths"].get("embeddings", "data/embeddings")) / embedding_slug(ecfg)
    limit = cfg.get("embed", {}).get("limit")
    metrics: dict[str, Any] = {"model": ecfg["id"], "devices": devices}
    genres = [cfg["pack"]] if cfg.get("pack") else list(cfg["genres"])
    for genre in genres:
        processed = _path(cfg["paths"]["data_processed"]) / genre
        sections = list(read_jsonl(processed / "sections.jsonl", Section))[:limit]
        stats = embed_and_store(embedder, [[s.doc_id, s.idx] for s in sections], [s.text for s in sections],
                                out_root / genre / "sections", {"genre": genre, "unit": "section"})
        metrics.update({f"{genre}/sections_{k}": v for k, v in stats.items()})
        docs = list(read_jsonl(processed / "documents.jsonl", Document))[:limit]
        titled = [d for d in docs if d.title]
        stats = embed_and_store(embedder, [d.id for d in titled], [d.title for d in titled],
                                out_root / genre / "titles", {"genre": genre, "unit": "title"})
        metrics.update({f"{genre}/titles_{k}": v for k, v in stats.items()})
        print(f"[embed {genre}] {metrics[f'{genre}/sections_rows']} sections in {metrics[f'{genre}/sections_seconds']}s",
              flush=True)
    metrics["output"] = str(out_root)
    return metrics
