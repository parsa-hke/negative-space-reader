"""OmitBench synthetic and real runs (plan §6.1-6.3, §9.3). Stages (`bench.stage`):

  probs   every test section's and sentence's presence probabilities -> <run>/<genre>/{section_probs,sentence_probs}
          (a GPU pass: Colab job bench_probs_v1; the Mac for a limited smoke run). Never written into a pack.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any


def _path(p: str | Path) -> Path:
    from experiments.distil import _path as p_

    return p_(p)


def _genres(cfg: dict[str, Any]) -> list[str]:
    return [cfg["pack"]] if cfg.get("pack") else list(cfg["genres"])


def run_probs(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import numpy as np

    import engine.embed as E
    from bench.probs import bench_order, build_sentence_probs, check_outside_packs
    from experiments.distil import _line_ranges, _presence_model, load_colbert
    from packs.profiles import build_section_probs

    if (cfg["presence"].get("cascade") or {}).get("reranker"):
        raise NotImplementedError("bench probabilities with the reranker are not built in the lean tier")
    bc = cfg["bench"]
    pc = bc.get("probs") or {}
    colbert = load_colbert(cfg)
    ecfg = cfg["models"]["embedder"]
    embedder = E.Embedder(ecfg["id"], devices=[colbert.device], dtype=ecfg.get("dtype", "float16"),
                          max_length=ecfg.get("max_length", 1024), batch_size=ecfg.get("batch_size", 32),
                          normalize=True, revision=ecfg.get("revision"))
    metrics: dict[str, Any] = {"machine": f"{platform.machine()} {platform.system()}", "device": colbert.device}
    for genre in _genres(cfg):
        out = Path(run_dir) / genre
        check_outside_packs(out, _path(cfg["paths"]["packs"]))
        pm = _presence_model(cfg, genre, colbert, ["test"], pc.get("limit"), allow_test=True,
                             order=lambda ids: bench_order(ids, seed=bc["seed"]))
        docs, types = pm["docs"], pm["types"]
        chunk = pc.get("chunk_documents", 250)
        log = lambda m, s: print(f"[bench probs {genre} {s}] {m}", flush=True)  # noqa: E731
        sec = build_section_probs(docs, lambda d: pm["sections"][d], pm["score"], types, out / "section_probs",
                                  chunk_documents=chunk, log=lambda m: log(m, "sections"))
        path = _path(cfg["paths"]["data_processed"]) / genre / "sentences.jsonl"
        ranges = _line_ranges(path, {d["id"] for d in docs})
        fh = open(path, "rb")

        def sentences_of(doc_id: str) -> list[tuple[int, int, str]]:
            rows = []
            for a, b in ranges.get(doc_id, []):
                fh.seek(a)
                rows += [json.loads(line) for line in fh.read(b - a).splitlines() if line.strip()]
            return [(int(r["section_idx"]), int(r["idx"]), r["text"]) for r in rows]

        def score(texts: list[str]) -> Any:
            clf = pm["heads"].predict_proba(np.asarray(embedder.encode(texts)))
            return pm["model"].predict(clf, pm["scorer"].score(texts), np.full(clf.shape, np.nan))

        sent = build_sentence_probs(docs, sentences_of, score, types, out / "sentence_probs", chunk_documents=chunk,
                                    log=lambda m: log(m, "sentences"))
        fh.close()
        (out / "documents.json").write_text(json.dumps([d["id"] for d in docs]) + "\n")
        metrics.update({f"{genre}/documents": len(docs), **{f"{genre}/sections_{k}": v for k, v in sec.items()},
                        **{f"{genre}/sentences_{k}": v for k, v in sent.items()}})
    return metrics


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    stage = (cfg.get("bench") or {}).get("stage")
    if stage == "probs":
        return run_probs(cfg, run_dir)
    raise ValueError(f"unknown bench.stage {stage!r} (probs)")
