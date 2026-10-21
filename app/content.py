"""The numbers the demo's About pages show, copied from the recorded tables (plan 7.4's rule: never typed) into
`serve.examples_dir`/results.json by `python -m app.examples`, so a Space needs neither the tables nor the runs.

    results = build_results(cfg)          # {corpus, schema, synthetic, revision, gate, llm_gap, speed, presence, sources}

Every entry names the table it came from. A missing table leaves its entry out (the page then says so) rather than
inventing a number.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _ci(text: str) -> list[float] | None:
    m = re.findall(r"[-+]?\d*\.?\d+", text or "")
    return [float(m[-2]), float(m[-1])] if len(m) >= 2 else None


def _signed(text: str) -> dict[str, Any] | None:
    """'+0.0110 [-0.0106, +0.0326]' -> {diff, ci}."""
    m = re.findall(r"[-+]?\d*\.?\d+", text or "")
    return {"diff": float(m[0]), "ci": [float(m[1]), float(m[2])]} if len(m) >= 3 else None


def build_results(cfg: dict[str, Any], genres: list[str] | None = None) -> dict[str, Any]:
    import yaml

    from packs.builder import _path

    tables = _path(cfg["paths"]["tables"])
    packs = _path(cfg["paths"]["packs"])
    genres = list(genres or cfg["genres"])
    out: dict[str, Any] = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "genres": genres, "sources": []}

    def use(name: str) -> Path | None:
        p = tables / name
        if p.is_file():
            out["sources"].append(name)
            return p
        return None

    p = use("corpus_stats.csv")
    if p:
        corpus: dict[str, Any] = {}
        for r in _rows(p):
            g = r["genre"]
            c = corpus.setdefault(g, {"documents": 0, "entities": 0, "splits": {}})
            c["documents"] += int(r["documents"])
            c["entities"] += int(r["entities"])
            c["splits"][r["split"]] = int(r["documents"])
            if r["split"] == "train":
                c.update(indexed=int(r["documents"]), date_min=r["date_min"], date_max=r["date_max"],
                         sections_per_doc_median=float(r["sections_per_doc_median"]),
                         chars_per_doc_median=float(r["chars_per_doc_median"]))
        out["corpus"] = corpus
    schema = {}
    for g in genres:
        f = packs / g / "schema.yaml"
        if f.is_file():
            types = yaml.safe_load(f.read_text(encoding="utf-8"))["types"]
            schema[g] = {"types": len(types), "examples": [t["id"] for t in types[:6]],
                         "weighted": sum(1 for t in types if float(t.get("severity_weight") or 1) != 1)}
    out["schema"] = schema
    for key, stem, regime in (("synthetic", "bench_synth_{g}.csv", "single_type"), ("revision", "bench_revision_{g}.csv", "revision")):
        block: dict[str, Any] = {}
        for g in genres:
            p = use(stem.format(g=g))
            if not p:
                continue
            block[g] = {}
            for r in _rows(p):
                if r["regime"] == regime and r["metric"] in ("recall@1", "recall@5", "recall@10", "mrr") and \
                        r["system"] in ("genre_uniform", "level1", "random", "level1 - genre_uniform"):
                    block[g].setdefault(r["system"], {})[r["metric"]] = {"value": float(r["value"]), "ci": [float(r["ci_low"]), float(r["ci_high"])],
                                                                           "n": int(float(r["n"]))}
        out[key] = block
    gate: dict[str, Any] = {}
    for g in genres:
        p = use(f"cohort_diagnosis_{g}.csv")
        if not p:
            continue
        for r in _rows(p):
            if r["variant"] == "default":
                gate[g] = {"edited_minus_uniform": _signed(r.get("single_type/edited_minus_uniform", "")),
                           "original_minus_uniform": _signed(r.get("single_type/original_minus_uniform", "")),
                           "leak": _signed(r.get("single_type/leak_original_minus_edited", "")),
                           "within_doc_auc": float(r["within_doc_auc"]) if r.get("within_doc_auc") else None}
        uniform = next((r for r in _rows(p) if r["variant"] == "genre_uniform"), None)
        if uniform and g in gate:
            gate[g]["uniform_within_doc_auc"] = float(uniform["within_doc_auc"]) if uniform.get("within_doc_auc") else None
    out["gate"] = gate
    llm: dict[str, Any] = {}
    for g in genres:
        p = use(f"llm_gap_candidate_{g}.csv")
        if not p:
            continue
        llm[g] = {}
        for r in _rows(p):
            llm[g][r["system"]] = {"n": int(r["n"]), **{k: {"value": float(r[k]), "ci": _ci(r.get(f"{k}_ci", ""))}
                                                      for k in ("recall@1", "recall@5", "recall@10", "mrr@10") if r.get(k)}}
    out["llm_gap"] = llm
    speed: dict[str, Any] = {}
    for g in genres:
        p = use(f"analyse_gate_{g}.csv")
        if not p:
            continue
        totals = sorted(float(r["total"]) for r in _rows(p))
        speed[g] = {"documents": len(totals), "max_seconds": totals[-1], "median_seconds": totals[len(totals) // 2],
                    "sections": [int(r["sections"]) for r in _rows(p)]}
    out["speed"] = speed
    p = use("presence_gate.csv")
    if p:
        out["presence"] = {r["genre"]: {"macro_f1": float(r["macro_f1"]), "teacher_ceiling": float(r["teacher_ceiling"]),
                                        "share_of_ceiling": float(r["share_of_ceiling"]), "gate": r["f1_gate"],
                                        "model": r["presence_model"]} for r in _rows(p) if r["genre"] in genres}
    return out


def write_results(cfg: dict[str, Any], folder: str | Path, genres: list[str] | None = None) -> dict[str, Any]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    res = build_results(cfg, genres)
    (folder / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res


def read_results(folder: str | Path) -> dict[str, Any] | None:
    p = Path(folder) / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
