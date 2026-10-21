"""Demo examples for the app (plan 7.1): test-split corpus documents and OmitBench-synthetic instances with the
chat-model answers cached by the 7.3 runs, written once to `serve.examples_dir` (app/examples/<genre>.json) so the
app, and a Space, needs neither the corpus nor the runs.

    python -m app.examples --config configs/serve.yaml      # rebuild app/examples/*.json from the corpus and runs

An example: {name, kind: corpus | benchmark, pack, doc_id, entity_id, category, date, markdown, instance_id?,
removed? (the types the benchmark deleted), chat? ({label: {ranked: [...], rank_first: int | None}})}. Benchmark
documents are the edited documents of `baselines.synth_run`, rebuilt as Markdown like corpus documents, so the app
segments them as the benchmark did. The chat answers are copies of the recorded per-instance results, never calls.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any


def _processed(cfg: dict[str, Any], genre: str) -> Path:
    from packs.builder import _path

    return _path(cfg["paths"]["data_processed"]) / genre


def _read_sections(processed: Path, want: set[str]) -> dict[str, list[Any]]:
    from data.schema import Section

    secs: dict[str, list[Section]] = {}
    with open(processed / "sections.jsonl", encoding="utf-8") as f:
        for line in f:
            d = line[12:line.find('"', 12)]  # {"doc_id": "<id>", ...: read the id before parsing the line
            if d in want:
                secs.setdefault(d, []).append(Section(**json.loads(line)))
    return {d: sorted(s, key=lambda x: x.idx) for d, s in secs.items()}


def _read_documents(processed: Path, want: set[str]) -> dict[str, Any]:
    from data.schema import Document

    docs = {}
    with open(processed / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in want:
                docs[row["id"]] = Document(**row)
    return docs


def _name(kind: str, genre: str, title: str | None, ident: str) -> str:
    return f"{'Corpus' if kind == 'corpus' else 'Benchmark'} · {genre}: {(title or ident)[:60]}"


def corpus_examples(cfg: dict[str, Any], genre: str, ids: list[str] | None, *, per_genre: int = 3,
                    seed: int = 0) -> list[dict[str, Any]]:
    """Named test-split documents, else per_genre documents of 6-30 sections drawn with the seed."""
    from engine.pipeline import document_markdown
    from packs.builder import _path

    processed = _processed(cfg, genre)
    split = json.loads((_path(cfg["paths"]["splits"]) / f"{genre}.json").read_text())["doc_ids"]["test"]
    pool = list(ids or []) or random.Random(seed).sample(sorted(split), min(200, len(split)))
    secs = _read_sections(processed, set(pool))
    docs = _read_documents(processed, set(pool))
    missing = [d for d in (ids or []) if d not in docs]
    if missing:
        raise KeyError(f"{genre}: documents not in the corpus: {missing}")
    chosen = list(ids or []) or [d for d in pool if d in docs and 6 <= len(secs.get(d, [])) <= 30][:per_genre]
    out = []
    for d in chosen:
        doc = docs[d]
        out.append({"name": _name("corpus", genre, doc.title, d), "kind": "corpus", "pack": genre, "doc_id": d,
                    "entity_id": doc.entity_id, "category": doc.category, "date": doc.date,
                    "markdown": document_markdown(doc, secs[d])})
    return out


def benchmark_examples(cfg: dict[str, Any], genre: str, instance_ids: list[str],
                       chat_runs: list[dict[str, str]]) -> list[dict[str, Any]]:
    """OmitBench-synthetic instances (edited document, deleted types) with the ranked lists each chat run recorded."""
    from data.segment import write_sections_markdown

    if not instance_ids:
        return []
    folder = Path(str(cfg["baselines"]["synth_run"]).format(genre=genre))
    want = set(instance_ids)
    inst = {f"{r['regime']}:{r['doc_id']}": r for r in map(json.loads, open(folder / "instances.jsonl", encoding="utf-8"))
            if f"{r['regime']}:{r['doc_id']}" in want}
    edited = {}
    with open(folder / "edited.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["instance_id"] in want:
                edited[r["instance_id"]] = r
    missing = [i for i in instance_ids if i not in inst or i not in edited]
    if missing:
        raise KeyError(f"{genre}: instances not in {folder}: {missing}")
    chat: dict[str, dict[str, dict[str, Any]]] = {i: {} for i in instance_ids}
    for run in chat_runs:
        path = Path(run["run"]) / genre / "per_instance.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path}: the chat run has no results for {genre}")
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["instance_id"] in want:
                ranked = [t for t in (r.get("ranked") or "").split("|") if t]
                chat[r["instance_id"]][run["label"]] = {"ranked": ranked,
                                                        "rank_first": int(r["rank_first"]) if r.get("rank_first") else None}
    out = []
    for i in instance_ids:
        e, meta = edited[i], inst[i]
        md = write_sections_markdown(e.get("title"), [(s.get("heading"), s["text"]) for s in e["sections"]])
        out.append({"name": _name("benchmark", genre, e.get("title"), i), "kind": "benchmark", "pack": genre,
                    "doc_id": meta["doc_id"], "entity_id": meta.get("entity_id"), "category": meta.get("category"),
                    "date": meta.get("date"), "markdown": md, "instance_id": i, "regime": meta["regime"],
                    "removed": list(meta["ground_truth"]), "chat": chat[i]})
    return out


def build_examples(cfg: dict[str, Any], genres: list[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    scfg = cfg.get("serve") or {}
    out = {}
    for g in genres or list(scfg.get("packs") or cfg["genres"]):
        ex = corpus_examples(cfg, g, (scfg.get("examples") or {}).get(g), per_genre=scfg.get("examples_per_genre", 3),
                             seed=cfg["run"].get("seed", 0))
        ex += benchmark_examples(cfg, g, list((scfg.get("bench_examples") or {}).get(g) or []), list(scfg.get("chat_runs") or []))
        out[g] = ex
    return out


def write_examples(cfg: dict[str, Any], folder: str | Path, genres: list[str] | None = None) -> dict[str, int]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    built = build_examples(cfg, genres)
    for g, ex in built.items():
        (folder / f"{g}.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {g: len(ex) for g, ex in built.items()}


def read_examples(folder: str | Path, genres: list[str]) -> list[dict[str, Any]] | None:
    """The written examples of every genre, or None when any genre's file is missing (the app then builds them)."""
    folder = Path(folder)
    if not all((folder / f"{g}.json").is_file() for g in genres):
        return None
    out: list[dict[str, Any]] = []
    for g in genres:
        out += json.loads((folder / f"{g}.json").read_text(encoding="utf-8"))
    return out


def load_examples(cfg: dict[str, Any], genres: list[str], *, per_genre: int = 3, seed: int = 0) -> list[dict[str, Any]]:
    """Examples for the app: the written files when present, else built from the corpus (and the runs, for benchmark
    examples, when named)."""
    scfg = cfg.get("serve") or {}
    written = read_examples(scfg.get("examples_dir", "app/examples"), genres) if scfg.get("examples_dir") else None
    if written is not None:
        return written
    out: list[dict[str, Any]] = []
    for g in genres:
        out += corpus_examples(cfg, g, (scfg.get("examples") or {}).get(g), per_genre=per_genre, seed=seed)
        out += benchmark_examples(cfg, g, list((scfg.get("bench_examples") or {}).get(g) or []), list(scfg.get("chat_runs") or []))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from experiments.config import load_config

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/serve.yaml")
    p.add_argument("--set", action="append", default=[], dest="sets")
    a = p.parse_args(argv)
    cfg = load_config(a.config, [], a.sets)
    cfg["run"]["seed"] = (cfg["run"].get("seeds") or [0])[0]
    from app.content import write_results

    folder = (cfg.get("serve") or {}).get("examples_dir", "app/examples")
    counts = write_examples(cfg, folder)
    res = write_results(cfg, folder)
    print(json.dumps({"examples": counts, "results_sources": res["sources"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
