"""Raw corpora -> data/processed/<genre>/{documents,sections,sentences}.jsonl + splits + stats (plan §2.8).

Launched through the entry point (task `build_data`, configs/data.yaml):

    python -m experiments.run --config configs/data.yaml --overlay local --overlay smoke
    python -m experiments.run --config configs/data.yaml --overlay local --set genres=[papers]

Per genre: candidate index -> entity-level splits (forced and hashed) -> stratified sample per split (year x category)
-> exact and near-duplicate removal (no near-copy of a test document survives in train/dev) -> segmentation ->
sentences -> revision pairs for test entities -> jsonl, split manifest, tables in reports/tables/, loader previews in
reports/figures/. Genre-specific knowledge is limited to which loaders feed a genre (`data_sources.<genre>`).
"""

from __future__ import annotations

import csv
import html
import json
import random
import statistics as st
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from data.dedup import duplicate_rates, exact_key, near_duplicate_clusters, signature
from data.revisions import chain_pairs
from data.schema import ParsedDocument, RevisionPair, Sentence, write_jsonl
from data.segment import split_sentences
from data.splits import assign_split, resolve_cross_split_duplicates, write_manifest

REPO = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------------------ helpers
def _p(cfg: dict[str, Any], key: str) -> Path:
    path = Path(cfg["paths"][key]) if key in cfg["paths"] else Path(key)
    return path if path.is_absolute() else REPO / path


def split_sizes(cfg: dict[str, Any]) -> dict[str, int]:
    """Train size is corpus.max_documents_per_genre; dev and test scale with the split fractions."""
    n_train = int(cfg["corpus"]["max_documents_per_genre"])
    fr = cfg["splits"]
    return {"train": n_train, "dev": max(1, round(n_train * fr["dev"] / fr["train"])),
            "test": max(1, round(n_train * fr["test"] / fr["train"]))}


def stratified_sample(items: list[Any], n: int, stratum: Callable[[Any], Any], rng: random.Random) -> list[Any]:
    """Proportional allocation over strata (largest remainder), uniform within a stratum."""
    if n >= len(items):
        return list(items)
    groups: dict[Any, list[Any]] = defaultdict(list)
    for it in items:
        groups[stratum(it)].append(it)
    total = len(items)
    quota = {k: n * len(v) / total for k, v in groups.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (quota[k] - alloc[k], str(k)), reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    out = []
    for k in sorted(groups, key=str):
        out += rng.sample(sorted(groups[k], key=str), min(alloc[k], len(groups[k])))
    return out


def with_sentences(docs: Iterable[ParsedDocument], min_chars: int) -> Iterable[tuple[ParsedDocument, list[Sentence]]]:
    for d in docs:
        sents = []
        for s in d.sections:
            for i, text in enumerate(split_sentences(s.text, min_chars=min_chars)):
                sents.append(Sentence(doc_id=d.document.id, section_idx=s.idx, idx=i, text=text))
        yield d, sents


def write_corpus(out_dir: Path, docs: list[ParsedDocument], min_sentence_chars: int) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(with_sentences(docs, min_sentence_chars))
    n_docs = write_jsonl(out_dir / "documents.jsonl", (d.document for d, _ in pairs))
    n_secs = write_jsonl(out_dir / "sections.jsonl", (s for d, _ in pairs for s in d.sections))
    n_sents = write_jsonl(out_dir / "sentences.jsonl", (s for _, ss in pairs for s in ss))
    return {"documents": n_docs, "sections": n_secs, "sentences": n_sents,
            "_per_doc": {d.document.id: (len(d.sections), len(ss)) for d, ss in pairs}}


def dedup_sample(docs: list[ParsedDocument], split_of: dict[str, str], dcfg: dict[str, Any],
                 protected: set[str] | None = None) -> tuple[list[ParsedDocument], dict]:
    """Exact duplicates across entities, then near-duplicate clusters: cross-split twins dropped (test wins), and at
    most dedup.max_per_cluster_per_year documents of different entities per cluster and year in train. `protected`
    documents (human-labelled data) are never dropped: their twins in other splits are dropped instead."""
    protected = protected or set()
    report: dict[str, Any] = {"input": len(docs)}
    by_key: dict[str, ParsedDocument] = {}
    exact_drop = set()
    for d in sorted(docs, key=lambda d: (d.document.id not in protected,
                                          -{"test": 2, "dev": 1, "train": 0}[split_of[d.document.id]], d.document.id)):
        key = exact_key(d.text())
        if key in by_key and by_key[key].document.entity_id != d.document.entity_id and d.document.id not in protected:
            exact_drop.add(d.document.id)
        else:
            by_key.setdefault(key, d)
    docs = [d for d in docs if d.document.id not in exact_drop]
    report["exact_duplicates_dropped"] = len(exact_drop)
    sigs = {d.document.id: signature(d.text(), num_perm=dcfg["num_perm"], k=dcfg["shingle_size"]) for d in docs}
    cluster = near_duplicate_clusters(sigs, threshold=dcfg["jaccard_threshold"], num_perm=dcfg["num_perm"])
    report.update({f"near_{k}": v for k, v in duplicate_rates(cluster).items()})
    cross = resolve_cross_split_duplicates(cluster, split_of)
    members: dict[Any, list[str]] = defaultdict(list)
    for doc_id, c in cluster.items():
        members[c].append(doc_id)
    for doc_id in [d for d in cross if d in protected]:
        cross.discard(doc_id)
        cross.update(m for m in members[cluster[doc_id]] if m not in protected and split_of[m] != split_of[doc_id])
    report["cross_split_dropped"] = len(cross)
    entity = {d.document.id: d.document.entity_id for d in docs}
    year = {d.document.id: (d.document.date or "")[:4] for d in docs}
    seen: dict[tuple, set[str]] = defaultdict(set)
    cap_drop = set()
    for doc_id in sorted(cluster, key=lambda x: (cluster[x], x)):
        if doc_id in cross or split_of[doc_id] != "train":
            continue
        key = (cluster[doc_id], year[doc_id])
        if entity[doc_id] in seen[key]:
            continue
        if len(seen[key]) >= dcfg["max_per_cluster_per_year"]:
            cap_drop.add(doc_id)
            continue
        seen[key].add(entity[doc_id])
    report["train_cluster_cap_dropped"] = len(cap_drop)
    kept = [d for d in docs if d.document.id not in cross | cap_drop]
    report["output"] = len(kept)
    return kept, report


def preview_html(docs: list[ParsedDocument], path: Path, title: str) -> None:
    """Section boundaries drawn as boxes, sentence boundaries as thin rules (plan §2.4)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title><style>"
             "body{font:14px/1.45 system-ui;margin:24px;max-width:960px}"
             ".sec{border:1px solid #888;border-radius:6px;margin:10px 0;padding:8px 12px}"
             ".sec:nth-child(odd){background:#f4f6fb}.h{font-weight:600;color:#1d3f8f}.meta{color:#555;font-size:12px}"
             ".s{border-bottom:1px dashed #bbb;padding:2px 0}</style>", f"<h1>{html.escape(title)}</h1>"]
    for d in docs:
        doc = d.document
        parts.append(f"<h2>{html.escape(doc.id)}</h2><div class='meta'>entity {html.escape(doc.entity_id)} · version "
                     f"{html.escape(str(doc.version))} · date {doc.date} · category {html.escape(str(doc.category))} · "
                     f"{len(d.sections)} sections</div><div>")
        for s in d.sections:
            sents = "".join(f"<div class='s'>{html.escape(x)}</div>" for x in split_sentences(s.text))
            parts.append(f"<div class='sec'><div class='h'>[{s.idx}] {html.escape(s.heading or '(no heading)')}</div>"
                         f"<div class='meta'>chars {s.char_span[0]}–{s.char_span[1]}</div>{sents}</div>")
        parts.append("</div>")
    path.write_text("".join(parts), encoding="utf-8")


def corpus_stats_rows(genre: str, docs: list[ParsedDocument], split_of: dict[str, str],
                      per_doc: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "dev", "test", "all"):
        ds = [d for d in docs if split == "all" or split_of[d.document.id] == split]
        if not ds:
            continue
        n_secs = [per_doc[d.document.id][0] for d in ds]
        sents_per_sec = [per_doc[d.document.id][1] / max(1, per_doc[d.document.id][0]) for d in ds]
        dates = sorted(d.document.date for d in ds if d.document.date)
        rows.append({
            "genre": genre, "split": split, "documents": len(ds), "entities": len({d.document.entity_id for d in ds}),
            "sections_per_doc_mean": round(st.mean(n_secs), 2), "sections_per_doc_median": st.median(n_secs),
            "sentences_per_section_mean": round(st.mean(sents_per_sec), 2),
            "chars_per_doc_median": st.median(len(d.text()) for d in ds),
            "date_min": dates[0] if dates else "", "date_max": dates[-1] if dates else "",
            "sources": json.dumps(dict(Counter(d.document.source for d in ds))),
            "top_categories": json.dumps(dict(Counter(d.document.category for d in ds).most_common(10))),
        })
    return rows


def write_table(path: Path, rows: list[dict[str, Any]], *, key: Callable[[dict], Any] | None = None) -> None:
    """Write rows to a CSV, replacing earlier rows of the same genre (so genres can be built separately)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    genres = {r["genre"] for r in rows}
    old = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if r.get("genre") not in genres]
    allrows = old + rows
    fields = list(dict.fromkeys(k for r in allrows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(allrows, key=key) if key else allrows)


# ------------------------------------------------------------------------------------ genre A
def build_privacy(cfg: dict[str, Any], genre: str = "privacy") -> dict[str, Any]:
    from data.loaders import opp115, privacy_pl

    src = cfg["data_sources"][genre]
    seg = cfg["segment"]
    seg_kw = dict(min_section_chars=seg["min_section_chars"], max_section_chars=seg["max_section_chars"],
                  fallback_section_chars=seg["fallback_section_chars"])
    seed, fr = cfg["splits"]["seed"], {k: cfg["splits"][k] for k in ("train", "dev", "test")}
    t0 = time.perf_counter()

    opp_docs = list(opp115.iter_documents(_p(cfg, src["opp115"]["dir"]), genre=genre))
    forced = {d.document.entity_id: src["opp115"]["split"] for d in opp_docs}

    pl = src["privacy_pl"]
    con = privacy_pl.connect(_p(cfg, pl["db"]))
    rows = privacy_pl.snapshot_index(con, analysis_subcorpus_only=pl["analysis_subcorpus_only"],
                                     file_types=pl["file_types"], min_year=pl.get("min_year"))
    candidates = privacy_pl.one_per_entity_year(rows)
    split_of_entity = {e: assign_split(e, seed=seed, fractions=fr, forced=forced)
                       for e in {r.entity_id for r in candidates}}
    rng = random.Random(seed)
    sizes = split_sizes(cfg)
    sampled = []
    for split, n in sizes.items():
        pool = [r for r in candidates if split_of_entity[r.entity_id] == split]
        sampled += stratified_sample(pool, n, lambda r: (r.year, r.category), rng)
    docs = list(privacy_pl.iter_documents(con, sampled, genre=genre, **seg_kw)) + opp_docs
    split_of = {d.document.id: split_of_entity.get(d.document.entity_id, forced.get(d.document.entity_id, "dev"))
                for d in docs}
    docs = [d for d in docs if d.sections or d.document.source == "opp115"]
    docs, dedup_report = dedup_sample(docs, split_of, cfg["dedup"], protected={d.document.id for d in opp_docs})

    # revision pairs: every snapshot of every test entity (plan §2.6b, §2.7)
    test_entities = {d.document.entity_id for d in docs if split_of[d.document.id] == "test"}
    chains: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.entity_id in test_entities:
            chains[r.entity_id].append(r)
    rv = cfg["revisions"]
    pairs: list[RevisionPair] = []
    version_docs: dict[str, ParsedDocument] = {}
    for entity in sorted(chains):
        chain = list(privacy_pl.iter_documents(con, sorted(chains[entity], key=lambda r: r.date or ""),
                                               genre=genre, **seg_kw))
        chain = [c for c in chain if c.sections]
        for p in chain_pairs(chain, same_threshold=rv["same_threshold"], min_change=rv["min_change"],
                             max_change=rv["max_change"], num_perm=cfg["dedup"]["num_perm"],
                             shingle_size=cfg["dedup"]["shingle_size"]):
            pairs.append(p)
            for c in chain:
                if c.document.id in (p.old_doc_id, p.new_doc_id):
                    version_docs[c.document.id] = c

    labels = list(opp115.iter_labels(_p(cfg, src["opp115"]["dir"]), threshold=src["opp115"]["consolidation_threshold"]))
    return _finish(cfg, genre, docs, split_of, dedup_report, pairs, list(version_docs.values()), t0,
                   extra_files={"opp115_labels.jsonl": labels},
                   previews={"privacy_pl": [d for d in docs if d.document.source == "privacy_pl"][:3],
                             "opp115": opp_docs[:3]},
                   sample_rows=_sample_table(genre, candidates, docs, split_of, lambda r: r.year))


# ------------------------------------------------------------------------------------ genre B
def build_papers(cfg: dict[str, Any], genre: str = "papers") -> dict[str, Any]:
    from data.loaders import unarxive

    src = cfg["data_sources"][genre]["unarxive"]
    seed, fr = cfg["splits"]["seed"], {k: cfg["splits"][k] for k in ("train", "dev", "test")}
    t0 = time.perf_counter()
    extract_path = _p(cfg, src["extract"])
    extract_report = {}
    if not extract_path.exists():
        extract_report = unarxive.extract(_p(cfg, src["archive"]), extract_path, categories=set(src["categories"]),
                                          min_date=src["min_date"])
    min_paragraphs = src.get("min_paragraphs", 5)
    candidates = [m for m in unarxive.iter_metadata(extract_path, cutoff=src["cutoff"])
                  if m["date"] and m["n_paragraphs"] >= min_paragraphs]
    split_of_entity = {m["paper_id"]: assign_split(m["paper_id"], seed=seed, fractions=fr) for m in candidates}
    rng = random.Random(seed)
    chosen = []
    for split, n in split_sizes(cfg).items():
        pool = [m for m in candidates if split_of_entity[m["paper_id"]] == split]
        chosen += stratified_sample(pool, n, lambda m: (m["date"][:4], m["category"]), rng)
    max_chars = cfg["segment"]["max_section_chars"]
    docs = [d for d in unarxive.iter_documents(extract_path, genre=genre, ids={m["paper_id"] for m in chosen},
                                               cutoff=src["cutoff"], max_section_chars=max_chars) if d.sections]
    split_of = {d.document.id: split_of_entity[d.document.entity_id] for d in docs}
    docs, dedup_report = dedup_sample(docs, split_of, cfg["dedup"])
    dedup_report.update(extract_report)
    # revision candidates (v1 -> last version up to the cutoff) for multi-version test papers; full texts are fetched
    # by fetch_paper_revisions, which is network-bound and resumable.
    candidates_rv = [{"entity_id": d.document.entity_id, "category": d.document.category,
                      "versions": d.document.meta["versions"]}
                     for d in docs if split_of[d.document.id] == "test" and len(d.document.meta["versions"]) >= 2]
    out_dir = _p(cfg, "data_processed") / genre
    write_jsonl(out_dir / "revision_candidates.jsonl", candidates_rv)
    pairs, version_docs = fetch_paper_revisions(cfg, genre, candidates_rv)
    return _finish(cfg, genre, docs, split_of, dedup_report, pairs, version_docs, t0,
                   previews={"unarxive": docs[:3], "arxiv_eprint": version_docs[:3]},
                   sample_rows=_sample_table(genre, candidates, docs, split_of, lambda m: m["date"][:4]),
                   extra_stats={"revision_candidates": len(candidates_rv)})


def fetch_paper_revisions(cfg: dict[str, Any], genre: str, candidates: list[dict[str, Any]]
                          ) -> tuple[list[RevisionPair], list[ParsedDocument]]:
    """Fetch and parse (v1, last version) e-prints for up to arxiv_eprints.max_pairs candidates; cached on disk."""
    from data.loaders.arxiv_versions import eprint_to_document, fetch_eprint

    ecfg = cfg["data_sources"][genre]["arxiv_eprints"]
    max_pairs = ecfg.get("max_pairs")
    todo = candidates if max_pairs is None else candidates[: int(max_pairs)]
    rv = cfg["revisions"]
    pairs, version_docs = [], []
    for c in todo:
        (v_old, d_old), (v_new, d_new) = c["versions"][0], c["versions"][-1]
        try:
            docs = [eprint_to_document(fetch_eprint(c["entity_id"], v, _p(cfg, ecfg["cache"]),
                                                    min_interval=ecfg["min_interval_s"]),
                                       arxiv_id=c["entity_id"], version=v, date=d, category=c["category"], genre=genre,
                                       max_section_chars=cfg["segment"]["max_section_chars"])
                    for v, d in ((v_old, d_old), (v_new, d_new))]
        except Exception as e:  # missing source (PDF-only submission), network error: logged, skipped
            print(f"[revisions] {c['entity_id']}: {type(e).__name__}: {e}")
            continue
        if not all(d.sections for d in docs):
            continue
        for p in chain_pairs(docs, same_threshold=rv["same_threshold"], min_change=rv["min_change"],
                             max_change=rv["max_change"], num_perm=cfg["dedup"]["num_perm"],
                             shingle_size=cfg["dedup"]["shingle_size"]):
            pairs.append(p)
            version_docs += docs
    return pairs, version_docs


# ------------------------------------------------------------------------------------ shared tail
def _sample_table(genre: str, candidates: list, docs: list[ParsedDocument], split_of: dict[str, str],
                  year_of: Callable[[Any], Any]) -> list[dict[str, Any]]:
    pool = Counter(str(year_of(c)) for c in candidates)
    got: dict[str, Counter] = defaultdict(Counter)
    for d in docs:
        got[split_of[d.document.id]][(d.document.date or "")[:4]] += 1
    return [{"genre": genre, "year": y, "candidates": pool[y], **{f"sampled_{s}": got[s][y] for s in ("train", "dev", "test")}}
            for y in sorted(set(pool) | {y for c in got.values() for y in c})]


def _finish(cfg, genre, docs, split_of, dedup_report, pairs, version_docs, t0, *, previews,
            sample_rows, extra_files=None, extra_stats=None) -> dict[str, Any]:
    out_dir = _p(cfg, "data_processed") / genre
    min_sent = cfg["segment"]["min_sentence_chars"]
    written = write_corpus(out_dir, docs, min_sent)
    rev = write_corpus(out_dir / "revisions", version_docs, min_sent) if version_docs else {"documents": 0}
    write_jsonl(out_dir / "revisions.jsonl", pairs)
    for name, records in (extra_files or {}).items():
        write_jsonl(out_dir / name, records)
    summary = write_manifest(_p(cfg, "splits") / f"{genre}.json", genre=genre, seed=cfg["splits"]["seed"],
                             fractions={k: cfg["splits"][k] for k in ("train", "dev", "test")},
                             doc_split={d.document.id: split_of[d.document.id] for d in docs},
                             entity_of={d.document.id: d.document.entity_id for d in docs})
    tables = _p(cfg, "tables")
    write_table(tables / "corpus_stats.csv", corpus_stats_rows(genre, docs, split_of, written["_per_doc"]))
    write_table(tables / "dedup.csv", [{"genre": genre, **dedup_report}])
    write_table(tables / "corpus_sample.csv", sample_rows)
    by_year = Counter((p.old_date or "")[:4] for p in pairs)
    write_table(tables / "revisions.csv", [{"genre": genre, "year": y, "pairs": n,
                                            "median_change": round(st.median(p.change_ratio for p in pairs
                                                                             if (p.old_date or "")[:4] == y), 3)}
                                           for y, n in sorted(by_year.items())] or [{"genre": genre, "year": "", "pairs": 0}])
    for source, ds in previews.items():
        if ds:
            preview_html(ds, _p(cfg, "figures") / f"loader_check_{source}.html", f"Loader check: {source}")
    return {f"{genre}/{k}": v for k, v in {
        "documents": written["documents"], "sections": written["sections"], "sentences": written["sentences"],
        **{f"split_{s}": n for s, n in summary["documents"].items()},
        "entities_test": summary["entities"]["test"], "test_sha256": summary["test_sha256"],
        "revision_pairs": len(pairs), "revision_documents": rev["documents"],
        **{f"dedup_{k}": v for k, v in dedup_report.items()}, **(extra_stats or {}),
        "seconds": round(time.perf_counter() - t0, 1),
    }.items()}


BUILDERS = {"privacy": build_privacy, "papers": build_papers}


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    genres = [cfg["pack"]] if cfg.get("pack") else cfg["genres"]
    metrics: dict[str, Any] = {}
    for genre in genres:
        metrics.update(BUILDERS[genre](cfg, genre))
        (run_dir / f"build_{genre}.json").write_text(json.dumps({k: v for k, v in metrics.items()
                                                                if k.startswith(genre)}, indent=1))
    return metrics
