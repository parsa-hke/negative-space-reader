"""Schema induction run (plan Phase 3). Stages, chosen by `induce.stage`:

  eye_test   (3.1) the extraction prompt on a fixed set of 20 dev sections per genre; writes eye_test_<genre>.jsonl,
             a review page eye_test.html (spans highlighted in the section text; side by side with an earlier run
             when induce.eye_test.compare_with is set) and automatic checks as metrics
  extract    (3.2) every section of a stratified sample of dev documents -> claims_<genre>.jsonl
  cluster    (3.3) embed the claim texts (vectors kept in an EmbeddingStore under paths.embeddings, reused by
             reruns), reduce dimensions, sweep HDBSCAN settings (or use schema.hdbscan when it is set) and keep the
             setting with 60-120 clusters and the least noise; writes clusters_<genre>.json, centroids, assignments
  name       (3.3) the teacher names each cluster of an earlier cluster run (induce.name.from_run), defines it
             and picks representative claims; writes schema_draft_<genre>.yaml (and packs/<genre>/schema_draft.yaml
             for a full run), with flags for the human pass (mixed clusters, example-label leakage, duplicate names)
  review     (3.4) the workbook for the human pass (packs/schema_review.py), from a cluster run, a name run,
             Claude's proposals and the pack's seeds; the reviewer's copy is reports/schema_review/
  candidate  (3.4) the reviewer's workbook -> packs/<genre>/schema_candidate.yaml, with the gate's field checks
  coverage   (3.5) the teacher maps what held-out dev sections assert to the candidate's types or "none"; every
             "none" gets a second look among the most similar types
  freeze     (3.6) candidate -> packs/<genre>/schema.yaml version 1 (sign-off, passing coverage; never overwrites)
  relink     after a rebuild that split long sections (2026-10-15): the extract-stage claims -> claims_<genre>.jsonl
             with section_idx pointing into the rebuilt sections (by the claims' verbatim spans); ids unchanged

The section selection is stored as ids in prompts/eval/ (tracked) so every model sees the same 20 sections; the
text is read from data/processed/ and its sha1 is checked, so a changed corpus cannot silently change the test.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from data.schema import Section, read_jsonl
from data.splits import load_manifest
from packs.schema_induction import (SectionExtraction, SectionRef, _sha1, extract, extraction_checks, read_refs,
                                    select_eye_test, summarise_checks, write_refs)

REPO = Path(__file__).resolve().parent.parent


def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def _genres(cfg: dict[str, Any]) -> list[str]:
    return [cfg["pack"]] if cfg.get("pack") else list(cfg["genres"])


def eye_test_sections(cfg: dict[str, Any], genre: str) -> list[tuple[SectionRef, Section]]:
    """The fixed eye-test sections for a genre, creating (and storing) the selection on first use."""
    ecfg = cfg["induce"]["eye_test"]
    sel_path = _path(ecfg["selection"])
    refs = [r for r in read_refs(sel_path) if r.genre == genre] if sel_path.exists() else []
    sections_path = _path(cfg["paths"]["data_processed"]) / genre / "sections.jsonl"
    if not refs:
        split = load_manifest(_path(cfg["paths"]["splits"]) / f"{genre}.json")
        dev = {d for d, s in split.items() if s == "dev"}
        refs = select_eye_test(read_jsonl(sections_path, Section), dev, genre, n=ecfg["sections_per_genre"],
                               seed=cfg["splits"]["seed"], min_chars=ecfg["min_chars"], max_chars=ecfg["max_chars"])
        others = [r for r in read_refs(sel_path) if r.genre != genre] if sel_path.exists() else []
        write_refs(sel_path, others + refs)
    wanted = {(r.doc_id, r.section_idx): r for r in refs}
    found = {}
    for s in read_jsonl(sections_path, Section):
        key = (s.doc_id, s.idx)
        if key in wanted:
            found[key] = s
    out = []
    for key, ref in wanted.items():
        s = found.get(key)
        if s is None or _sha1(s.text) != ref.text_sha1:
            raise RuntimeError(f"eye-test section {key} is missing or changed in {sections_path}; the corpus differs "
                               f"from the one the selection was made on")
        out.append((ref, s))
    return out


def _highlight(text: str, spans: list[str]) -> str:
    """Escape text and wrap each verbatim span occurrence in <mark>."""
    marks = []
    for sp in spans:
        i = text.find(sp.strip()) if sp.strip() else -1
        if i >= 0:
            marks.append((i, i + len(sp.strip())))
    out, pos = [], 0
    for start, end in sorted(marks):
        if start < pos:
            continue
        out += [html.escape(text[pos:start]), "<mark>", html.escape(text[start:end]), "</mark>"]
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def _table(rows: list[dict[str, Any]], grounded: list[bool], failed: bool = False) -> str:
    if failed:
        return "<p class='empty'>(FAILED: no valid output after retries)</p>"
    if not rows:
        return "<p class='empty'>(no assertions)</p>"
    body = "".join(
        f"<tr><td>{html.escape(r['claim'])}</td><td><code>{html.escape(r['type_guess'])}</code></td>"
        f"<td class='{'ok' if g else 'bad'}'>{html.escape(r['span'])}</td></tr>" for r, g in zip(rows, grounded))
    return f"<table><tr><th>claim</th><th>type_guess</th><th>span (red = not a verbatim quote)</th></tr>{body}</table>"


def write_review_page(path: Path, results: dict[str, list[dict[str, Any]]], model: str, prompt_tag: str,
                      previous: dict[str, list[dict[str, Any]]] | None = None, previous_model: str | None = None) -> None:
    css = ("body{font:14px/1.45 system-ui;margin:24px}h2{margin-top:40px}.sec{border:1px solid #999;border-radius:6px;"
           "padding:10px;margin:14px 0}.text{background:#f6f6f2;padding:8px;max-height:320px;overflow:auto}"
           "mark{background:#ffe27a}table{border-collapse:collapse;width:100%;margin-top:6px}td,th{border:1px solid #ccc;"
           "padding:4px 6px;vertical-align:top;font-size:13px}.bad{color:#b00020}.ok{color:#14532d}.cols{display:grid;"
           "grid-template-columns:1fr 1fr;gap:12px}.meta{color:#555;font-size:12px}.empty{color:#b00020}")
    parts = [f"<!doctype html><meta charset='utf-8'><title>Eye test: induce_extract</title><style>{css}</style>",
             f"<h1>Eye test: induce_extract {html.escape(prompt_tag)}</h1>",
             f"<p class='meta'>model: <b>{html.escape(model)}</b>"
             + (f" · compared with: <b>{html.escape(previous_model or '')}</b>" if previous else "") + "</p>",
             "<p>Check per section: is every claim stated in the text (no inference)? Is anything stated missing? "
             "Are claims general (no names) and one point each? Are spans verbatim (highlighted)?</p>"]
    for genre, rows in results.items():
        parts.append(f"<h2>{html.escape(genre)}</h2>")
        prev = {(r["doc_id"], r["section_idx"]): r for r in (previous or {}).get(genre, [])}
        for r in rows:
            spans = [a["span"] for a in r["assertions"]]
            this = _table(r["assertions"], r["checks"]["grounded"], r.get("failed", False))
            p = prev.get((r["doc_id"], r["section_idx"]))
            right = (f"<div><div class='meta'>{html.escape(previous_model or 'previous')}</div>"
                     f"{_table(p['assertions'], p['checks']['grounded'], p.get('failed', False))}</div>") if p else ""
            parts.append(
                f"<div class='sec'><div class='meta'>{html.escape(r['doc_id'])} · section {r['section_idx']} · "
                f"{r['position']} third · {len(r['text'])} chars</div><b>{html.escape(r['heading'] or '(no heading)')}</b>"
                f"<div class='text'>{_highlight(r['text'], spans)}</div>"
                f"<div class='cols' style='grid-template-columns:{'1fr 1fr' if p else '1fr'}'>"
                f"<div><div class='meta'>{html.escape(model)}</div>{this}</div>{right}</div></div>")
    path.write_text("".join(parts), encoding="utf-8")


def run_eye_test(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from experiments.run import resolve_device
    from models.teacher import PROMPTS, load_teacher

    icfg = cfg["induce"]
    tcfg = cfg["models"]["teacher"]
    teacher = load_teacher(tcfg, device=resolve_device(cfg["run"].get("device", "auto")), seed=cfg["run"]["seed"])
    prompt_tag = PROMPTS.get(icfg["prompt"]).tag
    results: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, Any] = {"model": tcfg["id"], "backend": tcfg["backend"], "prompt": f"{icfg['prompt']} {prompt_tag}"}
    for genre in _genres(cfg):
        pairs = eye_test_sections(cfg, genre)
        sections = [s for _, s in pairs]
        extractions = extract(teacher, sections, cfg["genre_info"][genre], max_assertions=icfg["max_assertions"],
                              max_tokens=icfg["max_tokens"], batch_size=icfg["batch_size"], skip_failures=True)
        rows = []
        for (ref, s), ex in zip(pairs, extractions):
            failed = ex is None
            ex = ex or SectionExtraction(assertions=[])
            rows.append({"doc_id": ref.doc_id, "section_idx": ref.section_idx, "position": ref.position,
                         "heading": s.heading, "text": s.text, "assertions": [a.model_dump() for a in ex.assertions],
                         "failed": failed, "checks": extraction_checks(s.text, ex)})
        results[genre] = rows
        with open(run_dir / f"eye_test_{genre}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        metrics.update({f"{genre}/{k}": v for k, v in summarise_checks([r["checks"] for r in rows]).items()})
        metrics[f"{genre}/failed_sections"] = sum(r["failed"] for r in rows)
    previous, previous_model = None, None
    if icfg["eye_test"].get("compare_with"):
        prev_dir = _path(icfg["eye_test"]["compare_with"])
        previous = {g: [json.loads(l) for l in open(prev_dir / f"eye_test_{g}.jsonl", encoding="utf-8")]
                    for g in results if (prev_dir / f"eye_test_{g}.jsonl").exists()}
        prev_metrics = json.loads((prev_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"] if (prev_dir / "metrics.json").exists() else {}
        previous_model = prev_metrics.get("model", prev_dir.name)
        metrics["compared_with"] = str(prev_dir)
    write_review_page(run_dir / "eye_test.html", results, tcfg["id"], prompt_tag, previous, previous_model)
    return metrics


def sample_documents(cfg: dict[str, Any], genre: str) -> list[str]:
    """Stratified (year x category) sample of dev documents for extraction (plan §3.2). Deterministic."""
    import random

    from data.build import stratified_sample
    from data.schema import Document

    xcfg = cfg["induce"]["extract"]
    split = load_manifest(_path(cfg["paths"]["splits"]) / f"{genre}.json")
    docs = [d for d in read_jsonl(_path(cfg["paths"]["data_processed"]) / genre / "documents.jsonl", Document)
            if split.get(d.id) == xcfg["split"]]
    rng = random.Random(cfg["splits"]["seed"])
    chosen = stratified_sample(sorted(docs, key=lambda d: d.id), xcfg["documents_per_genre"],
                               lambda d: ((d.date or "")[:4], d.category), rng)
    return sorted(d.id for d in chosen)


def run_extract(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Plan §3.2: every section of a stratified sample of dev documents -> claims_<genre>.jsonl (one row per claim,
    with its source section, prompt version and teacher build). Streams to disk per batch; a rerun after an
    interruption replays finished batches from the teacher's response cache."""
    import time

    from experiments.run import resolve_device
    from models.teacher import PROMPTS, TEACHER_STATS, load_teacher
    from packs.schema_induction import _norm

    icfg, xcfg = cfg["induce"], cfg["induce"]["extract"]
    tcfg = cfg["models"]["teacher"]
    teacher = load_teacher(tcfg, device=resolve_device(cfg["run"].get("device", "auto")), seed=cfg["run"]["seed"])
    prompt_tag = PROMPTS.get(icfg["prompt"]).tag
    route = teacher._route()
    metrics: dict[str, Any] = {"model": tcfg["id"], "backend": tcfg["backend"], "route": route,
                               "prompt": f"{icfg['prompt']} {prompt_tag}"}
    for genre in _genres(cfg):
        t0 = time.perf_counter()
        doc_ids = set(sample_documents(cfg, genre))
        (run_dir / f"sample_{genre}.json").write_text(json.dumps(sorted(doc_ids), indent=0))
        sections = [s for s in read_jsonl(_path(cfg["paths"]["data_processed"]) / genre / "sections.jsonl", Section)
                    if s.doc_id in doc_ids and len(s.text) >= xcfg["min_section_chars"]]
        # documents in a fixed pseudo-random order (sections of a document stay together): if a session must stop
        # early, the finished part is a random subset of the sample, not the alphabetically first documents
        sections.sort(key=lambda s: (_sha1(f"{cfg['splits']['seed']}:{s.doc_id}"), s.idx))
        claims_path, sections_path = run_dir / f"claims_{genre}.jsonl", run_dir / f"sections_{genre}.jsonl"
        counts = {"claims": 0, "grounded": 0, "failed": 0, "empty": 0}
        with open(claims_path, "w", encoding="utf-8") as cf, open(sections_path, "w", encoding="utf-8") as sf:
            def write_batch(start: int, results: list) -> None:
                for s, ex in zip(sections[start:start + len(results)], results):
                    n = 0 if ex is None else len(ex.assertions)
                    counts["failed"] += ex is None
                    counts["empty"] += ex is not None and n == 0
                    sf.write(json.dumps({"doc_id": s.doc_id, "section_idx": s.idx, "heading": s.heading,
                                         "chars": len(s.text), "n_claims": n, "failed": ex is None}) + "\n")
                    text = _norm(s.text)
                    for j, a in enumerate(ex.assertions if ex is not None else []):
                        grounded = bool(a.span.strip()) and _norm(a.span) in text
                        counts["claims"] += 1
                        counts["grounded"] += grounded
                        cf.write(json.dumps({"claim_id": f"{s.doc_id}#{s.idx}#{j}", "genre": genre, "doc_id": s.doc_id,
                                             "section_idx": s.idx, "heading": s.heading, "claim": a.claim,
                                             "type_guess": a.type_guess, "span": a.span, "span_grounded": grounded,
                                             "prompt": f"{icfg['prompt']} {prompt_tag}", "teacher": route},
                                            ensure_ascii=False) + "\n")
                cf.flush()
                sf.flush()
                done = start + len(results)
                elapsed = time.perf_counter() - t0
                eta = elapsed / max(done, 1) * (len(sections) - done)
                print(f"[extract {genre}] {done}/{len(sections)} sections ({done / len(sections):.0%}), "
                      f"{counts['claims']} claims, {counts['failed']} failed, {elapsed / 60:.1f} min elapsed, "
                      f"about {eta / 60:.0f} min left for {genre}", flush=True)

            extract(teacher, sections, cfg["genre_info"][genre], max_assertions=icfg["max_assertions"],
                    max_tokens=icfg["max_tokens"], batch_size=icfg["batch_size"], skip_failures=True,
                    on_batch=write_batch)
        n = len(sections)
        metrics.update({f"{genre}/documents": len(doc_ids), f"{genre}/sections": n,
                        f"{genre}/claims": counts["claims"],
                        f"{genre}/claims_per_section": round(counts["claims"] / max(1, n), 2),
                        f"{genre}/span_grounding_rate": round(counts["grounded"] / max(1, counts["claims"]), 3),
                        f"{genre}/empty_sections": counts["empty"], f"{genre}/failed_sections": counts["failed"],
                        f"{genre}/seconds": round(time.perf_counter() - t0, 1)})
    metrics["teacher_tokens"] = {k: TEACHER_STATS.summary()[k] for k in ("prompt_tokens", "completion_tokens", "cache_hits")}
    return metrics


def embed_claims(cfg: dict[str, Any], genre: str, ids: list[str], texts: list[str]) -> tuple[Any, int, str | None]:
    """Unit-norm claim vectors from models.embedder, under the instruction prompt induce.cluster.embed_prompt (or
    none). Vectors already in the claims EmbeddingStore (same text sha1; one store per prompt version) are reused;
    only new texts are embedded, and the store is rewritten. Vectors are always the stored float16 values, so a
    fresh run and a rerun cluster identical inputs. Returns (matrix, number embedded now, prompt tag or None)."""
    import numpy as np

    import engine.embed as E
    from experiments.run import resolve_device
    from models.teacher import PROMPTS

    ecfg = cfg["models"]["embedder"]
    prompt_id = cfg["induce"]["cluster"].get("embed_prompt")
    instruction, store_name, tag = None, "claims", None
    if prompt_id:
        p = PROMPTS.get(prompt_id)
        instruction = p.render(document_kind=cfg["genre_info"][genre]["document_kind"]).rstrip("\n") + " "
        store_name, tag = f"claims__{p.id}_v{p.version}_{p.hash}", f"{p.id} {p.tag}"
    store_path = _path(cfg["paths"]["embeddings"]) / E.embedding_slug(ecfg) / genre / store_name
    found = E.EmbeddingStore.load(store_path).lookup(texts) if (store_path / "rows.jsonl").exists() else [None] * len(texts)
    missing = [i for i, v in enumerate(found) if v is None]
    if missing:
        embedder = E.Embedder(ecfg["id"], devices=[resolve_device(cfg["run"].get("device", "auto"))],
                              dtype=ecfg.get("dtype", "float16"), max_length=ecfg.get("max_length", 1024),
                              batch_size=cfg["induce"]["cluster"].get("embed_batch_size") or ecfg.get("batch_size", 32),
                              normalize=True, truncate_dim=ecfg.get("truncate_dim"), revision=ecfg.get("revision"))
        fresh = embedder.encode([texts[i] for i in missing], show_progress=True, prompt=instruction)
        for i, v in zip(missing, fresh):
            found[i] = v
        E.EmbeddingStore.save(store_path, ids, texts, np.stack(found),
                              {"model": ecfg["id"], "genre": genre, "unit": "claim", "normalized": True,
                               "prompt": tag, "instruction": instruction})
    X = np.stack(found).astype(np.float16).astype(np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12), len(missing), tag


def _cluster_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    h = cfg["schema"]["hdbscan"]
    if h.get("min_cluster_size"):
        return [{"min_cluster_size": h["min_cluster_size"], "min_samples": h.get("min_samples"),
                 "selection": h.get("cluster_selection_method", "eom")}]
    g = cfg["induce"]["cluster"]["grid"]
    return [{"min_cluster_size": m, "min_samples": s, "selection": sel}
            for sel in g["selection"] for m in g["min_cluster_size"] for s in g["min_samples"]]


def cluster_claims_input(cfg: dict[str, Any], genre: str) -> tuple[list[dict[str, Any]], int]:
    """The claims a cluster run works on: the extract-stage file minus excluded documents, capped at
    schema.sample_documents documents (a fixed pseudo-random choice; the cap only bites in smoke mode).
    Returns (claims, claims excluded by document)."""
    from packs.schema_induction import load_claims

    ccfg = cfg["induce"]["cluster"]
    path = _path(ccfg["claims"][genre])
    claims = load_claims(path, exclude_docs=ccfg.get("exclude_docs") or [])
    n_excluded = sum(1 for _ in open(path, encoding="utf-8")) - len(claims)
    docs = sorted({c["doc_id"] for c in claims}, key=lambda d: _sha1(f"{cfg['splits']['seed']}:{d}"))
    keep = set(docs[: cfg["schema"]["sample_documents"]])
    return [c for c in claims if c["doc_id"] in keep], n_excluded


def run_cluster(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Plan §3.3, first half: claims -> embeddings -> reduced vectors -> HDBSCAN (swept or fixed) -> summaries."""
    import numpy as np

    from packs.schema_induction import center_by_document, reduce_dims, summarise_clusters, tune_clusters

    ccfg, scfg = cfg["induce"]["cluster"], cfg["schema"]
    metrics: dict[str, Any] = {"embedder": cfg["models"]["embedder"]["id"], "reduce": ccfg["reduce"]}
    for genre in _genres(cfg):
        claims, n_excluded = cluster_claims_input(cfg, genre)
        X, embedded_now, embed_tag = embed_claims(cfg, genre, [c["claim_id"] for c in claims], [c["claim"] for c in claims])
        metrics["embed_prompt"] = embed_tag
        # the clustering space: claims minus their document's shared (topic) direction; summaries use it too, so
        # "most typical member" means most typical of the move, not of the topic
        X = center_by_document(X, [c["doc_id"] for c in claims], alpha=ccfg.get("center_by_document", 0.0))
        r = ccfg["reduce"]
        Z = reduce_dims(X, method=r["method"], dims=r["dims"], seed=cfg["run"]["seed"],
                        n_neighbors=r.get("n_neighbors", 15), min_dist=r.get("min_dist", 0.0))
        labels, sweep, best = tune_clusters(Z, _cluster_grid(cfg), target=scfg["target_clusters"],
                                            max_cluster_share=ccfg["max_cluster_share"])
        clusters = summarise_clusters(claims, labels, X)
        np.save(run_dir / f"centroids_{genre}.npy", np.asarray([c.pop("centroid") for c in clusters], dtype=np.float32))
        (run_dir / f"clusters_{genre}.json").write_text(json.dumps(clusters, ensure_ascii=False), encoding="utf-8")
        cluster_of = {m: c["id"] for c in clusters for m in c["members"]}
        with open(run_dir / f"assignments_{genre}.jsonl", "w", encoding="utf-8") as f:
            for c in claims:
                f.write(json.dumps({"claim_id": c["claim_id"], "cluster": cluster_of.get(c["claim_id"], "noise")}) + "\n")
        sizes = sorted(c["size"] for c in clusters)
        docs = {c["doc_id"] for c in claims}
        metrics.update({
            f"{genre}/claims": len(claims), f"{genre}/excluded_claims": n_excluded,
            f"{genre}/documents": len(docs), f"{genre}/embedded_now": embedded_now,
            f"{genre}/sweep": sweep, f"{genre}/chosen": sweep[best],
            f"{genre}/n_clusters": len(clusters), f"{genre}/noise_rate": sweep[best]["noise_rate"],
            f"{genre}/size_min": sizes[0] if sizes else 0, f"{genre}/size_median": sizes[len(sizes) // 2] if sizes else 0,
            f"{genre}/size_max": sizes[-1] if sizes else 0,
            f"{genre}/clusters_under_5pct_documents": sum(c["documents"] < 0.05 * len(docs) for c in clusters)})
    return metrics


def run_name(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Plan §3.3, second half: the teacher names the clusters of an earlier cluster run -> schema draft."""
    import datetime as dt

    import yaml

    from experiments.run import resolve_device
    from models.teacher import PROMPTS, load_teacher
    from packs.schema_induction import (NAME_PROMPT, draft_types, name_clusters, naming_sample, prompt_example_labels,
                                        render_name_prompt)

    ncfg = cfg["induce"]["name"]
    src = _path(ncfg["from_run"])
    src_cfg = yaml.safe_load((src / "config.yaml").read_text(encoding="utf-8"))
    src_metrics = json.loads((src / "metrics.json").read_text(encoding="utf-8")) if (src / "metrics.json").exists() else {}
    tcfg = cfg["models"]["teacher"]
    teacher = load_teacher(tcfg, device=resolve_device(cfg["run"].get("device", "auto")), seed=cfg["run"]["seed"])
    example_labels = prompt_example_labels(cfg["induce"]["prompt"])
    metrics: dict[str, Any] = {"model": tcfg["id"], "route": teacher._route(), "from_run": str(src),
                               "example_labels": sorted(example_labels)}
    for genre in _genres(cfg):
        all_clusters = json.loads((src / f"clusters_{genre}.json").read_text(encoding="utf-8"))
        clusters = all_clusters[: ncfg.get("limit")]
        claim_rows, _ = cluster_claims_input(src_cfg, genre)
        claims = {c["claim_id"]: c for c in claim_rows}
        n_docs = len({c["doc_id"] for c in claim_rows})
        shown = [naming_sample(c["members"], n=ncfg["claims_per_cluster"], seed=cfg["run"]["seed"]) for c in clusters]
        prompts = [render_name_prompt([claims[i]["claim"] for i in ids], cfg["genre_info"][genre],
                                      cluster_size=c["size"], n_documents=c["documents"],
                                      type_hints=[tuple(t) for t in c["type_guess_top"]])
                   for c, ids in zip(clusters, shown)]
        names = name_clusters(teacher, prompts, max_tokens=ncfg["max_tokens"], batch_size=cfg["induce"]["batch_size"])
        types = draft_types(clusters, shown, names, claims, example_labels=example_labels, n_documents=n_docs,
                            n_representative=cfg["schema"]["representative_claims"])
        with open(run_dir / f"names_{genre}.jsonl", "w", encoding="utf-8") as f:
            for c, ids, nm in zip(clusters, shown, names):
                f.write(json.dumps({"cluster": c["id"], "shown": ids, "output": nm.model_dump() if nm else None},
                                   ensure_ascii=False) + "\n")
        chosen = src_metrics.get("metrics", {}).get(f"{genre}/chosen", {})
        draft = {
            "genre": genre, "status": "draft", "version": None,
            "note": "Plan §3.3 output. Not a schema: the human pass (§3.4) merges, splits and prunes these to 40-60 "
                    "types and adds examples, near-misses, severity weights, section hints and label sources.",
            "induced": {
                "date": dt.date.today().isoformat(), "claims_file": src_cfg["induce"]["cluster"]["claims"][genre],
                "excluded_documents": src_cfg["induce"]["cluster"].get("exclude_docs") or [],
                "claims": len(claim_rows), "documents": n_docs,
                "extraction_teacher": claim_rows[0].get("teacher") if claim_rows else None,
                "extraction_prompt": claim_rows[0].get("prompt") if claim_rows else None,
                "embedder": src_cfg["models"]["embedder"]["id"], "reduce": src_cfg["induce"]["cluster"]["reduce"],
                "hdbscan": {k: chosen.get(k) for k in ("min_cluster_size", "min_samples", "selection")},
                "clusters": len(all_clusters), "named": len(clusters), "noise_rate": chosen.get("noise_rate"),
                "cluster_run": src.name, "name_run": run_dir.name,
                "naming_teacher": teacher._route(), "naming_prompt": f"{NAME_PROMPT} {PROMPTS.get(NAME_PROMPT).tag}"},
            "types": types}
        text = yaml.safe_dump(draft, sort_keys=False, allow_unicode=True, width=120)
        (run_dir / f"schema_draft_{genre}.yaml").write_text(text, encoding="utf-8")
        if ncfg.get("write_pack") and not ncfg.get("limit"):
            out = _path(cfg["paths"]["packs"]) / genre / "schema_draft.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            metrics[f"{genre}/written"] = str(out)
        metrics.update({
            f"{genre}/named": sum(n is not None for n in names), f"{genre}/failed": sum(n is None for n in names),
            f"{genre}/mixed": sum("mixed" in t["flags"] for t in types),
            f"{genre}/duplicate_names": len({t["name"] for t in types if "duplicate_name" in t["flags"]}),
            f"{genre}/example_label_names": [t["name"] for t in types if "example_label_name" in t["flags"]]})
    return metrics


def run_review(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Plan §3.4: the review workbook for the human pass, from a cluster run, a name run, Claude's proposals
    (induce.review.proposals, '{genre}' filled in) and the pack's seeds. Written to the run directory and to
    induce.review.out_dir/schema_review_<genre>.xlsx (the copy the reviewer edits)."""
    import random
    import shutil

    import numpy as np
    import yaml

    from packs.schema_induction import center_by_document
    from packs.schema_review import build_workbook, complete_proposals

    rcfg = cfg["induce"]["review"]
    crun = _path(rcfg["cluster_run"])
    ccfg = yaml.safe_load((crun / "config.yaml").read_text(encoding="utf-8"))
    metrics: dict[str, Any] = {"cluster_run": str(crun)}
    for genre in _genres(cfg):
        nrun = _path(rcfg["name_run"].format(genre=genre))  # one naming run per genre is allowed
        metrics[f"{genre}/name_run"] = str(nrun)
        claim_rows, _ = cluster_claims_input(ccfg, genre)
        claims = {c["claim_id"]: c for c in claim_rows}
        clusters = json.loads((crun / f"clusters_{genre}.json").read_text(encoding="utf-8"))
        centroids = np.load(crun / f"centroids_{genre}.npy")
        draft = yaml.safe_load((nrun / f"schema_draft_{genre}.yaml").read_text(encoding="utf-8"))
        names = {t["id"]: t for t in draft["types"]}
        prop_path = _path(rcfg["proposals"].format(genre=genre))
        proposals = yaml.safe_load(prop_path.read_text(encoding="utf-8")) if prop_path.exists() else {}
        proposals = complete_proposals(proposals, clusters, names, centroids)
        seeds = yaml.safe_load(_path(rcfg["seeds"].format(genre=genre)).read_text(encoding="utf-8"))
        # noise claims: a seeded sample, each with its nearest cluster in the clustering space
        assigned = {json.loads(l)["claim_id"]: json.loads(l)["cluster"]
                    for l in open(crun / f"assignments_{genre}.jsonl", encoding="utf-8")}
        noise_ids = sorted(i for i, c in assigned.items() if c == "noise")
        picked = random.Random(cfg["run"]["seed"]).sample(noise_ids, min(rcfg["noise_sample"], len(noise_ids)))
        noise = []
        if picked:
            X, _, _ = embed_claims(ccfg, genre, [c["claim_id"] for c in claim_rows], [c["claim"] for c in claim_rows])
            X = center_by_document(X, [c["doc_id"] for c in claim_rows],
                                   alpha=ccfg["induce"]["cluster"].get("center_by_document", 0.0))
            row_of = {c["claim_id"]: i for i, c in enumerate(claim_rows)}
            sims = X[[row_of[i] for i in picked]] @ (centroids / np.linalg.norm(centroids, axis=1, keepdims=True)).T
            for i, s in zip(picked, sims):
                j = int(np.argmax(s))
                noise.append({**claims[i], "nearest": clusters[j]["id"], "similarity": round(float(s[j]), 3)})
        out = run_dir / f"schema_review_{genre}.xlsx"
        build_workbook(out, genre=genre, clusters=clusters, names=names, centroids=centroids, claims=claims,
                       noise=noise, proposals=proposals, seeds=seeds,
                       n_documents=len({c["doc_id"] for c in claim_rows}),
                       require_label_source=cfg["presence"]["label_sources"].get(genre) != "teacher",
                       provenance={"cluster_run": crun.name, "name_run": nrun.name,
                                   "naming": draft["induced"].get("naming_teacher"),
                                   "proposals": str(prop_path.name) if proposals else "none"})
        shared = _path(rcfg["out_dir"]) / out.name
        shared.parent.mkdir(parents=True, exist_ok=True)
        if shared.exists():  # never overwrite a reviewer's edits silently
            shared.rename(shared.with_name(f"{shared.stem}.{dt_stamp()}.xlsx"))
        shutil.copy(out, shared)
        metrics.update({f"{genre}/clusters": len(clusters), f"{genre}/proposed_types": len(proposals.get("types", [])),
                        f"{genre}/noise_sampled": len(noise), f"{genre}/workbook": str(shared)})
    return metrics


def run_candidate(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Plan §3.4 read-back: the reviewer's workbook -> packs/<genre>/schema_candidate.yaml, with the §3.6 field
    checks. Undecided or inconsistent rows stop the stage (ValueError listing them)."""
    import datetime as dt
    import hashlib

    import yaml

    from packs.schema_review import read_workbook, schema_from_review, validate_schema

    kcfg, scfg = cfg["induce"]["candidate"], cfg["schema"]
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        wb = _path(kcfg["workbook"].format(genre=genre))
        review_cfg = yaml.safe_load((_path(kcfg["cluster_run"]) / "config.yaml").read_text(encoding="utf-8"))
        claims = {c["claim_id"]: c for c in cluster_claims_input(review_cfg, genre)[0]}
        schema = schema_from_review(read_workbook(wb), claims, genre=genre)
        need_labels = cfg["presence"]["label_sources"].get(genre) != "teacher"
        problems = validate_schema(schema, min_types=scfg["min_types"], max_types=scfg["max_types"],
                                   require_label_source=need_labels)
        schema = {"genre": genre, "status": "candidate", "version": None, **{k: v for k, v in schema.items() if k != "genre"},
                  "provenance": {"date": dt.date.today().isoformat(), "workbook": str(wb.name),
                                 "workbook_sha1": hashlib.sha1(wb.read_bytes()).hexdigest()[:12],
                                 "cluster_run": _path(kcfg["cluster_run"]).name, "candidate_run": run_dir.name},
                  "gate_problems": problems}
        text = yaml.safe_dump(schema, sort_keys=False, allow_unicode=True, width=120)
        (run_dir / f"schema_candidate_{genre}.yaml").write_text(text, encoding="utf-8")
        out = _path(cfg["paths"]["packs"]) / genre / "schema_candidate.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        metrics.update({f"{genre}/types": len(schema["types"]), f"{genre}/pruned_clusters": len(schema["pruned_clusters"]),
                        f"{genre}/gate_problems": problems, f"{genre}/written": str(out)})
    return metrics


def coverage_sections(cfg: dict[str, Any], genre: str) -> list[Section]:
    """Plan §3.5 sample: schema.coverage_check_sections random sections (seeded) of dev documents that were not
    in the extraction sample, each at least induce.extract.min_section_chars long. The test split is never read."""
    import random

    vcfg = cfg["induce"]["coverage"]
    split = load_manifest(_path(cfg["paths"]["splits"]) / f"{genre}.json")
    excluded = set(json.loads(_path(vcfg["exclude_sample"].format(genre=genre)).read_text(encoding="utf-8")))
    held_out = {d for d, s in split.items() if s == "dev" and d not in excluded}
    pool = [s for s in read_jsonl(_path(cfg["paths"]["data_processed"]) / genre / "sections.jsonl", Section)
            if s.doc_id in held_out and len(s.text) >= cfg["induce"]["extract"]["min_section_chars"]]
    pool.sort(key=lambda s: (s.doc_id, s.idx))
    return random.Random(cfg["run"]["seed"]).sample(pool, min(cfg["schema"]["coverage_check_sections"], len(pool)))


def run_coverage(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Plan §3.5: the teacher lists what each held-out section states and maps it to a schema type or "none"; every
    "none" gets a second look among the induce.coverage.verify_top_k most similar types (by embedding). Gate metric:
    the share of sections with a statement still "none" after the second look (< schema.max_unmapped_fraction); the
    first-pass share is reported alongside. A copy of the checked candidate and its fingerprint are kept."""
    import csv
    import shutil
    from collections import Counter

    import yaml

    import engine.embed as E
    from experiments.run import resolve_device
    from models.teacher import PROMPTS, load_teacher
    from packs.schema_induction import COVERAGE_PROMPT, VERIFY_PROMPT, classify_sections, schema_fingerprint, second_look

    vcfg = cfg["induce"]["coverage"]
    tcfg, ecfg = cfg["models"]["teacher"], cfg["models"]["embedder"]
    device = resolve_device(cfg["run"].get("device", "auto"))
    teacher = load_teacher(tcfg, device=device, seed=cfg["run"]["seed"])
    embedder = None
    metrics: dict[str, Any] = {"model": tcfg["id"], "route": teacher._route(), "embedder": ecfg["id"],
                               "verify_top_k": vcfg["verify_top_k"]}
    for genre in _genres(cfg):
        schema_path = _path(vcfg["schema"].format(genre=genre))
        schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        shutil.copy(schema_path, run_dir / f"schema_checked_{genre}.yaml")
        types = schema["types"]
        sections = coverage_sections(cfg, genre)
        results = classify_sections(teacher, sections, cfg["genre_info"][genre], types,
                                    max_assertions=cfg["induce"]["max_assertions"], max_tokens=vcfg["max_tokens"],
                                    batch_size=cfg["induce"]["batch_size"])
        rows = []
        for s, r in zip(sections, results):
            items = [{**a.model_dump(), "first_type": a.type} for a in r.assertions] if r is not None else []
            rows.append({"doc_id": s.doc_id, "section_idx": s.idx, "heading": s.heading, "text": s.text,
                         "assertions": items, "failed": r is None})
        pending = [(r, a) for r in rows for a in r["assertions"] if a["type"] == "none"]
        if pending:
            if embedder is None:
                embedder = E.Embedder(ecfg["id"], devices=[device], dtype=ecfg.get("dtype", "float16"),
                                      max_length=ecfg.get("max_length", 1024), batch_size=ecfg.get("batch_size", 32),
                                      normalize=True, revision=ecfg.get("revision"))
            final = second_look(teacher, embedder, [(a["claim"], r["heading"]) for r, a in pending],
                                cfg["genre_info"][genre], types, top_k=vcfg["verify_top_k"],
                                max_tokens=vcfg["verify_max_tokens"], batch_size=cfg["induce"]["batch_size"])
            for (r, a), t in zip(pending, final):
                a["type"] = t
        hits = Counter(a["type"] for r in rows for a in r["assertions"])
        for r in rows:
            r["first_pass_unmapped"] = sum(a["first_type"] == "none" for a in r["assertions"])
            r["unmapped"] = sum(a["type"] == "none" for a in r["assertions"])
        with open(run_dir / f"coverage_{genre}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(run_dir / f"unmapped_{genre}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "section_idx", "heading", "claim", "first_pass", "after_second_look", "section_text"])
            for r in rows:
                for a in r["assertions"]:
                    if a["first_type"] == "none":
                        w.writerow([r["doc_id"], r["section_idx"], r["heading"], a["claim"], a["first_type"], a["type"],
                                    r["text"][:2000]])
        ok = [r for r in rows if not r["failed"]]
        n_claims = sum(len(r["assertions"]) for r in ok)
        rate = sum(r["unmapped"] > 0 for r in ok) / max(1, len(ok))
        first = sum(r["first_pass_unmapped"] > 0 for r in ok)
        metrics.update({
            f"{genre}/schema": str(schema_path), f"{genre}/schema_fingerprint": schema_fingerprint(schema),
            f"{genre}/types": len(types), f"{genre}/sections": len(rows),
            f"{genre}/failed_sections": sum(r["failed"] for r in rows), f"{genre}/claims": n_claims,
            f"{genre}/first_pass_sections_with_unmapped": first,
            f"{genre}/first_pass_unmapped_section_rate": round(first / max(1, len(ok)), 3),
            f"{genre}/second_look_statements": len(pending),
            f"{genre}/recovered_statements": sum(a["type"] != "none" for _, a in pending),
            f"{genre}/sections_with_unmapped": sum(r["unmapped"] > 0 for r in ok),
            f"{genre}/unmapped_section_rate": round(rate, 3),
            f"{genre}/unmapped_claim_rate": round(hits["none"] / max(1, n_claims), 3),
            f"{genre}/passes": rate < cfg["schema"]["max_unmapped_fraction"],
            f"{genre}/types_never_hit": sorted(t["id"] for t in types if hits[t["id"]] == 0),
            f"{genre}/type_hits": dict(hits.most_common())})
    metrics["prompt"] = f"{COVERAGE_PROMPT} {PROMPTS.get(COVERAGE_PROMPT).tag}"
    metrics["verify_prompt"] = f"{VERIFY_PROMPT} {PROMPTS.get(VERIFY_PROMPT).tag}"
    return metrics


def run_freeze(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Plan §3.6: candidate -> packs/<genre>/schema.yaml, version 1. Requires the reviewer's sign-off
    (induce.freeze.signed_off_by), no field problems, and a passing coverage run on this very candidate. Never
    overwrites a frozen schema: a change after freezing is a new pack version (CLAUDE.md)."""
    import datetime as dt
    import hashlib

    import yaml

    from packs.schema_induction import schema_fingerprint
    from packs.schema_review import validate_schema

    fcfg, scfg = cfg["induce"]["freeze"], cfg["schema"]
    if not fcfg.get("signed_off_by"):
        raise ValueError("freezing needs the reviewer's sign-off: set induce.freeze.signed_off_by")
    cov_dir = _path(fcfg["coverage_run"])
    cov = json.loads((cov_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        out = _path(cfg["paths"]["packs"]) / genre / "schema.yaml"
        if out.exists():
            raise FileExistsError(f"{out} is frozen; a schema change is a new pack version (record it in CHANGELOG)")
        cand_path = _path(cfg["induce"]["coverage"]["schema"].format(genre=genre))
        cand_text = cand_path.read_text(encoding="utf-8")
        cand = yaml.safe_load(cand_text)
        if not cov.get(f"{genre}/passes"):
            raise ValueError(f"{genre}: coverage run {cov_dir.name} does not pass "
                             f"({cov.get(f'{genre}/unmapped_section_rate')} >= {scfg['max_unmapped_fraction']})")
        if cov.get(f"{genre}/schema") and _path(cov[f"{genre}/schema"]).resolve() != cand_path.resolve():
            raise ValueError(f"{genre}: coverage run {cov_dir.name} checked {cov[f'{genre}/schema']}, not {cand_path}")
        if cov.get(f"{genre}/schema_fingerprint") != schema_fingerprint(cand):
            raise ValueError(f"{genre}: {cand_path.name} changed since coverage run {cov_dir.name} checked it; "
                             f"rerun the coverage stage on this candidate")
        need_labels = cfg["presence"]["label_sources"].get(genre) != "teacher"
        problems = validate_schema(cand, min_types=scfg["min_types"], max_types=scfg["max_types"],
                                   require_label_source=need_labels)
        if problems:
            raise ValueError(f"{genre}: candidate fails the gate:\n  " + "\n  ".join(problems))
        frozen = {"genre": genre, "version": 1, "status": "frozen",
                  "frozen": {"date": dt.date.today().isoformat(), "signed_off_by": fcfg["signed_off_by"],
                             "coverage_run": cov_dir.name,
                             "unmapped_section_rate": cov.get(f"{genre}/unmapped_section_rate"),
                             "candidate_sha1": hashlib.sha1(cand_text.encode("utf-8")).hexdigest()[:12]},
                  **{k: v for k, v in cand.items() if k not in ("genre", "version", "status", "gate_problems")}}
        text = yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True, width=120)
        out.write_text(text, encoding="utf-8")
        (run_dir / f"schema_{genre}.yaml").write_text(text, encoding="utf-8")
        metrics.update({f"{genre}/written": str(out), f"{genre}/types": [t["id"] for t in frozen["types"]]})
    return metrics


def run_relink(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Claims of induce.cluster.claims -> claims_<genre>.jsonl re-linked to the current data/processed sections, for
    every genre with induce.relink.old_sections[genre] (the sections file the claims were extracted from)."""
    from collections import defaultdict

    from packs.schema_induction import load_claims, relink_claims

    rcfg = cfg["induce"]["relink"]
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        old_path = (rcfg.get("old_sections") or {}).get(genre)
        if not old_path:
            continue
        claims = load_claims(_path(cfg["induce"]["cluster"]["claims"][genre]))
        docs = {c["doc_id"] for c in claims}

        def sections(path: Path) -> dict[str, list[Section]]:
            by_doc: dict[str, list[Section]] = defaultdict(list)
            for sec in read_jsonl(path, Section):
                if sec.doc_id in docs:
                    by_doc[sec.doc_id].append(sec)
            return by_doc

        old = sections(_path(old_path))
        new = sections(_path(cfg["paths"]["data_processed"]) / genre / "sections.jsonl")
        out, counts = relink_claims(claims, old, new)
        with open(run_dir / f"claims_{genre}.jsonl", "w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_docs = sum(len(old[d]) != len(new[d]) for d in docs)
        moved = sum(c["section_idx"] != c["section_idx_unsplit"] for c in out)
        metrics.update({f"{genre}/claims": len(out), f"{genre}/documents": len(docs),
                        f"{genre}/documents_with_split_sections": split_docs, f"{genre}/claims_index_changed": moved,
                        f"{genre}/claims_span_ungrounded": sum(not c.get("span_grounded", True) for c in out),
                        **{f"{genre}/relink_{k}": v for k, v in sorted(counts.items())}})
        print(f"[relink {genre}] {len(out)} claims, {moved} moved: {counts}", flush=True)
    return metrics


def dt_stamp() -> str:
    import datetime as dt

    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    stage = cfg["induce"]["stage"]
    if stage == "eye_test":
        return run_eye_test(cfg, run_dir)
    if stage == "extract":
        return run_extract(cfg, run_dir)
    if stage == "cluster":
        return run_cluster(cfg, run_dir)
    if stage == "name":
        return run_name(cfg, run_dir)
    if stage == "review":
        return run_review(cfg, run_dir)
    if stage == "candidate":
        return run_candidate(cfg, run_dir)
    if stage == "coverage":
        return run_coverage(cfg, run_dir)
    if stage == "freeze":
        return run_freeze(cfg, run_dir)
    if stage == "relink":
        return run_relink(cfg, run_dir)
    raise NotImplementedError(f"induce stage {stage!r} is implemented in a later Phase 3 task")
