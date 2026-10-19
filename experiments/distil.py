"""Presence labels and the presence classifier (plan §4.1–4.3). Stages, chosen by `distil.stage`:

  labels   (4.1) human annotations -> section labels through each type's label_source, for packs whose
           presence.label_sources entry names an annotation file (data/processed/<genre>/<source>_labels.jsonl);
           writes labels_<genre>.jsonl, a per-type count table (reports/tables/presence_labels_<genre>.csv) and the
           list of types that need teacher labels
  select   (4.1, 4.2) the human sanity set and the teacher set -> packs/<genre>/label_sets.json (tracked; section ids
           and text sha1, so a changed corpus cannot silently change them). Genre with human annotations: the human
           set takes distil.label_sets.annotated_per_document sections from each of randomly drawn whole annotated
           documents, the teacher set comes from the others. Otherwise: human set from dev documents outside the schema-induction sample, teacher set
           from train; one section per document; both stratified by position and length (packs/presence_labels.py)
  select_documents (Phase 5, document-level presence) whole dev documents, every section, as the `documents`
           set of label_sets.json (distil.label_sets.documents_per_genre; stratified by year and category; only
           sources the indexed corpus has; never the induction sample, other label sets or test)
  teacher_labels (4.1) the two-pass teacher (prompts label_presence + label_presence_check) over the label sets;
           every type of the schema is labelled; writes teacher_labels_<genre>.jsonl (source teacher:<route>)
  score    (4.2) teacher labels vs a reference (OPP-115 labels, or the human set read back from the page), per type,
           final and first pass only -> reports/tables/teacher_vs_<name>_<genre>.csv
  heads    (4.3) presence.head linear: one logistic head per type on the section embeddings, trained on
           presence.train_labels rows, temperature-scaled and thresholded on the reference set's fit half, scored on its
           eval half (vs the reference, vs the teacher, vs annotations) next to the teacher's own agreement with the
           reference (the ceiling); heads saved to packs/<genre>/classifier/linear_<train_labels>/
  prototypes (4.4) the teacher's evidence spans on training sections -> 10-30 prototypes per type (packs/prototypes.py)
           -> packs/<genre>/prototypes/{prototypes.json, tokens.npz}; table reports/tables/prototypes_<genre>.csv
  cascade  (4.5, 4.6) prototype scores, the band (from the training sections), reranker scores for band pairs, the
           combination layer and per-type thresholds (fit half) -> packs/<genre>/cascade.json; ablations on the eval
           half (classifier only, prototypes only, prototypes + reranker, classifier + prototypes, full cascade) with
           latency per section -> reports/tables/presence_ablation_<genre>.csv, presence_cascade_<genre>.csv, figure
           reports/figures/presence_band_<genre>.png
  profiles (4.7) the presence model (packs/<genre>/cascade.json over the linear heads and prototypes; no reranker in
           the lean tier) over every section of every train-split document -> packs/<genre>/profiles/part-*.parquet
           (prob_<type> = best section probability, bit_<type> = at the type's threshold), resumable by chunk
  gate     (4.8) the gate table from the measured artefacts: the presence model's macro F1 on the reference eval half
           vs presence.gate.relative_to_ceiling x the teacher's (reports/tables/presence_ablation_<genre>.csv), the
           profiles' coverage of the indexable corpus and their throughput (distil.gate.profiles_run) ->
           reports/tables/presence_gate.csv
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from data.schema import Section, read_jsonl
from packs.presence_labels import annotation_labels, teacher_types

REPO = Path(__file__).resolve().parent.parent


def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def _genres(cfg: dict[str, Any]) -> list[str]:
    return [cfg["pack"]] if cfg.get("pack") else list(cfg["genres"])


def load_schema(cfg: dict[str, Any], genre: str) -> dict[str, Any]:
    schema = yaml.safe_load(open(_path(cfg["paths"]["packs"]) / genre / "schema.yaml", encoding="utf-8"))
    if schema.get("status") != "frozen":
        raise ValueError(f"packs/{genre}/schema.yaml is not frozen")
    return schema


def run_labels(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        source = cfg["presence"]["label_sources"][genre]
        schema = load_schema(cfg, genre)
        teach = teacher_types(schema["types"])
        metrics.update({f"{genre}/schema_version": schema["version"], f"{genre}/types": len(schema["types"]),
                        f"{genre}/teacher_types": len(teach)})
        (run_dir / f"teacher_types_{genre}.json").write_text(json.dumps(teach, indent=1), encoding="utf-8")
        processed = _path(cfg["paths"]["data_processed"]) / genre
        ann_path = processed / f"{source}_labels.jsonl"
        if source == "teacher" or not ann_path.exists():
            print(f"[labels {genre}] source {source!r}: no annotation file, every type needs teacher labels")
            continue
        annotations = [json.loads(line) for line in open(ann_path, encoding="utf-8")]
        docs = {a["doc_id"] for a in annotations}
        sections: dict[str, list[Section]] = defaultdict(list)
        for s in read_jsonl(processed / "sections.jsonl", Section):
            if s.doc_id in docs:
                sections[s.doc_id].append(s)
        rows, counts = annotation_labels(schema["types"], annotations, sections, source=source)
        with open(run_dir / f"labels_{genre}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({**r, "schema_version": schema["version"]}) + "\n")
        n = counts["sections"]
        table = [{"genre": genre, "type_id": t["id"], "label_source": "teacher" if t["id"] in teach else source,
                  "positives": counts["positives"].get(t["id"], ""),
                  "prevalence": round(counts["positives"][t["id"]] / n, 4) if t["id"] in counts["positives"] else ""}
                 for t in schema["types"]]
        out = _path(cfg["paths"]["tables"]) / f"presence_labels_{genre}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        pos = counts["positives"]
        metrics.update({f"{genre}/sections": n, f"{genre}/documents": counts["documents"],
                        f"{genre}/sections_without_annotation": counts["sections_without_annotation"],
                        f"{genre}/annotated_types": len(pos),
                        f"{genre}/types_under_20_positives": sum(v < 20 for v in pos.values()),
                        f"{genre}/types_with_0_positives": sum(v == 0 for v in pos.values())})
        print(f"[labels {genre}] {n} sections, {len(pos)} types from {source}, {len(teach)} teacher types", flush=True)
    return metrics


def _annotated_docs(cfg: dict[str, Any], genre: str) -> set[str]:
    source = cfg["presence"]["label_sources"][genre]
    path = _path(cfg["paths"]["data_processed"]) / genre / f"{source}_labels.jsonl"
    if source == "teacher" or not path.exists():
        return set()
    return {json.loads(line)["doc_id"] for line in open(path, encoding="utf-8")}


def run_select(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import hashlib
    import random

    from data.splits import load_manifest
    from packs.presence_labels import sample_sections

    dcfg, lcfg = cfg["distil"], cfg["distil"]["label_sets"]
    seed = cfg["splits"]["seed"]
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        out = _path(cfg["paths"]["packs"]) / genre / "label_sets.json"
        if out.exists() and not dcfg.get("overwrite_label_sets"):
            raise FileExistsError(f"{out} exists; the label sets are fixed once chosen (distil.overwrite_label_sets)")
        split = load_manifest(_path(cfg["paths"]["splits"]) / f"{genre}.json")
        annotated = _annotated_docs(cfg, genre)
        if annotated:
            k = lcfg["annotated_per_document"]
            eligible: dict[str, int] = defaultdict(int)
            for sec in read_jsonl(_path(cfg["paths"]["data_processed"]) / genre / "sections.jsonl", Section):
                if sec.doc_id in annotated and len(sec.text) >= lcfg["min_chars"]:
                    eligible[sec.doc_id] += 1
            docs = sorted(annotated)
            random.Random(seed).shuffle(docs)
            human_docs: set[str] = set()
            for d in docs:  # draw whole documents until the human set can be filled
                if sum(min(k, eligible[x]) for x in human_docs) >= dcfg["human_set_per_genre"]:
                    break
                human_docs.add(d)
            pools = {"human": human_docs, "teacher": set(annotated) - human_docs}
            per_doc = {"human": k, "teacher": 10 ** 6}
        else:
            sample_path = _path(lcfg["exclude_sample"].format(genre=genre))
            excluded = set(json.loads(sample_path.read_text(encoding="utf-8"))) if sample_path.exists() else set()
            pools = {"human": {d for d, s in split.items() if s == "dev" and d not in excluded},
                     "teacher": {d for d, s in split.items() if s == "train"}}
            per_doc = {"human": 1, "teacher": 1}
        if any(split.get(d) == "test" for pool in pools.values() for d in pool):
            raise ValueError("a label set would read the sealed test split")
        wanted = pools["human"] | pools["teacher"]
        by_doc: dict[str, list[Section]] = defaultdict(list)
        for sec in read_jsonl(_path(cfg["paths"]["data_processed"]) / genre / "sections.jsonl", Section):
            if sec.doc_id in wanted:
                by_doc[sec.doc_id].append(sec)
        sets = {}
        for name, n in (("human", dcfg["human_set_per_genre"]), ("teacher", dcfg["label_sections_per_genre"])):
            chosen = sample_sections({d: by_doc[d] for d in pools[name] if d in by_doc}, n=n, seed=seed,
                                     min_chars=lcfg["min_chars"], per_doc=per_doc[name])
            sets[name] = [{"doc_id": c.doc_id, "section_idx": c.idx,
                           "text_sha1": hashlib.sha1(c.text.encode("utf-8")).hexdigest()} for c in chosen]
            metrics.update({f"{genre}/{name}_sections": len(chosen),
                            f"{genre}/{name}_documents": len({c.doc_id for c in chosen})})
        overlap = {r["doc_id"] for r in sets["human"]} & {r["doc_id"] for r in sets["teacher"]}
        if overlap:
            raise ValueError(f"{len(overlap)} documents in both label sets")
        out.write_text(json.dumps({"genre": genre, "seed": seed, "source": "annotated" if annotated else "splits",
                                   "human": sets["human"], "teacher": sets["teacher"]}, indent=1) + "\n",
                       encoding="utf-8")
        print(f"[select {genre}] human {len(sets['human'])}, teacher {len(sets['teacher'])} -> {out}", flush=True)
    return metrics


def run_select_documents(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Whole documents for document-level presence: `documents_per_genre` dev documents (stratified by year and
    category), every section of each, added to label_sets.json as the `documents` set. Only documents from sources
    that the indexed corpus (train split) also has, so they are segmented like the documents whose profiles the
    calibration will be applied to; never the schema-induction sample, another label set's documents, or test."""
    import hashlib

    from data.schema import Document
    from data.splits import load_manifest
    from packs.presence_labels import sample_documents

    lcfg = cfg["distil"]["label_sets"]
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        out = _path(cfg["paths"]["packs"]) / genre / "label_sets.json"
        sets = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"genre": genre}
        if "documents" in sets and not cfg["distil"].get("overwrite_label_sets"):
            raise FileExistsError(f"{out} has a documents set; label sets are fixed once chosen")
        split = load_manifest(_path(cfg["paths"]["splits"]) / f"{genre}.json")
        processed = _path(cfg["paths"]["data_processed"]) / genre
        docs = list(read_jsonl(processed / "documents.jsonl", Document))
        indexed_sources = {d.source for d in docs if split.get(d.id) == "train"}
        sample_path = _path(lcfg["exclude_sample"].format(genre=genre))
        excluded = set(json.loads(sample_path.read_text(encoding="utf-8"))) if sample_path.exists() else set()
        excluded |= {r["doc_id"] for name, refs in sets.items() if isinstance(refs, list) for r in refs}
        pool = [{"id": d.id, "date": d.date, "category": d.category} for d in docs
                if split.get(d.id) == "dev" and d.source in indexed_sources and d.id not in excluded]
        chosen = set(sample_documents(pool, n=lcfg["documents_per_genre"], seed=cfg["splits"]["seed"]))
        if any(split.get(d) == "test" for d in chosen):
            raise ValueError("the documents set would read the sealed test split")
        refs = [{"doc_id": sec.doc_id, "section_idx": sec.idx, "text_sha1": hashlib.sha1(sec.text.encode("utf-8")).hexdigest()}
                for sec in read_jsonl(processed / "sections.jsonl", Section) if sec.doc_id in chosen]
        sets["documents"] = refs
        sets["documents_selection"] = {"seed": cfg["splits"]["seed"], "documents": len(chosen), "pool": len(pool),
                                       "split": "dev", "sources": sorted(indexed_sources)}
        out.write_text(json.dumps(sets, indent=1) + "\n", encoding="utf-8")
        metrics.update({f"{genre}/documents_documents": len(chosen), f"{genre}/documents_sections": len(refs),
                        f"{genre}/documents_pool": len(pool)})
        print(f"[select_documents {genre}] {len(chosen)} documents, {len(refs)} sections (pool {len(pool)}) -> {out}",
              flush=True)
    return metrics


def load_label_set(cfg: dict[str, Any], genre: str, name: str) -> list[Section]:
    """The sections of one label set, checked against their stored text sha1."""
    import hashlib

    refs = json.loads((_path(cfg["paths"]["packs"]) / genre / "label_sets.json").read_text(encoding="utf-8"))[name]
    want = {(r["doc_id"], r["section_idx"]): r["text_sha1"] for r in refs}
    found: dict[tuple[str, int], Section] = {}
    for sec in read_jsonl(_path(cfg["paths"]["data_processed"]) / genre / "sections.jsonl", Section):
        key = (sec.doc_id, sec.idx)
        if key in want:
            if hashlib.sha1(sec.text.encode("utf-8")).hexdigest() != want[key]:
                raise ValueError(f"{genre} {name} set: section {key} changed since it was selected")
            found[key] = sec
    missing = [k for k in want if k not in found]
    if missing:
        raise ValueError(f"{genre} {name} set: {len(missing)} sections missing, e.g. {missing[:2]}")
    return [found[(r["doc_id"], r["section_idx"])] for r in refs]


def run_teacher_labels(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import engine.embed as E
    from experiments.run import resolve_device
    from models.teacher import PROMPTS, load_teacher
    from packs.presence_labels import CHECK_PROMPT, LABEL_PROMPT, teacher_labels

    tl = cfg["distil"]["teacher_labels"]
    tcfg, ecfg = cfg["models"]["teacher"], cfg["models"]["embedder"]
    device = resolve_device(cfg["run"].get("device", "auto"))
    teacher = load_teacher(tcfg, device=device, seed=cfg["run"]["seed"])
    embedder = E.Embedder(ecfg["id"], devices=[device], dtype=ecfg.get("dtype", "float16"),
                          max_length=ecfg.get("max_length", 1024), batch_size=ecfg.get("batch_size", 32),
                          normalize=True, revision=ecfg.get("revision"))
    route = teacher._route()
    metrics: dict[str, Any] = {"model": tcfg["id"], "route": route, "embedder": ecfg["id"],
                               "prompt": f"{LABEL_PROMPT} {PROMPTS.get(LABEL_PROMPT).tag}",
                               "check_prompt": f"{CHECK_PROMPT} {PROMPTS.get(CHECK_PROMPT).tag}"}
    for genre in _genres(cfg):
        schema = load_schema(cfg, genre)
        types = schema["types"]
        store_path = _path(cfg["paths"]["embeddings"]) / E.embedding_slug(ecfg) / genre / "sections"
        store = E.EmbeddingStore.load(store_path) if (store_path / "rows.jsonl").exists() else None
        out = open(run_dir / f"teacher_labels_{genre}.jsonl", "w", encoding="utf-8")
        for name in tl["sets"]:
            sections = load_label_set(cfg, genre, name)
            if tl.get("sample"):  # a seeded random subset (calibration), in set order
                import random

                keep = set(random.Random(cfg["splits"]["seed"]).sample(range(len(sections)), min(tl["sample"], len(sections))))
                sections = [x for i, x in enumerate(sections) if i in keep]
            sections = sections[: tl.get("limit") or None]
            vecs = store.lookup([s.text for s in sections]) if store is not None else [None] * len(sections)
            rows = teacher_labels(teacher, embedder, sections, cfg["genre_info"][genre], types,
                                  label_types=[t["id"] for t in types], max_assertions=tl["max_assertions"],
                                  max_tokens=tl["max_tokens"], check_max_tokens=tl["check_max_tokens"],
                                  k_claim=tl["k_claim"], k_section=tl["k_section"], cap=tl["max_candidates"],
                                  batch_size=cfg["induce"]["batch_size"],
                                  section_vectors={(s.doc_id, s.idx): v for s, v in zip(sections, vecs) if v is not None})
            for r in rows:
                out.write(json.dumps({**r, "set": name, "source": f"teacher:{route}", "schema_version":
                                      schema["version"]}, ensure_ascii=False) + "\n")
            ok = [r for r in rows if not r["failed"]]
            labels = sum(sum(r["labels"].values()) for r in ok)
            second = sum(v == "second_look" for r in ok for v in r["found_by"].values())
            grounded = sum(v for r in ok for v in r["grounded"].values())
            metrics.update({f"{genre}/{name}/sections": len(rows), f"{genre}/{name}/failed": len(rows) - len(ok),
                            f"{genre}/{name}/check_failed": sum(r["check_failed"] for r in ok),
                            f"{genre}/{name}/labels": labels,
                            f"{genre}/{name}/labels_per_section": round(labels / max(1, len(ok)), 2),
                            f"{genre}/{name}/second_look_labels": second,
                            f"{genre}/{name}/second_look_share": round(second / max(1, labels), 3),
                            f"{genre}/{name}/grounded_share": round(grounded / max(1, labels), 3),
                            f"{genre}/{name}/sections_with_no_type": sum(not any(r["labels"].values()) for r in ok)})
            print(f"[teacher_labels {genre}/{name}] {len(ok)}/{len(rows)} sections, {labels} labels, "
                  f"{second} from the second look", flush=True)
        out.close()
    return metrics


def load_reference(path: Path, proj: dict[str, str] | None = None) -> dict[tuple[str, int], dict[str, Any]]:
    """Reference labels as {key: {present, unsure}}: a labels-stage file (rows with labels {type: 0/1}) or the
    labelling page's answers (rows with present/unsure, read back from its database as JSON lines). proj maps type
    ids of an earlier schema version to the current one (packs/schema_versions.projection)."""
    from packs.schema_versions import project_labels

    proj = proj or {}
    ref = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("failed"):  # a judge section that failed has no labels: leave it out, never count it as all-negative
            continue
        key = (r["doc_id"], r.get("section_idx", r.get("idx")))
        if "labels" in r:
            labels = project_labels(r, proj)["labels"]
            ref[key] = {"present": [t for t, v in labels.items() if v], "unsure": [], "types": list(labels)}
        else:
            ids = lambda xs: list(dict.fromkeys(proj.get(t, t) for t in xs or []))  # noqa: E731
            ref[key] = {"present": ids(r.get("present")), "unsure": ids(r.get("unsure")), "types": None}
    return ref


def run_score(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Teacher labels vs a reference (OPP-115 labels or the human set), per type, final and first pass only; types =
    those the reference labels (a labels-stage file) or every schema type (page answers). Writes
    reports/tables/teacher_vs_<name>_<genre>.csv and macro/micro numbers as metrics."""
    from packs.presence_labels import agreement

    from packs.schema_versions import project_labels, projection

    sc = cfg["distil"]["score"]
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        schema = load_schema(cfg, genre)
        proj = projection(schema)  # labels made under an earlier version are read in this one
        teacher = {}
        for line in open(_path(sc["teacher"].format(genre=genre)), encoding="utf-8"):
            r = json.loads(line)
            if not r["failed"]:
                teacher[(r["doc_id"], r["section_idx"])] = project_labels(r, proj)
        ref = load_reference(_path(sc["reference"].format(genre=genre)), proj)
        typed = next(iter(ref.values()))["types"]
        types = typed or [t["id"] for t in schema["types"]]
        metrics[f"{genre}/schema_version"] = schema["version"]
        rows = agreement(ref, teacher, types)
        out = _path(cfg["paths"]["tables"]) / f"teacher_vs_{sc['name']}_{genre}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])

        def summary(suffix: str) -> dict[str, Any]:
            tp, fp, fn = (sum(r[k + suffix] for r in rows) for k in ("tp", "fp", "fn"))
            f1s = [r["f1" + suffix] for r in rows if r["human_positives"] >= sc.get("min_positives", 1)
                   and r["f1" + suffix] is not None]
            return {"micro_precision": round(tp / max(1, tp + fp), 3), "micro_recall": round(tp / max(1, tp + fn), 3),
                    "micro_f1": round(2 * tp / max(1, 2 * tp + fp + fn), 3),
                    "macro_f1": round(sum(f1s) / max(1, len(f1s)), 3), "types_scored": len(f1s)}

        n = len([k for k in ref if k in teacher])
        metrics.update({f"{genre}/sections": n, **{f"{genre}/final/{k}": v for k, v in summary("").items()},
                        **{f"{genre}/first_pass/{k}": v for k, v in summary("_first_pass").items()}})
        print(f"[score {genre}] {n} sections: final {summary('')} | first pass {summary('_first_pass')}", flush=True)
    return metrics


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])


def run_heads(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import numpy as np

    from engine.classifier import LinearHeads, ece, prf_table, summary
    from experiments.presence_data import PresenceData

    pc = cfg["presence"]
    mode, lin = pc.get("train_labels", "plan"), pc["linear"]
    min_pos = cfg["distil"].get("min_type_positives_report", 5)
    metrics: dict[str, Any] = {"head": "linear", "train_labels": mode}
    for genre in _genres(cfg):
        D = PresenceData(cfg, genre)
        metrics.update({f"{genre}/{k}": v for k, v in D.describe().items()})
        cache: dict[tuple, Any] = {}

        def train(C: float) -> LinearHeads:
            h = LinearHeads(D.types, C=C, class_weight=lin["class_weight"])
            for t in D.types:
                keys, y = D.train_rows(t, mode)
                sig = (keys[0], keys[-1], len(keys))
                if sig not in cache:
                    cache[sig] = D.X(keys)
                h.fit_type(t, cache[sig], y)
            return h

        fit, ev = D.reference_keys("fit"), D.reference_keys("eval")
        Xf, Xe = D.X(fit), D.X(ev)
        Yf, Ye = D.Y(fit, D.reference), D.Y(ev, D.reference)
        grid = lin.get("C_grid") or [lin["C"]]
        fit_scores = {}
        for C in grid:  # selection on the fit half only; the eval half is never used to choose
            h = train(C)
            h.calibrate(Xf, Yf)
            h.tune_thresholds(Xf, Yf)
            fit_scores[C] = summary(prf_table(Yf, h.predict_proba(Xf), D.types, thresholds=h.threshold),
                                    min_positives=min_pos)["macro_f1"] or 0.0
        best_C = max(grid, key=lambda C: (fit_scores[C], -abs(C - lin["C"])))
        heads = train(best_C)
        metrics.update({f"{genre}/C": best_C, f"{genre}/C_fit_macro_f1": {str(k): v for k, v in fit_scores.items()}})
        raw = heads.predict_proba(Xe)
        heads.calibrate(Xf, Yf)
        heads.tune_thresholds(Xf, Yf)
        Pe = heads.predict_proba(Xe)
        out_dir = _path(cfg["paths"]["packs"]) / genre / "classifier" / f"linear_{mode}"
        heads.save(out_dir)
        heads.save(run_dir / f"heads_{genre}")
        vs_ref = prf_table(Ye, Pe, D.types, thresholds=heads.threshold)
        vs_teacher = prf_table(D.Y(ev, D.teacher), Pe, D.types, thresholds=heads.threshold)
        ceiling = prf_table(Ye, D.Y(ev, D.teacher).astype(float), D.types, thresholds=np.full(len(D.types), 0.5))
        table = []
        for j, t in enumerate(D.types):
            table.append({"type_id": t, "trained_on": "annotations" if (mode == "plan" and t in D.rule_types) else "teacher",
                          "mode": heads.info[t]["mode"], "train_rows": heads.info[t]["rows"],
                          "train_positives": heads.info[t]["positives"], "eval_positives": vs_ref[j]["positives"],
                          "precision": vs_ref[j]["precision"], "recall": vs_ref[j]["recall"], "f1": vs_ref[j]["f1"],
                          "f1_vs_teacher": vs_teacher[j]["f1"], "teacher_f1_vs_reference": ceiling[j]["f1"],
                          "threshold": vs_ref[j]["threshold"], "temperature": round(float(heads.temperature[j]), 3)})
        _write_table(_path(cfg["paths"]["tables"]) / f"presence_heads_{genre}_{mode}.csv", table)
        res = {"vs_reference": summary(vs_ref, min_positives=min_pos), "vs_teacher": summary(vs_teacher, min_positives=min_pos),
               "teacher_vs_reference_ceiling": summary(ceiling, min_positives=min_pos),
               "ece_raw": ece(Ye, raw), "ece_calibrated": ece(Ye, Pe),
               "heads": sum(i["mode"] == "head" for i in heads.info.values()),
               "priors": sum(i["mode"] == "prior" for i in heads.info.values())}
        if D.annotations:
            rule = [D.types.index(t) for t in D.rule_types]
            ann = [k for k in ev if k in D.annotations]
            Ya = D.Y(ann, D.annotations)[:, rule]
            Pa = heads.predict_proba(D.X(ann))[:, rule]
            res["vs_annotations_rule_types"] = summary(prf_table(Ya, Pa, D.rule_types, thresholds=heads.threshold[rule]),
                                                       min_positives=min_pos)
        for k, v in res.items():
            if isinstance(v, dict):
                metrics.update({f"{genre}/{k}/{kk}": vv for kk, vv in v.items()})
            else:
                metrics[f"{genre}/{k}"] = v
        print(f"[heads {genre}] eval vs reference {res['vs_reference']} | ceiling (teacher) "
              f"{res['teacher_vs_reference_ceiling']} | ECE {res['ece_raw']} -> {res['ece_calibrated']}", flush=True)
    return metrics


def load_colbert(cfg: dict[str, Any]) -> Any:
    from engine.late_interaction import ColBERT
    from experiments.run import resolve_device

    li = cfg["models"]["late_interaction"]
    return ColBERT(li["id"], revision=li.get("revision"), device=resolve_device(cfg["run"].get("device", "auto")),
                   batch_size=li.get("batch_size", 64))


def run_prototypes(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import numpy as np

    from experiments.presence_data import PresenceData
    from packs.prototypes import n_prototypes, select_prototypes, span_candidates

    k_min, k_max = cfg["presence"]["prototypes_per_type"]
    colbert = load_colbert(cfg)
    metrics: dict[str, Any] = {"late_interaction": colbert.model_id, "revision": colbert.revision}
    for genre in _genres(cfg):
        D = PresenceData(cfg, genre, need_vectors=False)
        cands = span_candidates(D.teacher_full, D.types, keys=set(D.teacher_train))
        chosen: dict[str, list[dict[str, Any]]] = {}
        tokens, lengths, type_idx, table = [], [], [], []
        for j, t in enumerate(D.types):
            c = cands[t]
            if not c:
                chosen[t] = []
                table.append({"type_id": t, "candidates": 0, "prototypes": 0})
                continue
            enc = colbert.encode([x["text"] for x in c], mode="query")
            pooled = np.stack([e.astype(np.float32).mean(0) for e in enc])
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
            picked = select_prototypes(pooled, k=n_prototypes(len(c), k_min, k_max), seed=cfg["run"]["seed"])
            chosen[t] = [c[i] for i in picked]
            for i in picked:
                tokens.append(enc[i])
                lengths.append(len(enc[i]))
                type_idx.append(j)
            table.append({"type_id": t, "candidates": len(c), "prototypes": len(picked)})
        out = _path(cfg["paths"]["packs"]) / genre / "prototypes"
        out.mkdir(parents=True, exist_ok=True)
        (out / "prototypes.json").write_text(json.dumps({
            "genre": genre, "schema_version": D.schema["version"], "types": D.types, "late_interaction": colbert.model_id,
            "revision": colbert.revision, "source": cfg["presence"]["labels"]["teacher"], "prototypes": chosen},
            indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out / "tokens.npz", tokens=np.concatenate(tokens) if tokens else np.zeros((0, colbert.dim), np.float16),
                 lengths=np.array(lengths, dtype=np.int32), type_idx=np.array(type_idx, dtype=np.int32))
        _write_table(_path(cfg["paths"]["tables"]) / f"prototypes_{genre}.csv", table)
        n = [r["prototypes"] for r in table]
        metrics.update({f"{genre}/prototypes": sum(n), f"{genre}/types_without_prototypes": sum(x == 0 for x in n),
                        f"{genre}/types_under_10": sum(0 < x < 10 for x in n)})
        print(f"[prototypes {genre}] {sum(n)} prototypes for {len(n)} types ({sum(x == 0 for x in n)} without any)",
              flush=True)
    return metrics


def _band_figure(path: Path, genre: str, types: list[str], proto: Any, Y: Any, low: float, high: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(types)
    cols = 8
    rows = 1 + (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 1.9, rows * 1.5 + 1.0))
    ax = fig.add_subplot(rows, 1, 1)
    bins = np.linspace(min(proto.min(), 0), 1, 50)
    ax.hist(proto[Y == 0], bins=bins, alpha=0.6, density=True, label="labelled negative", color="#7a8791")
    ax.hist(proto[Y == 1], bins=bins, alpha=0.6, density=True, label="labelled positive", color="#2f6690")
    for x in (low, high):
        ax.axvline(x, color="#a3322a", lw=1)
    ax.set_title(f"{genre}: best prototype match per section and type (training sections); band [{low:.3f}, {high:.3f})",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    for j, t in enumerate(types):
        a = fig.add_subplot(rows, cols, cols + j + 1)
        a.hist(proto[Y[:, j] == 0, j], bins=bins, alpha=0.6, density=True, color="#7a8791")
        if (Y[:, j] == 1).any():
            a.hist(proto[Y[:, j] == 1, j], bins=bins, alpha=0.6, density=True, color="#2f6690")
        a.axvline(low, color="#a3322a", lw=0.6)
        a.axvline(high, color="#a3322a", lw=0.6)
        a.set_title(t[:28], fontsize=5.5)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_cascade(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import time

    import numpy as np

    from engine.classifier import LinearHeads, ece, prf_table, summary, tune_thresholds
    from engine.presence import FEATURES, Cascade, PrototypeScorer, Reranker, band_thresholds, rerank_pairs, type_query
    from experiments.presence_data import PresenceData
    from experiments.run import resolve_device

    pc = cfg["presence"]
    mode = pc.get("train_labels", "plan")
    min_pos = cfg["distil"].get("min_type_positives_report", 5)
    device = resolve_device(cfg["run"].get("device", "auto"))
    colbert = load_colbert(cfg)
    rcfg = cfg["models"]["reranker"]
    use_rr = bool((pc.get("cascade") or {}).get("reranker", True))
    reranker = Reranker(rcfg["id"], revision=rcfg.get("revision"), device=device, max_length=rcfg.get("max_length", 512),
                        batch_size=rcfg.get("batch_size", 16)) if use_rr else None
    metrics: dict[str, Any] = {"late_interaction": colbert.model_id, "reranker": rcfg["id"] if use_rr else None,
                               "train_labels": mode}
    for genre in _genres(cfg):
        D = PresenceData(cfg, genre)
        T = len(D.types)
        heads = LinearHeads.load(_path(cfg["paths"]["packs"]) / genre / "classifier" / f"linear_{mode}")
        if heads.types != D.types:
            raise ValueError(f"{genre}: classifier types differ from the schema; rerun the heads stage")
        scorer = PrototypeScorer.from_pack(colbert, _path(cfg["paths"]["packs"]) / genre / "prototypes", T,
                                           min_sentence_chars=cfg["segment"]["min_sentence_chars"])
        # band from the training sections (teacher labels)
        train = sorted(D.teacher_train)
        t0 = time.perf_counter()
        proto_train = scorer.score([D.sections[k].text for k in train])
        proto_rate = len(train) / (time.perf_counter() - t0)
        Ytr = D.Y(train, D.teacher)
        q_lo, q_hi = pc["band"].get("quantiles", [0.05, 0.95])
        low, high = band_thresholds(proto_train[Ytr == 1], proto_train[Ytr == 0], q_low=q_lo, q_high=q_hi)
        if pc["band"].get("low") is not None:
            low, high = pc["band"]["low"], pc["band"]["high"]
        _band_figure(_path(cfg["paths"]["figures"]) / f"presence_band_{genre}.png", genre, D.types, proto_train, Ytr, low, high)
        halves: dict[str, dict[str, Any]] = {}
        for half in ("fit", "eval"):
            keys = D.reference_keys(half)
            texts = [D.sections[k].text for k in keys]
            clf = heads.predict_proba(D.X(keys))
            t0 = time.perf_counter()
            proto = scorer.score(texts)
            t_proto = (time.perf_counter() - t0) / len(keys)
            pairs = rerank_pairs(proto, low=low, high=high)
            t0 = time.perf_counter()
            scores = reranker.score([(type_query(D.schema["types"][j]), texts[i]) for i, j in pairs]) if use_rr else []
            t_rr = (time.perf_counter() - t0) / len(keys)
            rr = np.full(proto.shape, np.nan, dtype=np.float32)
            for (i, j), v in zip(pairs, scores):
                rr[i, j] = v
            halves[half] = {"keys": keys, "Y": D.Y(keys, D.reference), "clf": clf, "proto": proto, "rr": rr,
                            "t_proto": t_proto, "t_rr": t_rr, "band_pairs": len(pairs)}
        # section embedding cost (the classifier's input) on a sample, for the latency column
        from engine.embed import Embedder

        ecfg = cfg["models"]["embedder"]
        emb = Embedder(ecfg["id"], devices=[device], dtype=ecfg.get("dtype", "float16"),
                       max_length=ecfg.get("max_length", 1024), batch_size=ecfg.get("batch_size", 32))
        sample = [D.sections[k].text for k in halves["eval"]["keys"][:64]]
        emb.encode(sample[:4])
        t0 = time.perf_counter()
        emb.encode(sample)
        t_emb = (time.perf_counter() - t0) / len(sample)
        del emb
        F, E = halves["fit"], halves["eval"]

        def stepped(proto: Any, rr: Any) -> Any:  # prototypes + reranker: clear miss 0, clear hit 1, band -> reranker
            return np.where(proto >= high, 1.0, np.where(proto < low, 0.0, np.nan_to_num(rr, nan=0.0)))

        variants: dict[str, tuple[Any, Any, float, bool]] = {}
        variants["classifier only"] = (E["clf"], heads.threshold, t_emb, True)
        th = tune_thresholds(F["Y"], F["proto"])
        variants["prototypes only"] = (E["proto"], th, E["t_proto"], False)
        if use_rr:
            th = tune_thresholds(F["Y"], stepped(F["proto"], F["rr"]))
            variants["prototypes + reranker"] = (stepped(E["proto"], E["rr"]), th, E["t_proto"] + E["t_rr"], False)
        nan_f, nan_e = np.full(F["rr"].shape, np.nan), np.full(E["rr"].shape, np.nan)
        cp = Cascade(D.types, low=low, high=high)
        cp.fit(F["clf"], F["proto"], nan_f, F["Y"])
        variants["classifier + prototypes"] = (cp.predict(E["clf"], E["proto"], nan_e), cp.threshold, t_emb + E["t_proto"], True)
        model, model_name = cp, "classifier + prototypes"
        if use_rr:
            full = Cascade(D.types, low=low, high=high)
            full.fit(F["clf"], F["proto"], F["rr"], F["Y"])
            variants["full cascade"] = (full.predict(E["clf"], E["proto"], E["rr"]), full.threshold,
                                        t_emb + E["t_proto"] + E["t_rr"], True)
            full.save(_path(cfg["paths"]["packs"]) / genre / "cascade_full.json")
            model, model_name = full, "full cascade"
        # the presence model the profiles and the pipeline use (presence.cascade.reranker decides which)
        model.save(_path(cfg["paths"]["packs"]) / genre / "cascade.json")
        Pfull, metrics[f"{genre}/presence_model"] = variants[model_name][0], model_name
        ceiling = summary(prf_table(E["Y"], D.Y(E["keys"], D.teacher).astype(float), D.types,
                                    thresholds=np.full(T, 0.5)), min_positives=min_pos)
        table = []
        for name, (P, th, latency, prob) in variants.items():
            sm = summary(prf_table(E["Y"], P, D.types, thresholds=th), min_positives=min_pos)
            table.append({"variant": name, **sm, "ece": ece(E["Y"], P) if prob else None,
                          "ms_per_section": round(1000 * latency, 1)})
            metrics.update({f"{genre}/{name.replace(' ', '_')}/{k}": v for k, v in sm.items()})
        table.append({"variant": "teacher labels (ceiling)", **ceiling, "ece": None, "ms_per_section": None})
        _write_table(_path(cfg["paths"]["tables"]) / f"presence_ablation_{genre}.csv", table)
        per_type = prf_table(E["Y"], Pfull, D.types, thresholds=model.threshold)
        _write_table(_path(cfg["paths"]["tables"]) / f"presence_cascade_{genre}.csv", per_type)
        metrics.update({f"{genre}/band_low": round(low, 4), f"{genre}/band_high": round(high, 4),
                        f"{genre}/band_share_eval": round(E["band_pairs"] / E["proto"].size, 3),
                        f"{genre}/prototype_sections_per_s": round(proto_rate, 1),
                        f"{genre}/reranker_pairs_eval": E["band_pairs"], f"{genre}/ceiling_macro_f1": ceiling["macro_f1"],
                        f"{genre}/combiner_coef": dict(zip(FEATURES, model.coef.round(3).tolist()))})
        metrics.update({f"{genre}/{k}": v for k, v in {
            "ceiling": ceiling, "model_eval": summary(per_type, min_positives=min_pos),
            "model_ms_per_section": round(1000 * variants[model_name][2], 1)}.items()})
        print(f"[cascade {genre}] band [{low:.3f}, {high:.3f}), {E['band_pairs']} band pairs in eval; ablation:", flush=True)
        for r in table:
            print(f"   {r['variant']:26} micro F1 {r['micro_f1']}  macro F1 {r['macro_f1']}  ECE {r['ece']}  "
                  f"{r['ms_per_section']} ms/section", flush=True)
    return metrics


def profile_table_path(cfg: dict[str, Any], pf: dict[str, Any], genre: str) -> Path:
    """The prevalence table of a profiles run: reports/tables for the pack's own profiles; next to the profiles for a
    run that writes them elsewhere (smoke, a Colab check), so it never overwrites the corpus table."""
    name = f"profile_prevalence_{genre}.csv"
    return _path(pf["out"].format(genre=genre)) / name if pf.get("out") else _path(cfg["paths"]["tables"]) / name


def _presence_model(cfg: dict[str, Any], genre: str, colbert: Any, splits: list[str], limit: int | None) -> dict[str, Any]:
    """The pack's lean presence model (linear heads on the stored section embeddings + ColBERT prototypes, combined
    by cascade.json) and the documents of the given splits with their sections: shared by profiles and section_probs."""
    import numpy as np

    from data.schema import Document
    from data.splits import load_manifest
    from engine.classifier import LinearHeads
    from engine.embed import EmbeddingStore, embedding_slug, text_sha1
    from engine.presence import Cascade, PrototypeScorer

    mode = cfg["presence"].get("train_labels", "teacher")
    packs = _path(cfg["paths"]["packs"]) / genre
    schema = load_schema(cfg, genre)
    types = [t["id"] for t in schema["types"]]
    heads = LinearHeads.load(packs / "classifier" / f"linear_{mode}")
    model = Cascade.load(packs / "cascade.json")
    if heads.types != types or model.types != types:
        raise ValueError(f"{genre}: classifier or cascade types differ from the schema; rerun heads and cascade")
    scorer = PrototypeScorer.from_pack(colbert, packs / "prototypes", len(types),
                                       min_sentence_chars=cfg["segment"]["min_sentence_chars"])
    split = load_manifest(_path(cfg["paths"]["splits"]) / f"{genre}.json")
    wanted = {d for d, sp in split.items() if sp in splits}
    if any(split.get(d) == "test" for d in wanted):
        raise ValueError("presence over the sealed test split is not part of the indexed corpus")
    processed = _path(cfg["paths"]["data_processed"]) / genre
    docs = [{"id": d.id, "split": split[d.id], "category": d.category, "date": d.date}
            for d in read_jsonl(processed / "documents.jsonl", Document) if d.id in wanted]
    docs = docs[: limit or None]
    keep = {d["id"] for d in docs}
    sections: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for sec in read_jsonl(processed / "sections.jsonl", Section):
        if sec.doc_id in keep:
            sections[sec.doc_id].append((sec.idx, sec.text))
    store = EmbeddingStore.load(_path(cfg["paths"]["embeddings"]) / embedding_slug(cfg["models"]["embedder"])
                                / genre / "sections")
    row_of = {h: i for i, h in enumerate(store.sha1)}

    def score(batch: list[str]) -> Any:
        idx = [row_of.get(text_sha1(t)) for t in batch]
        if any(i is None for i in idx):
            raise ValueError(f"{genre}: {sum(i is None for i in idx)} sections have no stored embedding")
        clf = heads.predict_proba(store.vectors[idx])
        proto = scorer.score(batch)
        return model.predict(clf, proto, np.full(clf.shape, np.nan))

    return {"types": types, "model": model, "docs": docs, "sections": sections, "score": score, "packs": packs,
            "scorer": scorer, "heads": heads}


def run_profiles(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import platform

    import numpy as np

    from packs.profiles import build_profiles, read_profiles

    pc = cfg["presence"]
    if (pc.get("cascade") or {}).get("reranker"):
        raise NotImplementedError("profiles with the reranker are not built in the lean tier (decision record 005)")
    colbert = load_colbert(cfg)
    pf = cfg["distil"].get("profiles") or {}
    metrics: dict[str, Any] = {"machine": f"{platform.machine()} {platform.system()}", "device": colbert.device}
    for genre in _genres(cfg):
        pm = _presence_model(cfg, genre, colbert, pf.get("splits", ["train"]), pf.get("limit"))
        types, model, docs, packs, score = pm["types"], pm["model"], pm["docs"], pm["packs"], pm["score"]
        texts = {d: [t for _, t in secs] for d, secs in pm["sections"].items()}
        texts = defaultdict(list, texts)
        out = _path(pf["out"].format(genre=genre)) if pf.get("out") else packs / "profiles"
        res = build_profiles(docs, lambda d: texts[d], score, types, model.threshold, out,
                             chunk_documents=pf.get("chunk_documents", 500),
                             log=lambda m: print(f"[profiles {genre}] {m}", flush=True))
        df = read_profiles(out)
        bits = df[[f"bit_{t}" for t in types]].to_numpy() if len(df) else np.zeros((0, len(types)))
        metrics.update({f"{genre}/{k}": v for k, v in res.items()})
        metrics.update({f"{genre}/documents": len(df), f"{genre}/types_present_per_document":
                        round(float(bits.sum(1).mean()), 2) if len(df) else None,
                        f"{genre}/types_never_present": [t for t, c in zip(types, bits.sum(0)) if c == 0]})
        prevalence = [{"type_id": t, "documents_with_type": int(c), "prevalence": round(float(c) / max(1, len(df)), 4)}
                      for t, c in zip(types, bits.sum(0))]
        if prevalence:
            _write_table(profile_table_path(cfg, pf, genre), prevalence)
    return metrics


def run_section_probs(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Every section's presence probabilities for the given splits (default train + dev) ->
    packs/<genre>/section_probs/part-*.parquet: the input of document-level presence (fitted on whole labelled dev
    documents, applied to the train split's profiles). One GPU pass; calibration is refitted on the Mac."""
    import platform

    from packs.profiles import build_section_probs

    if (cfg["presence"].get("cascade") or {}).get("reranker"):
        raise NotImplementedError("section probabilities with the reranker are not built in the lean tier")
    colbert = load_colbert(cfg)
    sc = cfg["distil"].get("section_probs") or {}
    metrics: dict[str, Any] = {"machine": f"{platform.machine()} {platform.system()}", "device": colbert.device}
    for genre in _genres(cfg):
        pm = _presence_model(cfg, genre, colbert, sc.get("splits", ["train", "dev"]), sc.get("limit"))
        out = _path(sc["out"].format(genre=genre)) if sc.get("out") else pm["packs"] / "section_probs"
        res = build_section_probs(pm["docs"], lambda d: pm["sections"][d], pm["score"], pm["types"], out,
                                  chunk_documents=sc.get("chunk_documents", 500),
                                  log=lambda m: print(f"[section_probs {genre}] {m}", flush=True))
        metrics.update({f"{genre}/{k}": v for k, v in res.items()})
        metrics[f"{genre}/documents"] = len(pm["docs"])
    return metrics


def run_doc_presence(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Document-level presence (engine/doc_presence.py): fitted on the whole labelled documents (label set
    `documents`, labels in presence.labels.documents; a document states t when some labelled section does) from every
    section's stored probabilities (packs/<genre>/section_probs). Fit / eval halves by document; the v1 rule (best
    section >= the section threshold) is scored on the same eval half. Feature sets are chosen by cross-validated log
    loss on the fit half; the chosen model is refitted on every labelled document and saved as
    packs/<genre>/doc_presence.json. With distil.doc_presence.write_profiles, profiles v2 of the indexed split are
    written to packs/<genre>/profiles (prob_ = document probability, bit_ = prob >= 0.5, best_ = the v1 best-section
    probability); the v1 profiles are moved to packs/<genre>/profiles_v1."""
    import numpy as np

    from engine.doc_presence import ALL_FEATURES, FEATURES, DocPresence, evaluate, split_documents
    from packs.profiles import read_profiles
    from packs.schema_versions import project_labels, projection

    dc = cfg["distil"].get("doc_presence") or {}
    min_pos, C_grid = dc.get("min_positives", 5), tuple(dc.get("C_grid", (0.03, 0.1, 0.3, 1.0, 3.0)))
    variants = {"logit_max": ("logit_max",), ",".join(FEATURES): FEATURES, ",".join(ALL_FEATURES): ALL_FEATURES}
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        packs = _path(cfg["paths"]["packs"]) / genre
        schema = load_schema(cfg, genre)
        types = [t["id"] for t in schema["types"]]
        proj = projection(schema)
        cols = [f"prob_{t}" for t in types]
        sp = read_profiles(packs / "section_probs")
        refs = json.loads((packs / "label_sets.json").read_text(encoding="utf-8"))["documents"]
        labelled_docs = {r["doc_id"] for r in refs}
        n_sections = defaultdict(int)
        for r in refs:
            n_sections[r["doc_id"]] += 1
        rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for line in open(_path(cfg["presence"]["labels"]["documents"].format(genre=genre)), encoding="utf-8"):
            r = json.loads(line)
            if r["doc_id"] in labelled_docs:
                rows[r["doc_id"]].append(r)
        # a document counts only when every section was labelled: a failed section could hide a statement
        complete = sorted(d for d in labelled_docs if len(rows[d]) == n_sections[d] and not any(r.get("failed") for r in rows[d]))
        by_doc = {d: g.sort_values("section_idx")[cols].to_numpy(np.float64) for d, g in sp[sp.doc_id.isin(complete)].groupby("doc_id")}
        complete = [d for d in complete if d in by_doc and len(by_doc[d]) == n_sections[d]]
        Y = np.array([[int(any(project_labels(r, proj)["labels"].get(t, 0) for r in rows[d])) for t in types]
                      for d in complete])
        fit, ev = split_documents(complete, seed=cfg["presence"].get("reference_split_seed", 1234))
        fi, ei = [complete.index(d) for d in fit], [complete.index(d) for d in ev]
        threshold = np.array(json.loads((packs / "cascade.json").read_text())["threshold"])
        best = np.stack([by_doc[d].max(0) for d in complete])
        table = [{"variant": "v1: best section >= section threshold", "cv_log_loss_fit": None, "C": None, **evaluate(Y[ei], best[ei], best[ei] >= threshold,
                                                                                min_positives=min_pos)}]
        metrics.update({f"{genre}/v1/eval/{k}": v for k, v in table[0].items()})
        chosen = None
        for name, feats in variants.items():
            m = DocPresence(feats).fit([by_doc[complete[i]] for i in fi], Y[fi], C_grid=C_grid,
                                       seed=cfg["run"].get("seed", 0))
            P = np.stack([m.predict_proba(by_doc[complete[i]]) for i in ei])
            row = {"variant": f"document model: {name}", "cv_log_loss_fit": round(m.cv_log_loss, 4), "C": m.C,
                   **evaluate(Y[ei], P, P >= 0.5, min_positives=min_pos)}
            table.append(row)
            if chosen is None or m.cv_log_loss < chosen[1]:
                chosen = (name, m.cv_log_loss, P)
        name, _, P = chosen
        sel = next(r for r in table if r["variant"] == f"document model: {name}")
        metrics.update({f"{genre}/v2/eval/{k}": v for k, v in sel.items() if k != "variant"})
        labelled_prev = Y.mean(0)
        _write_table(_path(cfg["paths"]["tables"]) / f"doc_presence_{genre}.csv", table)
        _write_table(_path(cfg["paths"]["tables"]) / f"doc_presence_types_{genre}.csv", [
            {"type_id": t, "labelled_prevalence_all": round(float(labelled_prev[k]), 3),
             "labelled_prevalence_eval": round(float(Y[ei, k].mean()), 3),
             "v1_prevalence_eval": round(float((best[ei, k] >= threshold[k]).mean()), 3),
             "v2_prevalence_eval": round(float((P[:, k] >= 0.5).mean()), 3),
             "v2_mean_probability_eval": round(float(P[:, k].mean()), 3)} for k, t in enumerate(types)])
        final = DocPresence(variants[name]).fit([by_doc[d] for d in complete], Y, C_grid=C_grid,
                                                seed=cfg["run"].get("seed", 0))
        final.save(packs / "doc_presence.json", types=types, labelled_documents=len(complete),
                   eval_half=sel, labels=cfg["presence"]["labels"]["documents"].format(genre=genre))
        metrics.update({f"{genre}/labelled_documents": len(complete), f"{genre}/fit_documents": len(fit),
                        f"{genre}/eval_documents": len(ev), f"{genre}/selected_features": name,
                        f"{genre}/labelled_types_per_document": round(float(Y.sum(1).mean()), 2)})
        print(f"[doc_presence {genre}] {len(complete)} documents; v1 eval prevalence error "
              f"{table[0]['prevalence_abs_error']}, macro F1 {table[0]['macro_f1']} -> v2 ({name}) "
              f"{sel['prevalence_abs_error']}, macro F1 {sel['macro_f1']}, ECE {sel['ece']}", flush=True)
        if dc.get("write_profiles"):
            metrics.update({f"{genre}/profiles_v2/{k}": v for k, v in
                            _write_profiles_v2(cfg, genre, packs, sp, types, final, threshold).items()})
    return metrics


def _write_profiles_v2(cfg: dict[str, Any], genre: str, packs: Path, sp: Any, types: list[str], model: Any,
                       threshold: Any) -> dict[str, Any]:
    import shutil

    import numpy as np
    import pandas as pd

    from data.schema import Document

    cols = [f"prob_{t}" for t in types]
    splits = (cfg["distil"].get("profiles") or {}).get("splits", ["train"])
    meta: dict[str, tuple[Any, Any]] = {}
    docs_path = _path(cfg["paths"]["data_processed"]) / genre / "documents.jsonl"
    if docs_path.exists():
        meta = {d.id: (d.category, d.date) for d in read_jsonl(docs_path, Document)}
    out = packs / "profiles"
    manifest = out / "manifest.json"
    if out.is_dir() and not (manifest.exists() and json.loads(manifest.read_text()).get("version") == 2):
        old = packs / "profiles_v1"
        if old.exists():
            shutil.rmtree(old)
        out.rename(old)  # v1 kept aside (also in the hub's history)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = []
    for d, g in sp[sp.split.isin(splits)].groupby("doc_id", sort=True):
        P = g.sort_values("section_idx")[cols].to_numpy(np.float64)
        prob, best = model.predict_proba(P), P.max(0)
        cat, date = meta.get(d, (None, None))
        rows.append({"doc_id": d, "split": g["split"].iloc[0], "category": cat, "date": date, "sections": len(P),
                     **{f"prob_{t}": float(v) for t, v in zip(types, prob)},
                     **{f"bit_{t}": int(v >= 0.5) for t, v in zip(types, prob)},
                     **{f"best_{t}": float(v) for t, v in zip(types, best)}})
    df = pd.DataFrame(rows)
    chunk = (cfg["distil"].get("profiles") or {}).get("chunk_documents", 500)
    for k in range(0, len(df), chunk):
        df.iloc[k:k + chunk].to_parquet(out / f"part-{k // chunk:05d}.parquet", index=False)
    bits = df[[f"bit_{t}" for t in types]].to_numpy()
    v1 = (df[[f"best_{t}" for t in types]].to_numpy() >= threshold).astype(int)
    manifest.write_text(json.dumps({"version": 2, "presence": "document model (engine/doc_presence.py)",
                                    "doc_presence": "doc_presence.json", "section_probs": "section_probs/",
                                    "bit": "prob >= 0.5", "documents": len(df)}, indent=1) + "\n")
    prevalence = [{"type_id": t, "documents_with_type": int(bits[:, k].sum()), "prevalence": round(float(bits[:, k].mean()), 4),
                   "mean_probability": round(float(df[f"prob_{t}"].mean()), 4), "v1_prevalence": round(float(v1[:, k].mean()), 4)}
                  for k, t in enumerate(types)]
    _write_table(_path(cfg["paths"]["tables"]) / f"profile_prevalence_{genre}.csv", prevalence)
    return {"documents": len(df), "types_present_per_document": round(float(bits.sum(1).mean()), 2),
            "v1_types_present_per_document": round(float(v1.sum(1).mean()), 2)}


def _line_ranges(path: Path, keep: set[str]) -> dict[str, list[tuple[int, int]]]:
    """doc_id -> contiguous byte ranges of its lines in a jsonl file whose lines start with {"doc_id": ...}."""
    import re

    head = re.compile(rb'^\{"doc_id": "((?:[^"\\]|\\.)*)"')
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    pos, cur, start = 0, None, 0
    with open(path, "rb") as fh:
        for line in fh:
            m = head.match(line)
            d = m.group(1).decode("utf-8") if m else json.loads(line)["doc_id"]
            if d != cur:
                if cur in keep:
                    out[cur].append((start, pos))
                cur, start = d, pos
            pos += len(line)
    if cur in keep:
        out[cur].append((start, pos))
    return out


def run_sentence_evidence(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Evidence sentences (plan §0.2 #14): the lean presence model (embedder + linear heads, ColBERT prototypes,
    cascade) on every sentence of the indexed documents (data/processed/<genre>/sentences.jsonl); for each document
    and type the best sentence, kept when P >= min_prob -> packs/<genre>/evidence/part-*.parquet (doc_id, type,
    section_idx, sentence_idx, p, text), resumable by chunk. analyse() quotes these instead of scoring peer
    sentences at query time (about 9 ms per sentence on the Mac). A GPU job (Colab)."""
    import platform
    import time

    import numpy as np
    import pandas as pd

    import engine.embed as E

    sc = cfg["distil"].get("sentence_evidence") or {}
    colbert = load_colbert(cfg)
    ecfg = cfg["models"]["embedder"]
    embedder = E.Embedder(ecfg["id"], devices=[colbert.device], dtype=ecfg.get("dtype", "float16"),
                          max_length=ecfg.get("max_length", 1024), batch_size=ecfg.get("batch_size", 32),
                          normalize=True, revision=ecfg.get("revision"))
    metrics: dict[str, Any] = {"machine": f"{platform.machine()} {platform.system()}", "device": colbert.device}
    for genre in _genres(cfg):
        pm = _presence_model(cfg, genre, colbert, sc.get("splits", ["train"]), sc.get("limit"))
        types, docs = pm["types"], pm["docs"]
        heads, model, scorer = pm["heads"], pm["model"], pm["scorer"]
        path = _path(cfg["paths"]["data_processed"]) / genre / "sentences.jsonl"
        ranges = _line_ranges(path, {d["id"] for d in docs})
        out = _path(sc["out"].format(genre=genre)) if sc.get("out") else pm["packs"] / "evidence"
        out.mkdir(parents=True, exist_ok=True)
        chunk, min_prob = sc.get("chunk_documents", 500), sc.get("min_prob", 0.2)
        n_parts, written, n_sent, t0 = (len(docs) + chunk - 1) // chunk, 0, 0, time.perf_counter()
        fh = open(path, "rb")
        for part in range(n_parts):
            dest = out / f"part-{part:05d}.parquet"
            if dest.exists():
                continue
            rows_in: list[dict[str, Any]] = []
            for d in docs[part * chunk:(part + 1) * chunk]:
                for a, b in ranges.get(d["id"], []):
                    fh.seek(a)
                    rows_in += [json.loads(line) for line in fh.read(b - a).splitlines() if line.strip()]
            texts = [r["text"] for r in rows_in]
            if texts:
                clf = heads.predict_proba(np.asarray(embedder.encode(texts)))
                P = model.predict(clf, scorer.score(texts), np.full(clf.shape, np.nan))
            else:
                P = np.zeros((0, len(types)))
            owner = np.array([r["doc_id"] for r in rows_in])
            rows_out = []
            for d in dict.fromkeys(owner.tolist()):
                ix = np.flatnonzero(owner == d)
                best = ix[P[ix].argmax(0)]
                for k, t in enumerate(types):
                    p = float(P[best[k], k])
                    if p >= min_prob:
                        r = rows_in[best[k]]
                        rows_out.append({"doc_id": d, "type": t, "section_idx": int(r["section_idx"]),
                                         "sentence_idx": int(r["idx"]), "p": round(p, 4), "text": r["text"]})
            tmp = dest.with_suffix(".tmp")
            pd.DataFrame(rows_out, columns=["doc_id", "type", "section_idx", "sentence_idx", "p", "text"]).to_parquet(tmp, index=False)
            tmp.rename(dest)
            written += 1
            n_sent += len(texts)
            rate = n_sent / max(1e-6, time.perf_counter() - t0)
            print(f"[sentence_evidence {genre}] part {part + 1}/{n_parts}: {len(texts)} sentences ({rate:.0f} sentences/s)",
                  flush=True)
        fh.close()
        seconds = time.perf_counter() - t0
        metrics.update({f"{genre}/parts": n_parts, f"{genre}/parts_written": written, f"{genre}/sentences_scored": n_sent,
                        f"{genre}/seconds": round(seconds, 1), f"{genre}/documents": len(docs),
                        f"{genre}/sentences_per_second": round(n_sent / seconds, 1) if n_sent and seconds else None})
    return metrics


def run_gate(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from data.splits import load_manifest
    from packs.profiles import read_profiles

    pc, g = cfg["presence"], cfg["presence"]["gate"]
    model_name = "full cascade" if (pc.get("cascade") or {}).get("reranker") else "classifier + prototypes"
    prof_run = (cfg["distil"].get("gate") or {}).get("profiles_run")
    prof = json.loads((_path(prof_run) / "metrics.json").read_text(encoding="utf-8"))["metrics"] if prof_run else {}
    rows, metrics = [], {"presence_model": model_name}
    for genre in _genres(cfg):
        table = {r["variant"]: r for r in csv.DictReader(open(_path(cfg["paths"]["tables"]) / f"presence_ablation_{genre}.csv",
                                                              encoding="utf-8"))}
        model, ceiling = float(table[model_name]["macro_f1"]), float(table["teacher labels (ceiling)"]["macro_f1"])
        ratio = model / ceiling if ceiling else 0.0
        split = load_manifest(_path(cfg["paths"]["splits"]) / f"{genre}.json")
        indexable = sum(1 for s in split.values() if s in (cfg["distil"].get("profiles") or {}).get("splits", ["train"]))
        df = read_profiles(_path(cfg["paths"]["packs"]) / genre / "profiles")
        row = {"genre": genre, "presence_model": model_name, "macro_f1": model, "teacher_ceiling": ceiling,
               "share_of_ceiling": round(ratio, 3), "required_share": g["relative_to_ceiling"],
               "f1_gate": "pass" if ratio >= g["relative_to_ceiling"] else "not met",
               "profiled_documents": len(df), "indexable_documents": indexable,
               "profiles_complete": len(df) == indexable,
               "sections_per_second": prof.get(f"{genre}/sections_per_second"), "machine": prof.get("machine"),
               "device": prof.get("device"), "reference": "silver (decision record 004)"}
        rows.append(row)
        metrics.update({f"{genre}/{k}": v for k, v in row.items() if k != "genre"})
        print(f"[gate {genre}] {row}", flush=True)
    _write_table(_path(cfg["paths"]["tables"]) / "presence_gate.csv", rows)
    return metrics


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    stage = cfg["distil"].get("stage")
    if stage == "labels":
        return run_labels(cfg, run_dir)
    if stage == "select":
        return run_select(cfg, run_dir)
    if stage == "select_documents":
        return run_select_documents(cfg, run_dir)
    if stage == "teacher_labels":
        return run_teacher_labels(cfg, run_dir)
    if stage == "score":
        return run_score(cfg, run_dir)
    if stage == "heads":
        return run_heads(cfg, run_dir)
    if stage == "prototypes":
        return run_prototypes(cfg, run_dir)
    if stage == "cascade":
        return run_cascade(cfg, run_dir)
    if stage == "profiles":
        return run_profiles(cfg, run_dir)
    if stage == "section_probs":
        return run_section_probs(cfg, run_dir)
    if stage == "sentence_evidence":
        return run_sentence_evidence(cfg, run_dir)
    if stage == "doc_presence":
        return run_doc_presence(cfg, run_dir)
    if stage == "gate":
        return run_gate(cfg, run_dir)
    raise NotImplementedError(f"distil stage {stage!r} is implemented later in Phase 4")
