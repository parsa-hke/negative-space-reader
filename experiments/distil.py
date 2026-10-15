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
  teacher_labels (4.1) the two-pass teacher (prompts label_presence + label_presence_check) over the label sets;
           every type of the schema is labelled; writes teacher_labels_<genre>.jsonl (source teacher:<route>)
  score    (4.2) teacher labels vs a reference (OPP-115 labels, or the human set read back from the page), per type,
           final and first pass only -> reports/tables/teacher_vs_<name>_<genre>.csv
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


def load_reference(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Reference labels as {key: {present, unsure}}: a labels-stage file (rows with labels {type: 0/1}) or the
    labelling page's answers (rows with present/unsure, read back from its database as JSON lines)."""
    ref = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["doc_id"], r.get("section_idx", r.get("idx")))
        if "labels" in r:
            ref[key] = {"present": [t for t, v in r["labels"].items() if v], "unsure": [],
                        "types": list(r["labels"])}
        else:
            ref[key] = {"present": r.get("present") or [], "unsure": r.get("unsure") or [], "types": None}
    return ref


def run_score(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Teacher labels vs a reference (OPP-115 labels or the human set), per type, final and first pass only; types =
    those the reference labels (a labels-stage file) or every schema type (page answers). Writes
    reports/tables/teacher_vs_<name>_<genre>.csv and macro/micro numbers as metrics."""
    from packs.presence_labels import agreement

    sc = cfg["distil"]["score"]
    metrics: dict[str, Any] = {}
    for genre in _genres(cfg):
        teacher = {}
        for line in open(_path(sc["teacher"].format(genre=genre)), encoding="utf-8"):
            r = json.loads(line)
            if not r["failed"]:
                teacher[(r["doc_id"], r["section_idx"])] = r
        ref = load_reference(_path(sc["reference"].format(genre=genre)))
        typed = next(iter(ref.values()))["types"]
        types = typed or [t["id"] for t in load_schema(cfg, genre)["types"]]
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


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    stage = cfg["distil"].get("stage")
    if stage == "labels":
        return run_labels(cfg, run_dir)
    if stage == "select":
        return run_select(cfg, run_dir)
    if stage == "teacher_labels":
        return run_teacher_labels(cfg, run_dir)
    if stage == "score":
        return run_score(cfg, run_dir)
    raise NotImplementedError(f"distil stage {stage!r} is implemented later in Phase 4")
