"""Section-level presence labels for a pack's frozen schema (plan §4.1).

Every type carries a `label_source`: a list of alternatives, each either `teacher` (labelled by the teacher model)
or a rule over a human annotation file with rows {doc_id, section_idx, category, attributes: {name: value},
annotation_id}:

    Category                                   any annotation of that category
    Category :: Attr = Value & Attr2 = Value2  one annotation of that category with every listed attribute value

A section is a positive for a type when one of its annotations matches one alternative. Every section of an annotated
document is labelled (annotators read the whole document, so a section with no matching annotation is a negative).
Names must be the annotation file's exact names: an unknown category, attribute or value is refused rather than
producing a silent zero. Nothing here names a genre; the annotation source is configured per pack.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from data.schema import Section

TEACHER = "teacher"


class LabelSourceError(ValueError):
    pass


def parse_label_source(entry: str) -> tuple[str, list[tuple[str, str]]]:
    """'Category :: A = v & B = w' -> ('Category', [('A', 'v'), ('B', 'w')]); 'Category' -> ('Category', [])."""
    category, _, rest = (x.strip() for x in entry.partition("::"))
    conditions = []
    for part in [p.strip() for p in rest.split("&")] if rest else []:
        name, eq, value = (x.strip() for x in part.partition("="))
        if not eq or not name or not value:
            raise LabelSourceError(f"cannot parse condition {part!r} in {entry!r}")
        conditions.append((name, value))
    if not category:
        raise LabelSourceError(f"no category in {entry!r}")
    return category, conditions


def _sources(t: dict[str, Any]) -> list[str]:
    ls = t.get("label_source") or []
    return [ls] if isinstance(ls, str) else list(ls)


def teacher_types(types: list[dict[str, Any]]) -> list[str]:
    """Ids of the types that need teacher labels (no rule over human annotations)."""
    return [t["id"] for t in types if any(s.strip() == TEACHER for s in _sources(t))]


def _rules(types: list[dict[str, Any]]) -> dict[str, list[tuple[str, list[tuple[str, str]]]]]:
    return {t["id"]: [parse_label_source(s) for s in _sources(t) if s.strip() != TEACHER]
            for t in types if t["id"] not in teacher_types(types)}


def _check_names(rules: dict[str, list], annotations: list[dict[str, Any]]) -> None:
    values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for a in annotations:
        cat = values[a["category"]]
        for name, value in (a.get("attributes") or {}).items():
            cat[name].add(value)
    problems = []
    for tid, alternatives in rules.items():
        for category, conditions in alternatives:
            if category not in values:
                problems.append(f"{tid}: category {category!r}")
                continue
            for name, value in conditions:
                if value not in values[category].get(name, set()):
                    problems.append(f"{tid}: {category} :: {name} = {value!r}")
    if problems:
        raise LabelSourceError("label_source names not in the annotations: " + "; ".join(problems))


def _matches(a: dict[str, Any], category: str, conditions: list[tuple[str, str]]) -> bool:
    attrs = a.get("attributes") or {}
    return a["category"] == category and all(attrs.get(n) == v for n, v in conditions)


def annotation_labels(types: list[dict[str, Any]], annotations: Iterable[dict[str, Any]],
                      sections: dict[str, list[Section]], *, source: str
                      ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows {doc_id, section_idx, labels: {type: 0/1}, evidence: {type: [annotation ids]}, source} for every section
    of every annotated document, over the types whose label_source is a rule (teacher types are left out)."""
    annotations = list(annotations)
    rules = _rules(types)
    _check_names(rules, annotations)
    by_section: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for a in annotations:
        by_section[(a["doc_id"], a["section_idx"])].append(a)
    docs = sorted({a["doc_id"] for a in annotations})
    missing = [d for d in docs if d not in sections]
    if missing:
        raise LabelSourceError(f"{len(missing)} annotated documents have no sections, e.g. {missing[:3]}")
    rows, positives, unannotated = [], {tid: 0 for tid in rules}, 0
    for d in docs:
        for s in sections[d]:
            anns = by_section.get((d, s.idx), [])
            unannotated += not anns
            labels, evidence = {}, {}
            for tid, alternatives in rules.items():
                hits = sorted({a["annotation_id"] for a in anns
                               for category, conditions in alternatives if _matches(a, category, conditions)})
                labels[tid] = int(bool(hits))
                positives[tid] += labels[tid]
                if hits:
                    evidence[tid] = hits
            rows.append({"doc_id": d, "section_idx": s.idx, "labels": labels, "evidence": evidence, "source": source})
    known = {(d, s.idx) for d in docs for s in sections[d]}
    stray = sum(1 for key in by_section if key not in known)
    if stray:
        raise LabelSourceError(f"{stray} annotated sections are not in the corpus (section indices out of line)")
    return rows, {"sections": len(rows), "documents": len(docs), "sections_without_annotation": unannotated,
                  "positives": positives}


# ------------------------------------------------------------------------------------ teacher labels (plan §4.1)
LABEL_PROMPT = "label_presence"
CHECK_PROMPT = "label_presence_check"


def first_pass_model(type_ids: list[str]) -> type:
    """{"statements": [{claim, type: schema id or "none", span}]}, the type constrained to the schema."""
    from typing import Literal

    from pydantic import create_model

    TypeId = Literal[tuple(list(type_ids) + ["none"])]  # type: ignore[misc]
    Item = create_model("LabelStatement", claim=(str, ...), type=(TypeId, ...), span=(str, ...))
    return create_model("SectionLabels", statements=(list[Item], ...))  # type: ignore[valid-type]


def check_model(type_ids: list[str]) -> type:
    """{"present": [{type, span}]}; ids outside the candidates shown are dropped after validation."""
    from typing import Literal

    from pydantic import create_model

    TypeId = Literal[tuple(type_ids)]  # type: ignore[misc]
    Item = create_model("CheckedType", type=(TypeId, ...), span=(str, ...))
    return create_model("SecondLook", present=(list[Item], ...))  # type: ignore[valid-type]


def render_label_prompt(section: Section, genre: dict[str, Any], types: list[dict[str, Any]], *,
                        max_assertions: int) -> str:
    from models.teacher import PROMPTS

    return PROMPTS.get(LABEL_PROMPT).render(
        document_kind=genre["document_kind"], author_role=genre["author_role"],
        type_catalogue="\n".join(f"{t['id']}: {t['definition']}" for t in types),
        section_heading=section.heading or "(no heading)", section_text=section.text, max_assertions=max_assertions)


def render_check_prompt(section: Section, genre: dict[str, Any], candidates: list[dict[str, Any]],
                        found: list[dict[str, Any]] | None = None) -> str:
    """found: the types the first pass gave the section (shown from label_presence_check v2 on)."""
    from models.teacher import PROMPTS
    from packs.schema_induction import type_card

    return PROMPTS.get(CHECK_PROMPT).render(
        document_kind=genre["document_kind"], author_role=genre["author_role"],
        section_heading=section.heading or "(no heading)", section_text=section.text,
        candidates="\n".join(f"- {type_card(t)}" for t in candidates),
        found="\n".join(f"- {t['id']}: {t['definition']}" for t in found or []) or "- (none)")


def select_candidates(claim_vecs: "np.ndarray", section_vec: "np.ndarray | None", type_vecs: "np.ndarray", *,
                      found: set[int], k_claim: int, k_section: int, cap: int) -> list[int]:
    """Second-look candidates for one section (type indices): for each statement in order, its k_claim most similar
    types that the first pass did not find; then the section's k_section most similar such types; no repeats; at
    most cap. Callers put the statements mapped to "none" first."""
    import numpy as np

    out: list[int] = []

    def take(sims: "np.ndarray", k: int) -> None:
        picked = 0
        for j in np.argsort(-sims, kind="stable"):
            if picked >= k:
                break
            if int(j) in found:
                continue
            picked += 1
            if int(j) not in out:
                out.append(int(j))

    for v in claim_vecs:
        take(type_vecs @ v, k_claim)
    if section_vec is not None:
        take(type_vecs @ section_vec, k_section)
    return out[:cap]


def _unit(X: "np.ndarray") -> "np.ndarray":
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def teacher_labels(teacher: Any, embedder: Any, sections: list[Section], genre: dict[str, Any],
                   types: list[dict[str, Any]], *, label_types: list[str], max_assertions: int, max_tokens: int,
                   check_max_tokens: int, k_claim: int, k_section: int, cap: int, batch_size: int = 8,
                   section_vectors: dict[tuple[str, int], Any] | None = None) -> list[dict[str, Any]]:
    """Two-pass teacher labels for each section over label_types (the prompt always shows the whole schema, so a
    statement is never forced into a wrong type because its own type was left out).

    Pass 1 (label_presence): statements -> schema id or "none", with spans. Pass 2 (label_presence_check): the types
    most similar to the section's statements ("none" ones first) and to the section itself, not found in pass 1,
    shown with definitions and examples; the teacher says which are stated. Each row records labels {type: 0/1},
    evidence {type: span}, grounded {type: span occurs in the section}, found_by {type: first | second_look},
    the candidates shown, the statements, and failures (failed: pass 1 failed, no labels; check_failed: pass 1 only).
    """
    import numpy as np

    from packs.schema_induction import _norm, _structured_in_batches, type_card

    ids = [t["id"] for t in types]
    wanted = set(label_types)
    unknown = wanted - set(ids)
    if unknown:
        raise LabelSourceError(f"label types not in the schema: {sorted(unknown)}")
    first = _structured_in_batches(teacher, [render_label_prompt(s, genre, types, max_assertions=max_assertions)
                                             for s in sections], first_pass_model(ids), max_tokens=max_tokens,
                                   batch_size=batch_size, skip_failures=True)
    T = _unit(embedder.encode([type_card(t) for t in types]))
    statements = [[] if r is None else sorted(r.statements, key=lambda st: st.type != "none") for r in first]
    flat = [st.claim for sts in statements for st in sts]
    C = _unit(embedder.encode(flat)) if flat else np.zeros((0, T.shape[1]), dtype=np.float32)
    vectors = section_vectors or {}
    need = [s for s in sections if (s.doc_id, s.idx) not in vectors]
    if need:
        vectors = {**vectors, **dict(zip([(s.doc_id, s.idx) for s in need], _unit(embedder.encode([s.text for s in need]))))}
    rows, checks, pos = [], [], 0
    for s, r, sts in zip(sections, first, statements):
        row: dict[str, Any] = {"doc_id": s.doc_id, "section_idx": s.idx, "labels": {}, "evidence": {}, "grounded": {},
                               "found_by": {}, "candidates": [], "statements": [], "failed": r is None,
                               "check_failed": False}
        rows.append(row)
        claim_vecs, pos = C[pos:pos + len(sts)], pos + len(sts)
        if r is None:
            continue
        text = _norm(s.text)
        row["statements"] = [st.model_dump() for st in sts]
        for st in sts:
            if st.type != "none" and st.type not in row["found_by"]:
                row["found_by"][st.type] = "first"
                row["evidence"][st.type] = st.span
            if st.type != "none" and not row["grounded"].get(st.type) and _norm(st.span) in text:
                row["grounded"][st.type], row["evidence"][st.type] = True, st.span
        for tid in row["found_by"]:
            row["grounded"].setdefault(tid, False)
        found = {ids.index(t) for t in row["found_by"]}
        cand = select_candidates(claim_vecs, _unit(vectors[(s.doc_id, s.idx)]), T, found=found, k_claim=k_claim,
                                 k_section=k_section, cap=cap)
        row["candidates"] = [ids[j] for j in cand]
        if cand:
            checks.append((row, s, [types[j] for j in cand], [types[j] for j in sorted(found)]))
    answers = _structured_in_batches(teacher, [render_check_prompt(s, genre, c, f) for _, s, c, f in checks],
                                     check_model(ids), max_tokens=check_max_tokens, batch_size=batch_size,
                                     skip_failures=True)
    for (row, s, cand, _), ans in zip(checks, answers):
        if ans is None:
            row["check_failed"] = True
            continue
        shown, text = {t["id"] for t in cand}, _norm(s.text)
        for item in ans.present:
            if item.type in shown and item.type not in row["found_by"]:
                row["found_by"][item.type] = "second_look"
                row["evidence"][item.type] = item.span
                row["grounded"][item.type] = _norm(item.span) in text
    for row in rows:
        if row["failed"]:
            continue
        row["labels"] = {t: int(t in row["found_by"]) for t in label_types}
        for key in ("found_by", "evidence", "grounded"):
            row[key] = {t: v for t, v in row[key].items() if t in wanted}
    return rows


def sample_sections(pool: dict[str, list[Section]], *, n: int, seed: int, min_chars: int, per_doc: int,
                    length_bins: tuple[int, ...] = (600, 2000), position_bins: int = 5) -> list[Section]:
    """n sections stratified by relative position in the document (position_bins equal bins) and length (bins cut
    at length_bins), allocated in proportion to the pool, at most per_doc per document; seeded and order-free.
    pool: every section of each candidate document (positions are relative to the whole document)."""
    import bisect
    import random

    rng = random.Random(seed)
    strata: dict[tuple[int, int], list[Section]] = defaultdict(list)
    for d in sorted(pool):
        secs = sorted(pool[d], key=lambda s: s.idx)
        for k, s in enumerate(secs):
            if len(s.text) < min_chars:
                continue
            rel = k / max(1, len(secs) - 1)
            strata[(min(int(rel * position_bins), position_bins - 1), bisect.bisect_right(length_bins, len(s.text)))
                   ].append(s)
    total = sum(len(v) for v in strata.values())
    if total == 0:
        return []
    n = min(n, total)
    quota = {k: n * len(v) / total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    used: dict[str, int] = defaultdict(int)
    chosen: list[Section] = []
    keys = sorted(strata, key=lambda k: (len(strata[k]), k))  # small strata pick first
    for k in keys:
        items = strata[k][:]
        rng.shuffle(items)
        got = 0
        for s in items:
            if got >= alloc[k]:
                break
            if used[s.doc_id] < per_doc:
                chosen.append(s)
                used[s.doc_id] += 1
                got += 1
    if len(chosen) < n:  # a stratum ran out of documents: fill from the rest of the pool
        rest = [s for k in keys for s in strata[k] if s not in chosen]
        rng.shuffle(rest)
        for s in rest:
            if len(chosen) >= n:
                break
            if used[s.doc_id] < per_doc:
                chosen.append(s)
                used[s.doc_id] += 1
    return sorted(chosen, key=lambda s: (s.doc_id, s.idx))


def agreement(human: dict[tuple[str, int], dict[str, Any]], teacher: dict[tuple[str, int], dict[str, Any]],
              type_ids: list[str]) -> list[dict[str, Any]]:
    """Per-type agreement of teacher labels with human labels (plan §4.2) on the sections both labelled.
    human[key] = {present: [...], unsure: [...]}; teacher[key] = a teacher_labels row. A section the human marked
    unsure for a type is left out for that type. Counts are given for the final labels and for the first pass alone
    (labels found by the second look removed), which measures what the second look adds and what it costs."""
    def prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f = 2 * p * r / (p + r) if p and r else (0.0 if (tp + fp) and (tp + fn) else None)
        return p, r, f

    keys = [k for k in human if k in teacher and teacher[k].get("labels")]
    out = []
    for tid in type_ids:
        c = dict(tp=0, fp=0, fn=0, tp_first_pass=0, fp_first_pass=0, fn_first_pass=0)
        n = unsure = pos = 0
        for k in keys:
            h, t = human[k], teacher[k]
            if tid in (h.get("unsure") or []):
                unsure += 1
                continue
            n += 1
            gold = tid in (h.get("present") or [])
            pos += gold
            final = bool(t["labels"].get(tid))
            first = final and t.get("found_by", {}).get(tid) == "first"
            for suffix, pred in (("", final), ("_first_pass", first)):
                if pred and gold:
                    c["tp" + suffix] += 1
                elif pred:
                    c["fp" + suffix] += 1
                elif gold:
                    c["fn" + suffix] += 1
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        p1, r1, f1 = prf(c["tp_first_pass"], c["fp_first_pass"], c["fn_first_pass"])
        out.append({"type_id": tid, "sections": n, "human_positives": pos, "unsure": unsure, **c,
                    "precision": p, "recall": r, "f1": f,
                    "precision_first_pass": p1, "recall_first_pass": r1, "f1_first_pass": f1})
    return out


def sample_documents(docs: list[dict[str, Any]], *, n: int, seed: int) -> list[str]:
    """n whole documents stratified by year and category (proportional allocation, largest remainder), seeded and
    independent of input order. docs: {"id", "date", "category"}. Returns document ids, sorted."""
    import random

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for d in sorted(docs, key=lambda d: d["id"]):
        strata[((d.get("date") or "")[:4], d.get("category") or "")].append(d["id"])
    total = sum(len(v) for v in strata.values())
    n = min(n, total)
    keys = sorted(strata)
    quota = {k: n * len(strata[k]) / total for k in keys}
    take = {k: int(quota[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(quota[k] - take[k]), k))[: n - sum(take.values())]:
        take[k] += 1
    rng = random.Random(seed)
    return sorted(i for k in keys for i in rng.sample(strata[k], take[k]))
