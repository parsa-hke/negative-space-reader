"""Schema induction: teacher extraction -> clustering -> naming -> freeze (plan Phase 3).

Step 1 (plan §3.1–3.2): the extraction output model, prompt rendering, the batched extraction call, the eye-test
section selection, and automatic faithfulness checks that support (never replace) the by-eye review.
Step 2 (plan §3.3): dimension reduction and an HDBSCAN sweep over claim embeddings, cluster summaries (central
claims, extraction labels, typical headings and position), the naming prompt and call, example-label leakage.
Genre knowledge enters only as data: `genre_info.<genre>` in the config fills the prompt's placeholders.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, Field

from data.schema import Section
from data.segment import split_sentences

EXTRACT_PROMPT = "induce_extract"
NAME_PROMPT = "induce_name"


# ------------------------------------------------------------------------------------ output model
class ExtractedAssertion(BaseModel):
    claim: str = Field(description="normalised one-sentence paraphrase of one thing the section asserts")
    type_guess: str = Field(description="short snake_case label for the kind of assertion")
    span: str = Field(description="verbatim quote from the section that supports the claim")


class SectionExtraction(BaseModel):
    assertions: list[ExtractedAssertion]


# ------------------------------------------------------------------------------------ prompt
def render_extract_prompt(section: Section, genre: dict[str, Any], *, max_assertions: int,
                          prompt_id: str = EXTRACT_PROMPT) -> str:
    from models.teacher import PROMPTS

    return PROMPTS.get(prompt_id).render(
        document_kind=genre["document_kind"], author_role=genre["author_role"],
        section_heading=section.heading or "(no heading)", section_text=section.text,
        max_assertions=max_assertions)


def extract(teacher: Any, sections: list[Section], genre: dict[str, Any], *, max_assertions: int,
            max_tokens: int, batch_size: int = 8, skip_failures: bool = False,
            on_batch: Callable[[int, list[SectionExtraction | None]], None] | None = None) -> list[SectionExtraction | None]:
    """One SectionExtraction per section (order preserved). Extra assertions beyond max_assertions are cut.
    With skip_failures, a batch that fails validation is retried item by item and items that still fail become
    None (counted by the caller) instead of aborting a long run. on_batch(start, results) is called after each
    batch, so results can be streamed to disk."""
    def cut(i: int, results: list[SectionExtraction | None]) -> None:
        results[:] = [SectionExtraction(assertions=r.assertions[:max_assertions]) if r is not None else None
                      for r in results]
        if on_batch is not None:
            on_batch(i, results)

    prompts = [render_extract_prompt(s, genre, max_assertions=max_assertions) for s in sections]
    return _structured_in_batches(teacher, prompts, SectionExtraction, max_tokens=max_tokens, batch_size=batch_size,
                                  skip_failures=skip_failures, on_batch=cut)


def _structured_in_batches(teacher: Any, prompts: list[str], output_model: type[BaseModel], *, max_tokens: int,
                           batch_size: int, skip_failures: bool,
                           on_batch: Callable[[int, list[Any]], None] | None = None) -> list[Any]:
    """Structured teacher calls in batches. With skip_failures, a batch that fails validation is retried item by
    item and items that still fail become None. on_batch(start, results) may edit results in place."""
    from models.teacher import StructuredOutputError

    out: list[Any] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        try:
            results: list[Any] = list(teacher.generate_structured_batch(chunk, output_model, max_tokens=max_tokens))
        except StructuredOutputError:
            if not skip_failures:
                raise
            results = []
            for p in chunk:
                try:
                    results.append(teacher.generate_structured(p, output_model, max_tokens=max_tokens))
                except StructuredOutputError:
                    results.append(None)
        if on_batch is not None:
            on_batch(i, results)
        out += results
    return out


# ------------------------------------------------------------------------------------ eye-test selection
@dataclass
class SectionRef:
    genre: str
    doc_id: str
    section_idx: int
    text_sha1: str
    position: str  # first | middle | last third of the document


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def select_eye_test(sections: Iterable[Section], dev_doc_ids: set[str], genre: str, *, n: int, seed: int,
                    min_chars: int, max_chars: int) -> list[SectionRef]:
    """n sections from n different dev documents, balanced over the first / middle / last third of the document,
    within [min_chars, max_chars]. Deterministic given the seed."""
    by_doc: dict[str, list[Section]] = {}
    for s in sections:
        if s.doc_id in dev_doc_ids:
            by_doc.setdefault(s.doc_id, []).append(s)
    pools: dict[str, list[tuple[str, Section]]] = {"first": [], "middle": [], "last": []}
    for doc_id, secs in sorted(by_doc.items()):
        k = len(secs)
        for s in secs:
            if min_chars <= len(s.text) <= max_chars:
                pos = "first" if s.idx < k / 3 else "middle" if s.idx < 2 * k / 3 else "last"
                pools[pos].append((doc_id, s))
    rng = random.Random(seed)
    for p in pools.values():
        rng.shuffle(p)
    chosen: list[SectionRef] = []
    used: set[str] = set()
    order = ["first", "middle", "last"]
    i = 0
    while len(chosen) < n and any(pools.values()):
        pos = order[i % 3]
        i += 1
        while pools[pos]:
            doc_id, s = pools[pos].pop()
            if doc_id not in used:
                used.add(doc_id)
                chosen.append(SectionRef(genre, doc_id, s.idx, _sha1(s.text), pos))
                break
    return chosen


# ------------------------------------------------------------------------------------ automatic checks
_SNAKE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+){0,5}$")
_WS = re.compile(r"\s+")
_FIRST_PERSON = re.compile(r"\b(we|our|us|i|my)\b", re.I)


def _norm(text: str) -> str:
    return _WS.sub(" ", text.replace("“", '"').replace("”", '"').replace("’", "'")).strip().lower()


def extraction_checks(section_text: str, extraction: SectionExtraction) -> dict[str, Any]:
    """Per-section diagnostics: is every span a verbatim quote, are type guesses snake_case, are claims normalised
    (no first person), duplicates, and how much of the section the spans cover (an under-extraction signal)."""
    text = _norm(section_text)
    a = extraction.assertions
    grounded = [bool(x.span.strip()) and _norm(x.span) in text for x in a]
    sentences = [_norm(s) for s in split_sentences(section_text)]
    spans = [_norm(x.span) for x, g in zip(a, grounded) if g]
    covered = [any(sp and (sp in s or s in sp) for sp in spans) for s in sentences]
    claims = [_norm(x.claim) for x in a]
    return {
        "n_assertions": len(a),
        "grounded_spans": sum(grounded),
        "snake_case_types": sum(bool(_SNAKE.match(x.type_guess)) for x in a),
        "first_person_claims": sum(bool(_FIRST_PERSON.search(x.claim)) for x in a),
        "duplicate_claims": len(claims) - len(set(claims)),
        "sentences": len(sentences),
        "sentences_covered": sum(covered),
        "grounded": grounded,
    }


def summarise_checks(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = sum(r["n_assertions"] for r in rows)
    s = sum(r["sentences"] for r in rows)
    return {
        "sections": len(rows),
        "assertions_per_section": round(n / max(1, len(rows)), 2),
        "empty_sections": sum(r["n_assertions"] == 0 for r in rows),
        "span_grounding_rate": round(sum(r["grounded_spans"] for r in rows) / max(1, n), 3),
        "snake_case_rate": round(sum(r["snake_case_types"] for r in rows) / max(1, n), 3),
        "first_person_claim_rate": round(sum(r["first_person_claims"] for r in rows) / max(1, n), 3),
        "duplicate_claims": sum(r["duplicate_claims"] for r in rows),
        "sentence_coverage": round(sum(r["sentences_covered"] for r in rows) / max(1, s), 3),
    }


def write_refs(path: str | Path, refs: list[SectionRef]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([r.__dict__ for r in refs], indent=1) + "\n", encoding="utf-8")


def read_refs(path: str | Path) -> list[SectionRef]:
    return [SectionRef(**r) for r in json.loads(Path(path).read_text(encoding="utf-8"))]


# ==================================================================================== step 2: cluster and name (§3.3)
def load_claims(path: str | Path, *, exclude_docs: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Claims written by the extract stage, minus whole documents listed in exclude_docs."""
    excluded = set(exclude_docs)
    with open(path, encoding="utf-8") as f:
        return [r for r in (json.loads(line) for line in f) if r["doc_id"] not in excluded]


def align_sections(old: list[Section], new: list[Section]) -> dict[int, list[int]]:
    """One document before and after a rebuild that only split sections: old section idx -> the new section indices
    that hold its text, in order. Raises ValueError when the texts do not line up (the rebuild changed content)."""
    out: dict[int, list[int]] = {}
    j = 0
    for s in old:
        target, got, chunks = _norm(s.text), "", []
        while j < len(new) and len(got) < len(target):
            got = _norm(f"{got} {new[j].text}")
            chunks.append(new[j].idx)
            j += 1
        if got != target:
            raise ValueError(f"{s.doc_id}: section {s.idx} does not match the rebuilt sections {chunks}")
        out[s.idx] = chunks
    if j != len(new):
        raise ValueError(f"{old[0].doc_id if old else '?'}: {len(new) - j} rebuilt sections left over")
    return out


def relink_claims(claims: list[dict[str, Any]], old_sections: dict[str, list[Section]],
                  new_sections: dict[str, list[Section]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Point each claim at the section that holds it after long sections were split. Claim ids are kept; each row
    gains section_idx_unsplit (the old index) and relink, how the new index was found:
      same_section        the old section was not split (its index may still shift)
      span                the claim's span occurs in one chunk of the split section (first chunk if several)
      span_across_chunks  the span runs over a chunk boundary: the chunk where it starts
      overlap             the span is not in the section verbatim: the chunk sharing most of its words
      first_chunk         no word in common: the first chunk of the old section."""
    maps = {d: align_sections(old_sections[d], new_sections[d]) for d in {c["doc_id"] for c in claims}}
    out, counts = [], {}
    for c in claims:
        chunks = maps[c["doc_id"]][c["section_idx"]]
        texts = [_norm(new_sections[c["doc_id"]][i].text) for i in chunks]
        span = _norm(c.get("span") or "")
        if len(chunks) == 1:
            idx, how = chunks[0], "same_section"
        elif span and (hits := [i for i, t in zip(chunks, texts) if span in t]):
            idx, how = hits[0], "span"
        elif span and (pos := " ".join(texts).find(span)) >= 0:
            ends = [sum(len(t) + 1 for t in texts[: k + 1]) for k in range(len(texts))]
            idx, how = chunks[next(k for k, e in enumerate(ends) if pos < e)], "span_across_chunks"
        else:
            words = set(re.findall(r"\w+", span))
            shared = [len(words & set(re.findall(r"\w+", t))) for t in texts]
            best = max(range(len(chunks)), key=lambda k: (shared[k], -k))
            idx, how = (chunks[best], "overlap") if shared[best] else (chunks[0], "first_chunk")
        out.append({**c, "section_idx": idx, "section_idx_unsplit": c["section_idx"], "relink": how})
        counts[how] = counts.get(how, 0) + 1
    return out, counts


def center_by_document(X: "np.ndarray", doc_ids: list[str], *, alpha: float) -> "np.ndarray":
    """Subtract alpha times each document's mean claim vector from its claims, then re-normalise. What every claim
    of a document shares is its subject matter; the kind of statement varies within the document. Removing the
    shared part keeps topic from deciding the clusters (papers: without it, clusters were research topics)."""
    import numpy as np

    if not alpha:
        return X
    _, inv = np.unique(np.asarray(doc_ids), return_inverse=True)
    means = np.zeros((inv.max() + 1, X.shape[1]), dtype=np.float64)
    np.add.at(means, inv, X)
    means /= np.bincount(inv)[:, None]
    Y = (X - alpha * means[inv]).astype(np.float32)
    return Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)


def reduce_dims(X: "np.ndarray", *, method: str | None, dims: int, seed: int, n_neighbors: int = 15,
                min_dist: float = 0.0) -> "np.ndarray":
    """Rows of X (unit vectors) -> `dims` dimensions, re-normalised so that euclidean distance follows cosine.
    HDBSCAN on raw 1024-d vectors labels most points noise (density is flat in high dimensions)."""
    import numpy as np

    if method in (None, "none") or dims >= X.shape[1]:
        return X
    if method == "pca":
        from sklearn.decomposition import PCA

        Z = PCA(n_components=dims, random_state=seed).fit_transform(X)
    elif method == "umap":
        import umap

        Z = umap.UMAP(n_components=dims, metric="cosine", n_neighbors=n_neighbors, min_dist=min_dist,
                      random_state=seed).fit_transform(X)
    else:
        raise ValueError(f"unknown reduction {method!r} (pca | umap | none)")
    Z = np.asarray(Z, dtype=np.float32)
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def hdbscan_labels(Z: "np.ndarray", *, min_cluster_size: int, min_samples: int | None, selection: str) -> "np.ndarray":
    import hdbscan

    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                           cluster_selection_method=selection, core_dist_n_jobs=-1).fit_predict(Z)


def tune_clusters(Z: "np.ndarray", grid: list[dict[str, Any]], *, target: tuple[int, int] | list[int],
                  max_cluster_share: float) -> tuple["np.ndarray", list[dict[str, Any]], int]:
    """Run HDBSCAN for every setting in grid. A setting is eligible when its cluster count is inside target and no
    cluster holds more than max_cluster_share of the claims; the eligible setting with the least noise wins. With
    none eligible, the setting closest to the target range wins (reported, not hidden). Returns (labels of the
    chosen setting, one row per setting, index of the chosen row)."""
    import time

    import numpy as np

    lo, hi = target
    rows, best, best_key, best_labels = [], -1, None, None
    for i, params in enumerate(grid):
        t0 = time.perf_counter()
        labels = hdbscan_labels(Z, min_cluster_size=params["min_cluster_size"], min_samples=params.get("min_samples"),
                                selection=params.get("selection", "eom"))
        sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.zeros(0, dtype=int)
        n = int(len(sizes))
        noise = float((labels < 0).mean())
        largest = float(sizes.max() / len(labels)) if n else 0.0
        eligible = lo <= n <= hi and largest <= max_cluster_share
        rows.append({**params, "n_clusters": n, "noise_rate": round(noise, 4), "largest_share": round(largest, 4),
                     "eligible": eligible, "seconds": round(time.perf_counter() - t0, 1)})
        key = (0, noise) if eligible else (1, max(lo - n, n - hi, 0), noise)
        if best_key is None or key < best_key:
            best, best_key, best_labels = i, key, labels
        print(f"[cluster] {params} -> {n} clusters, noise {noise:.1%}, largest {largest:.1%}", flush=True)
    return best_labels, rows, best


def _heading_key(heading: str | None) -> str:
    return _WS.sub(" ", re.sub(r"[^a-z ]+", " ", (heading or "").lower())).strip() or "(no heading)"


def summarise_clusters(claims: list[dict[str, Any]], labels: "np.ndarray", X: "np.ndarray", *,
                       top: int = 5) -> list[dict[str, Any]]:
    """One summary per cluster, largest first, ids c000, c001, ...: members ordered by cosine similarity to the
    cluster centroid (most typical first), mean similarity (cohesion), document count, the most frequent extraction
    labels and section headings, and the mean relative position of the source section in its document (0 = first
    section, 1 = last; approximated by the highest section index with a claim)."""
    from collections import Counter

    import numpy as np

    last_idx: dict[str, int] = {}
    for c in claims:
        last_idx[c["doc_id"]] = max(last_idx.get(c["doc_id"], 0), c["section_idx"])
    out = []
    for lab in sorted(set(int(x) for x in labels) - {-1}):
        idx = np.flatnonzero(labels == lab)
        centroid = X[idx].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        sims = X[idx] @ centroid
        order = idx[np.argsort(-sims, kind="stable")]
        members = [claims[i] for i in order]
        pos = [m["section_idx"] / last_idx[m["doc_id"]] if last_idx[m["doc_id"]] else 0.0 for m in members]
        out.append({"size": len(idx), "documents": len({m["doc_id"] for m in members}),
                    "cohesion": round(float(sims.mean()), 4), "position_mean": round(float(np.mean(pos)), 3),
                    "type_guess_top": Counter(m["type_guess"] for m in members).most_common(top),
                    "heading_top": Counter(_heading_key(m["heading"]) for m in members).most_common(3),
                    "members": [m["claim_id"] for m in members], "centroid": centroid.tolist()})
    out.sort(key=lambda c: -c["size"])
    for k, c in enumerate(out):
        c["id"] = f"c{k:03d}"
    return out


def naming_sample(members: list[str], *, n: int, seed: int) -> list[str]:
    """What the teacher sees: the n//2 most typical members, then a seeded random sample of the rest."""
    head = members[: n // 2]
    rest = members[n // 2:]
    return head + random.Random(seed).sample(rest, min(n - len(head), len(rest)))


def prompt_example_labels(prompt_id: str = EXTRACT_PROMPT) -> set[str]:
    """The snake_case labels a prompt shows as examples (quoted in its text), minus its own field names. A cluster
    named after one of them has copied the prompt, not found a genre type."""
    from models.teacher import PROMPTS

    template = PROMPTS.get(prompt_id).template
    return set(re.findall(r'"([a-z][a-z0-9]*(?:_[a-z0-9]+)+)"', template)) - {"type_guess"}


class ClusterName(BaseModel):
    name: str = Field(description="snake_case label of 2 to 5 words for the kind of statement")
    definition: str = Field(description="one sentence that decides membership")
    representative: list[int] = Field(description="numbers of the three most representative claims")
    single_kind: bool = Field(description="false if the group mixes kinds of statement")
    split_note: str = Field(description="the kinds mixed, if any")


def render_name_prompt(claims: list[str], genre: dict[str, Any], *, cluster_size: int, n_documents: int,
                       type_hints: list[tuple[str, int]], prompt_id: str = NAME_PROMPT) -> str:
    from models.teacher import PROMPTS

    return PROMPTS.get(prompt_id).render(
        document_kind=genre["document_kind"], author_role=genre["author_role"], n_shown=len(claims),
        cluster_size=cluster_size, n_documents=n_documents,
        claims="\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1)),
        type_hints=", ".join(f"{label} ({n})" for label, n in type_hints) or "(none)")


def name_clusters(teacher: Any, prompts: list[str], *, max_tokens: int, batch_size: int = 8,
                  skip_failures: bool = True) -> list[ClusterName | None]:
    return _structured_in_batches(teacher, prompts, ClusterName, max_tokens=max_tokens, batch_size=batch_size,
                                  skip_failures=skip_failures)


def snake(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_") or "unnamed"


def draft_types(clusters: list[dict[str, Any]], shown: list[list[str]], names: list[ClusterName | None],
                claims: dict[str, dict[str, Any]], *, example_labels: set[str], n_documents: int,
                n_representative: int = 3) -> list[dict[str, Any]]:
    """One draft type per named cluster. Representative claims are the teacher's picks (valid, distinct), topped up
    with the most typical members. Flags for the human pass: failed naming, a mixed cluster, a name that copies an
    example label of the extraction prompt, and a name shared with another cluster (a merge candidate)."""
    from collections import Counter

    counts = Counter(snake(n.name) for n in names if n is not None)
    out = []
    for c, ids, nm in zip(clusters, shown, names):
        picks: list[str] = []
        for k in (nm.representative if nm else []):
            if 1 <= k <= len(ids) and ids[k - 1] not in picks:
                picks.append(ids[k - 1])
        for cid in c["members"]:
            if len(picks) >= n_representative:
                break
            if cid not in picks:
                picks.append(cid)
        name = snake(nm.name) if nm else f"unnamed_{c['id']}"
        flags = []
        if nm is None:
            flags.append("naming_failed")
        elif not nm.single_kind:
            flags.append("mixed")
        if any(label == name or label in name for label in example_labels):
            flags.append("example_label_name")
        if nm is not None and counts[name] > 1:
            flags.append("duplicate_name")
        leak = sum(claims[m]["type_guess"] in example_labels for m in c["members"]) / c["size"]
        out.append({
            "id": c["id"], "name": name, "definition": nm.definition if nm else "",
            "size": c["size"], "documents": c["documents"],
            "document_share": round(c["documents"] / max(1, n_documents), 3), "cohesion": c["cohesion"],
            "single_kind": nm.single_kind if nm else None, "split_note": nm.split_note if nm else "",
            "flags": flags, "example_label_share": round(leak, 3),
            "representative_claims": [{"claim_id": i, "claim": claims[i]["claim"]} for i in picks[:n_representative]],
            "type_guess_top": [[label, n] for label, n in c["type_guess_top"]],
            "heading_top": [[h, n] for h, n in c["heading_top"]], "position_mean": c["position_mean"]})
    return out


# ==================================================================================== schema review: coverage (§3.5)
COVERAGE_PROMPT = "induce_coverage"


def coverage_output_model(type_ids: list[str]) -> type[BaseModel]:
    """Output model whose `type` may only be a schema id or "none" (enforced by structured output and validation)."""
    from typing import Literal

    from pydantic import create_model

    TypeId = Literal[tuple(list(type_ids) + ["none"])]  # type: ignore[misc]
    Item = create_model("CoverageAssertion", claim=(str, ...), type=(TypeId, ...))
    return create_model("SectionCoverage", assertions=(list[Item], ...))  # type: ignore[valid-type]


def render_coverage_prompt(section: Section, genre: dict[str, Any], types: list[dict[str, Any]], *,
                           max_assertions: int, prompt_id: str = COVERAGE_PROMPT) -> str:
    from models.teacher import PROMPTS

    return PROMPTS.get(prompt_id).render(
        document_kind=genre["document_kind"], author_role=genre["author_role"],
        type_catalogue="\n".join(f"{t['id']}: {t['definition']}" for t in types),
        section_heading=section.heading or "(no heading)", section_text=section.text, max_assertions=max_assertions)


def classify_sections(teacher: Any, sections: list[Section], genre: dict[str, Any], types: list[dict[str, Any]], *,
                      max_assertions: int, max_tokens: int, batch_size: int = 8) -> list[Any]:
    model = coverage_output_model([t["id"] for t in types])
    prompts = [render_coverage_prompt(s, genre, types, max_assertions=max_assertions) for s in sections]
    return _structured_in_batches(teacher, prompts, model, max_tokens=max_tokens, batch_size=batch_size,
                                  skip_failures=True)


VERIFY_PROMPT = "induce_coverage_verify"


def schema_fingerprint(schema: dict[str, Any]) -> str:
    """A hash of what a schema says (types, seed mapping, pruned clusters), not of its file: rebuilding a candidate
    (new run names, dates, workbook hash in its provenance) keeps the fingerprint; changing a type changes it."""
    body = {k: schema.get(k) for k in ("types", "seed_mapping", "pruned_clusters")}
    return _sha1(json.dumps(body, sort_keys=True, ensure_ascii=False))[:12]


def type_card(t: dict[str, Any]) -> str:
    """One type as the second look shows it (and as it is embedded for retrieval): id, definition, examples."""
    examples = " | ".join(t.get("positive_examples", [])[:2])
    return f"{t['id']}: {t['definition']}" + (f" Examples: {examples}" if examples else "")


def verify_output_model(type_ids: list[str]) -> type[BaseModel]:
    from typing import Literal

    from pydantic import create_model

    return create_model("VerifyAnswer", type=(Literal[tuple(list(type_ids) + ["none"])], ...))  # type: ignore[misc]


def render_verify_prompt(statement: str, heading: str | None, genre: dict[str, Any], candidates: list[dict[str, Any]],
                         prompt_id: str = VERIFY_PROMPT) -> str:
    from models.teacher import PROMPTS

    return PROMPTS.get(prompt_id).render(
        document_kind=genre["document_kind"], author_role=genre["author_role"], statement=statement,
        section_heading=heading or "(no heading)", candidates="\n".join(f"- {type_card(t)}" for t in candidates))


def second_look(teacher: Any, embedder: Any, statements: list[tuple[str, str | None]], genre: dict[str, Any],
                types: list[dict[str, Any]], *, top_k: int, max_tokens: int, batch_size: int = 8) -> list[str]:
    """For each (statement, heading) the first pass mapped to "none": retrieve the top_k types whose card is most
    similar to the statement (cosine of embeddings) and let the teacher decide again among those. Returns the
    final type id or "none" per statement (a failed call keeps "none")."""
    import numpy as np

    if not statements:
        return []
    T = embedder.encode([type_card(t) for t in types])
    S = embedder.encode([s for s, _ in statements])
    T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
    order = np.argsort(-(S @ T.T), axis=1)[:, :top_k]
    prompts = [render_verify_prompt(s, h, genre, [types[j] for j in row]) for (s, h), row in zip(statements, order)]
    model = verify_output_model([t["id"] for t in types])
    answers = _structured_in_batches(teacher, prompts, model, max_tokens=max_tokens, batch_size=batch_size,
                                     skip_failures=True)
    return [a.type if a is not None else "none" for a in answers]
