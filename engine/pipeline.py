"""analyse(document, pack) -> Report (plan §0.2 #1, #13, #14; §5.6).

    pack = GenrePack.load(cfg, "privacy")
    report = analyse("policy.md", pack)          # or the text itself
    print(report.to_json())

Steps (each timed): read -> segment (the corpus's generic Markdown segmentation) -> embed sections and title with the
index's embedder -> section presence (linear heads + prototypes, combined by cascade.json) -> document presence
(doc_presence.json; the v1 best-section rule when a pack has none, flagged) -> document vector -> cohort (never the
document's own entity) -> level-1 expectation with Wilson intervals -> rank -> peer evidence -> template explanation.
No LLM on this path, and the uploaded text is not stored.

analyse() = score(prepare(..)) (plan 7.1): prepare() is the slow part, up to the document vector, done once per
upload; score() is the rest, so the app re-ranks under other cohort settings (k, category filter, date window)
without embedding the upload again.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Report:
    genre: str
    title: str | None
    sections: int
    present: list[dict[str, Any]]
    absences: list[dict[str, Any]]
    cohort: dict[str, Any]
    timings: dict[str, float]
    pack: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)


def read_document(text_or_path: str | Path) -> str:
    """Plain text or Markdown, given as text or as a path to a .txt / .md / .html file (tags stripped)."""
    p = Path(text_or_path) if isinstance(text_or_path, Path) or (len(str(text_or_path)) < 4096 and "\n" not in str(text_or_path)) else None
    if p is not None and p.is_file():
        text = p.read_text(encoding="utf-8", errors="replace")
        if p.suffix.lower() in (".html", ".htm"):
            import html
            import re

            text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
            text = re.sub(r"(?i)<(br|/p|/div|/h[1-6]|/li)\s*/?>", "\n", text)
            text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        return text
    return str(text_or_path)


def guess_title(text: str) -> str | None:
    """The first Markdown heading, else the first non-empty line that is not an HTML comment (at most 200 characters)."""
    import re

    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not re.match(r"^<!--.*-->$", ln.strip())]
    for ln in lines[:50]:
        m = re.match(r"^#{1,6}\s+(.+)$", ln)
        if m:
            return m.group(1).strip()[:200]
    return lines[0][:200] if lines else None


def segment(text: str, scfg: dict[str, Any]) -> list[tuple[str | None, str]]:
    """Sections of an upload: the corpus's generic Markdown segmentation (data/segment.py), or, for Markdown that
    carries its own sections (document_markdown's format), those sections as written, only oversized ones split."""
    from data.segment import normalise_sections, read_sections_markdown, segment_markdown, split_oversized

    marked = read_sections_markdown(text)
    if marked is not None:
        raw = marked[1]
    else:
        raw = normalise_sections(segment_markdown(text, fallback_chars=scfg.get("fallback_section_chars", 1500)),
                                 min_chars=scfg.get("min_section_chars", 80), max_chars=scfg.get("max_section_chars", 8000))
    raw = split_oversized(raw, max_chars=scfg.get("max_section_chars", 8000))
    return [(r.heading, r.text) for r in raw if r.text.strip()]


def document_markdown(doc: Any, sections: list[Any]) -> str:
    """A corpus document rebuilt as Markdown (title, then each section under its heading) that segment() reads back
    as the stored sections, so the query is segmented like its indexed peers (CHANGELOG 2026-10-20: re-segmenting
    plain Markdown changed 25% of privacy and 89% of papers test documents)."""
    from data.segment import write_sections_markdown

    return write_sections_markdown(doc.title, [(s.heading, s.text) for s in sections])


@dataclass
class Prepared:
    """The slow, per-upload part of analyse() (segment, embed, presence): computed once, re-scored under other cohort
    settings by score(). Held in memory only (plan §0.2 #21: uploads are never stored)."""
    sections: list[tuple[str | None, str]]
    title: str | None
    S: np.ndarray            # (sections, dim) section embeddings
    vector: np.ndarray       # document vector (plan §0.2 #9)
    P: np.ndarray            # (sections, T) section presence
    p_doc: np.ndarray        # (T,) document presence
    bits: np.ndarray         # (T,) present types
    timings: dict[str, float]


@dataclass
class Embedded:
    """An upload segmented and embedded with a pack's embedder: shared by every pack with the same embedder and
    segmentation settings (pack auto-detection embeds once)."""
    sections: list[tuple[str | None, str]]
    title: str | None
    S: np.ndarray
    title_vector: np.ndarray | None
    embedder: str
    timings: dict[str, float]


def _timer(timings: dict[str, float]) -> Any:
    tick = [time.perf_counter()]

    def lap(name: str) -> None:
        now = time.perf_counter()
        timings[name] = round(now - tick[0], 3)
        tick[0] = now
    return lap


def embed_upload(text_or_path: str | Path, pack: Any, *, title: str | None = None) -> Embedded:
    timings: dict[str, float] = {}
    lap = _timer(timings)
    text = read_document(text_or_path)
    lap("read")
    secs = segment(text, pack.cfg["segment"])
    if not secs:
        raise ValueError("the document has no text to analyse")
    if title is None:
        title = guess_title(text)
    lap("segment")
    texts = [t for _, t in secs]
    V = np.asarray(pack.embedder.encode(texts + ([title] if title else [])))
    lap("embed")
    return Embedded(sections=secs, title=title, S=V[:len(texts)], title_vector=V[len(texts)] if title else None,
                    embedder=(getattr(pack, "info", {}) or {}).get("embedder", ""), timings=timings)


def document_vector_of(embedded: Embedded, pack: Any) -> np.ndarray:
    from engine.embed import document_vector

    return document_vector(embedded.S, embedded.title_vector,
                           title_abstract_weight=pack.cfg["models"]["embedder"].get("title_abstract_weight", 0.5))


def prepare(text_or_path: str | Path, pack: Any, *, title: str | None = None, embedded: Embedded | None = None) -> Prepared:
    """embedded: the upload already embedded by embed_upload (with this pack's embedder), else embedded here."""
    if embedded is None:
        embedded = embed_upload(text_or_path, pack, title=title)
    t0 = time.perf_counter()
    timings = dict(embedded.timings)
    lap = _timer(timings)
    secs, S = embedded.sections, embedded.S
    clf = pack.heads.predict_proba(S)
    proto = pack.scorer.score([t for _, t in secs])
    P = pack.cascade.predict(clf, proto, np.full(clf.shape, np.nan))
    lap("presence")
    if pack.doc_model is not None:
        p_doc = pack.doc_model.predict_proba(P)
        bits = (p_doc >= 0.5).astype(int)
    else:  # the v1 rule: the best section reaches the section threshold
        p_doc = P.max(0)
        bits = (p_doc >= pack.cascade.threshold).astype(int)
    lap("document_presence")
    vec = document_vector_of(embedded, pack)
    lap("document_vector")
    timings["prepare"] = round(sum(embedded.timings.values()) + time.perf_counter() - t0, 3)
    return Prepared(sections=secs, title=embedded.title, S=S, vector=vec, P=P, p_doc=p_doc, bits=bits, timings=timings)


def score(prepared: Prepared, pack: Any, *, category: str | None = None, entity_id: str | None = None,
          date_window: tuple[str | None, str | None] | None = None, k: int | None = None,
          category_filter: bool | None = None, top_k: int | None = None, explain_mode: str | None = None,
          teacher: Any = None) -> Report:
    """The fast part: cohort (k, category filter and date window override the config), expectation, rank, evidence,
    explanation. Timings: prepare's laps, then this call's; total = prepare + score."""
    from engine.cohort import CohortRetriever
    from engine.evidence import PeerEvidence
    from engine.expectation import level1
    from engine.explain import explain_absences
    from engine.rank import rank_absences

    cfg, T = pack.cfg, len(pack.types)
    secs, title, P, p_doc, bits = prepared.sections, prepared.title, prepared.P, prepared.p_doc, prepared.bits
    timings: dict[str, float] = {k_: v for k_, v in prepared.timings.items() if k_ != "prepare"}
    t0 = time.perf_counter()
    lap = _timer(timings)

    retriever = CohortRetriever.from_config(pack.index, cfg.get("cohort") or {})
    if category_filter is not None:
        retriever.category_filter = category_filter
    coh = retriever.retrieve(prepared.vector, entity_id=entity_id, category=category, date_window=date_window, k=k)
    peers = [h.id for h in coh.peers]
    lap("cohort")
    e = level1(pack.profiles, peers, z=(cfg.get("expectation") or {}).get("wilson_z", 1.96),
               use=(cfg.get("expectation") or {}).get("profile", "bit"))
    lap("expectation")
    rcfg = cfg.get("rank") or {}
    ranked = rank_absences(pack.types, e["prevalence"], p_doc, bits, weights=pack.weights(), prevalence=e["prevalence"],
                           top_k=top_k or rcfg.get("top_k", 10), threshold=rcfg.get("threshold"))
    lap("rank")
    def sentence_prob(groups: list[list[str]], t: int, top: int = 3) -> list[np.ndarray]:
        """P(sentence states t): ColBERT pre-ranks each group's sentences, the presence model scores the best `top`."""
        flat = [s for g in groups for s in g]
        proto_s = pack.scorer.score(flat)
        sel, start = [], 0
        for g in groups:
            sel += list(start + np.argsort(-proto_s[start:start + len(g), t])[:top])
            start += len(g)
        clf_s = pack.heads.predict_proba(np.asarray(pack.embedder.encode([flat[i] for i in sel])))
        p_s = pack.cascade.predict(clf_s, proto_s[sel], np.full(clf_s.shape, np.nan))[:, t]
        full = np.zeros(len(flat))
        full[sel] = p_s
        out, start = [], 0
        for g in groups:
            out.append(full[start:start + len(g)])
            start += len(g)
        return out

    ev = PeerEvidence(pack.best, pack.section_texts, sentence_prob, pack.prototypes, pack.types,
                      store=getattr(pack, "evidence_store", None))
    k_quotes = (cfg.get("evidence") or {}).get("max_peer_sentences", 5)
    rows_of = {d: pack.profiles.row[d] for d in peers if d in pack.profiles.row}
    cards = pack.cards
    absences = []
    for r in ranked:
        t = pack.types.index(r["type"])
        stating = [d for d in peers if d in rows_of and pack.profiles.bits[rows_of[d], t]]
        stating.sort(key=lambda d: -pack.profiles.probs[rows_of[d], t])  # the peers that state it most clearly first
        quotes = ev.for_type(r["type"], stating, k=k_quotes, min_prob=float(pack.cascade.threshold[t]))
        absences.append({**r, "count": float(e["count"][t]), "n": e["n"], "ci": [float(e["ci_low"][t]), float(e["ci_high"][t])],
                         "definition": cards[r["type"]].get("definition"), "section_hint": cards[r["type"]].get("section_hint"),
                         "evidence": quotes})
    lap("evidence")
    xcfg = cfg.get("explain") or {}
    mode = explain_mode or xcfg.get("mode", "template")
    if mode == "teacher" and teacher is None:
        from models.teacher import load_teacher

        teacher = load_teacher(cfg["models"]["teacher"])
    explain_absences(absences, cards, mode=mode, teacher=teacher, genre=(cfg.get("genre_info") or {}).get(pack.genre),
                     top_n=xcfg.get("top_n", 5), max_tokens=xcfg.get("max_tokens", 256))
    lap("explain")
    best_sec = P.argmax(0)
    present = [{"type": t, "p_present": float(p_doc[k]), "section": int(best_sec[k]), "heading": secs[int(best_sec[k])][0]}
               for k, t in enumerate(pack.types) if bits[k]]
    present.sort(key=lambda x: -x["p_present"])
    timings["total"] = round(prepared.timings.get("prepare", 0.0) + time.perf_counter() - t0, 3)
    cohort = {"k": coh.stats["k"], "n": coh.stats["n"], "fallback": coh.fallback, "filters": coh.filters_used,
              "stats": coh.stats, "nearest": [{"doc_id": h.id, "similarity": round(h.score, 4),
                                               "category": h.payload.get("category"), "date": h.payload.get("date")}
                                              for h in coh.peers[:10]]}
    return Report(genre=pack.genre, title=title, sections=len(secs), present=present, absences=absences, cohort=cohort,
                  timings=timings, pack={**pack.info, "types": T})


def analyse(text_or_path: str | Path, pack: Any, *, title: str | None = None, category: str | None = None,
            entity_id: str | None = None, date_window: tuple[str | None, str | None] | None = None,
            top_k: int | None = None, explain_mode: str | None = None, teacher: Any = None) -> Report:
    return score(prepare(text_or_path, pack, title=title), pack, category=category, entity_id=entity_id,
                 date_window=date_window, top_k=top_k, explain_mode=explain_mode, teacher=teacher)
