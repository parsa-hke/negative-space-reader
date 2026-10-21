"""Controlled-deletion generator for OmitBench-synthetic (plan 6.1, §0.2 #15).

For a held-out (test) document and types it states, delete what carries them and check, with the pack's own
document-level presence model, that the document no longer states them. Three regimes:

  single_type    one type, drawn uniformly from the document's present types; delete the sentences carrying it
  three_types    three such types; delete the union of their carrying sentences; all three must become absent
  whole_section  one type; delete every section carrying it (whole sections, headings included)

A sentence carries t when the presence model gives it P(t) >= the cascade's threshold for t (the threshold
analyse() quotes evidence at); a section carries t when its own probability passes it; when none passes, the best
one is taken. After each edit the changed sections are rescored (rescore) and document presence recomputed
(doc_prob). While a target is still present (>= 0.5) the edit escalates: in the sentence regimes, up to
sentence_rounds times the sentences_per_round most likely remaining carriers of each such target go too (only
sentences with P >= min_carrier_prob); then, in every regime, up to section_rounds times the whole section that now
scores the target highest. Otherwise the instance is rejected (still_present). Escalation is needed because document presence pools every section: in a long
paper a type stated in a dozen sections stays present after its clearest sentences go (CHANGELOG 2026-10-20). An edit
may remove at most max_removed_share of the document's characters. A section left shorter than min_section_chars is
dropped (the corpus's segmentation drops such sections too).

Ground truth is every type present before the edit and absent after it (Parsa, 2026-10-20): the sampled targets plus
any type deleted along with them (collateral, recorded). Edits work on the stored sections, not on re-segmented
Markdown: the corpus's sections are what the indexed peers' profiles were built from (the old plain-Markdown round
trip segmented 25% of privacy and 89% of papers test documents differently; document_markdown now round-trips
exactly, CHANGELOG 2026-10-20).
"""

from __future__ import annotations

import re
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

REGIMES = ("single_type", "three_types", "whole_section")


@dataclass
class BenchSection:
    idx: int
    heading: str | None
    text: str
    sentences: list[str]       # split_sentences(text), as stored in sentences.jsonl
    S: np.ndarray              # (n_sentences, T) sentence presence probabilities
    P: np.ndarray              # (T,) the section's presence probabilities
    V: np.ndarray              # (D,) the section's stored embedding


@dataclass
class BenchDoc:
    id: str
    entity: str | None
    title: str | None
    category: str | None
    date: str | None
    title_vector: np.ndarray | None
    sections: list[BenchSection]


def _pattern(sentence: str) -> re.Pattern[str]:
    return re.compile(r"\s+".join(re.escape(w) for w in sentence.split()))


def delete_sentences(text: str, sentences: Sequence[str], drop: set[int]) -> str:
    """The section text without the sentences in drop; the rest is kept verbatim (stored sentences differ from the
    text only in whitespace, so each is located with a whitespace-tolerant match, in order). If a sentence cannot be
    located, the section is rebuilt from the kept sentences."""
    if not drop:
        return text
    spans, pos = [], 0
    for s in sentences:
        m = _pattern(s).search(text, pos)
        if m is None:
            return " ".join(s for j, s in enumerate(sentences) if j not in drop)
        spans.append((m.start(), m.end()))
        pos = m.end()
    pieces, pos = [], 0
    for j, (a, b) in enumerate(spans):
        if j in drop:
            pieces.append(text[pos:a])
            pos = b
    pieces.append(text[pos:])
    out = ""
    for p in pieces:  # whitespace is tidied only where something was cut out
        left, right = out.rstrip(" \t"), p.lstrip(" \t")
        sep = " " if left and right and not left.endswith("\n") and not right.startswith("\n") else ""
        out = left + sep + right
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _doc_vector(V: np.ndarray, title_vector: np.ndarray | None, w: float) -> np.ndarray:
    from engine.embed import document_vector

    return document_vector(V, title_vector, title_abstract_weight=w)


def make_instance(doc: BenchDoc, regime: str, rng: np.random.Generator, *, types: Sequence[str],
                  thresholds: np.ndarray, rescore: Callable[[list[str]], tuple[np.ndarray, np.ndarray]],
                  doc_prob: Callable[[np.ndarray], np.ndarray], targets: Sequence[str] | None = None,
                  min_section_chars: int = 80, max_removed_share: float = 0.5, sentence_rounds: int = 8,
                  sentences_per_round: int = 5, min_carrier_prob: float = 0.02, section_rounds: int = 5,
                  title_abstract_weight: float = 0.5, trace: list[tuple[list[str], str]] | None = None
                  ) -> tuple[dict[str, Any] | None, str]:
    """One verified instance of the regime from doc, or (None, reason). rescore(texts) -> (P (n, T), V (n, D)) for
    edited section texts; doc_prob(P) -> (T,) document-level presence. targets fixes the types (tests); trace, when
    given, receives (drawn targets, outcome)."""
    inst, reason, drawn = _make_instance(doc, regime, rng, types=types, thresholds=thresholds, rescore=rescore,
                                         doc_prob=doc_prob, targets=targets, min_section_chars=min_section_chars,
                                         max_removed_share=max_removed_share, sentence_rounds=sentence_rounds,
                                         sentences_per_round=sentences_per_round, min_carrier_prob=min_carrier_prob,
                                         section_rounds=section_rounds, title_abstract_weight=title_abstract_weight)
    if trace is not None:
        trace.append((drawn, reason))
    return inst, reason


def _make_instance(doc: BenchDoc, regime: str, rng: np.random.Generator, *, types: Sequence[str],
                   thresholds: np.ndarray, rescore: Callable[[list[str]], tuple[np.ndarray, np.ndarray]],
                   doc_prob: Callable[[np.ndarray], np.ndarray], targets: Sequence[str] | None,
                   min_section_chars: int, max_removed_share: float, sentence_rounds: int, sentences_per_round: int,
                   min_carrier_prob: float, section_rounds: int, title_abstract_weight: float
                   ) -> tuple[dict[str, Any] | None, str, list[str]]:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r} ({', '.join(REGIMES)})")
    secs = doc.sections
    if not secs:
        return None, "no_sections", []
    P0 = np.stack([s.P for s in secs])
    p0 = doc_prob(P0)
    present = np.flatnonzero(p0 >= 0.5)
    want = 3 if regime == "three_types" else 1
    if targets is not None:
        tix = [list(types).index(t) for t in targets]
        drawn = list(targets)
        if any(p0[k] < 0.5 for k in tix):
            return None, "target_not_present", list(targets)
    elif len(present) < want:
        return None, "too_few_types", []
    else:
        tix = sorted(rng.choice(present, size=want, replace=False).tolist())
    drawn = [list(types)[k] for k in tix]
    total = sum(len(s.text) for s in secs)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def score(texts: list[str]) -> None:
        new = [t for t in dict.fromkeys(texts) if t not in cache]
        if new:
            P, V = rescore(new)
            cache.update({t: (P[i], V[i]) for i, t in enumerate(new)})

    def evaluate(kept: list[tuple[int, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """kept = [(section position, text)] -> (P, V, p_doc) of the edited document."""
        changed = [t for i, t in kept if t != secs[i].text]
        score(changed)
        P = np.stack([secs[i].P if t == secs[i].text else cache[t][0] for i, t in kept])
        V = np.stack([secs[i].V if t == secs[i].text else cache[t][1] for i, t in kept])
        return P, V, doc_prob(P)

    drop: dict[int, set[int]] = {i: set() for i in range(len(secs))}
    dropped: set[int] = set()
    if regime == "whole_section":
        t = tix[0]
        dropped = {i for i, s in enumerate(secs) if s.P[t] >= thresholds[t]} or {int(P0[:, t].argmax())}
    else:
        for t in tix:
            hits = [(i, j) for i, s in enumerate(secs) for j in range(len(s.sentences)) if s.S[j, t] >= thresholds[t]]
            if not hits:
                best = max(((i, j) for i, s in enumerate(secs) for j in range(len(s.sentences))),
                           key=lambda ij: secs[ij[0]].S[ij[1], t], default=None)
                hits = [best] if best is not None else []
            for i, j in hits:
                drop[i].add(j)
    level = sec_rounds = 0
    while True:
        kept = []
        for i, s in enumerate(secs):
            if i in dropped:
                continue
            text = delete_sentences(s.text, s.sentences, drop[i]) if drop[i] else s.text
            if text == s.text or len(text) >= min_section_chars:
                kept.append((i, text))
        if not kept:
            return None, "empty", drawn
        if 1 - sum(len(x) for _, x in kept) / total > max_removed_share:
            return None, "too_much_removed", drawn
        P1, V1, p1 = evaluate(kept)
        still = [t for t in tix if p1[t] >= 0.5]
        if not still:
            break
        pos = [i for i, _ in kept]
        cands = {t: sorted(((secs[i].S[j, t], i, j) for i in pos for j in range(len(secs[i].sentences))
                            if j not in drop[i] and secs[i].S[j, t] >= min_carrier_prob), reverse=True) for t in still}
        if regime != "whole_section" and level < sentence_rounds and any(cands.values()):
            level += 1  # the next most likely carriers of each target still present
            for t in still:
                for _, i, j in cands[t][:sentences_per_round]:
                    drop[i].add(j)
        elif sec_rounds < section_rounds:
            sec_rounds += 1  # then the whole section that now scores the type highest
            for t in still:
                dropped.add(pos[int(P1[:, t].argmax())])
        else:
            return None, "still_present", drawn
    deleted = sorted([secs[i].idx, j] for i in drop if i not in dropped for j in drop[i])
    deleted_sections = sorted(secs[i].idx for i in dropped)
    gt = [k for k in present if p1[k] < 0.5]
    names = list(types)
    return {"doc_id": doc.id, "entity_id": doc.entity, "regime": regime, "category": doc.category, "date": doc.date,
            "title": doc.title, "targets": [names[k] for k in tix], "ground_truth": [names[k] for k in gt],
            "collateral": [names[k] for k in gt if k not in tix], "deleted": deleted, "deleted_sections": deleted_sections,
            "rounds": level + sec_rounds, "escalation": {"sentence_rounds": level, "section_rounds": sec_rounds},
            "removed_share": round(1 - sum(len(x) for _, x in kept) / total, 4),
            "sections_before": len(secs), "sections_after": len(kept),
            "p_before": {names[k]: round(float(p0[k]), 4) for k in gt},
            "p_after": {names[k]: round(float(p1[k]), 4) for k in gt},
            "sections": [{"section_idx": secs[i].idx, "heading": secs[i].heading, "text": x, "edited": x != secs[i].text}
                         for i, x in kept],
            "P": P1.astype(np.float32), "p_doc": np.asarray(p1, dtype=np.float32),
            "doc_vector": _doc_vector(V1, doc.title_vector, title_abstract_weight)}, "ok", drawn


def instance_rng(seed: int, regime: str, doc_id: str) -> np.random.Generator:
    """The draws of one (document, regime) pair, independent of the order and of the other pairs."""
    return np.random.default_rng([seed, REGIMES.index(regime), zlib.crc32(doc_id.encode())])


def eta_seconds(counts: dict[str, int], n: int, *, tried: int, total: int | None, elapsed: float) -> float | None:
    """Seconds left: the open regime that needs the most documents at its acceptance so far (instances per document
    tried), at the time per document so far, capped by the documents left; every document left while a regime has no
    instance yet. None without a total and without a rate."""
    if not tried:
        return None
    per_doc = elapsed / tried
    left = None if total is None else max(total - tried, 0)
    need = 0.0
    for c in counts.values():
        if c >= n:
            continue
        if c == 0:
            if left is None:
                return None
            need = float(left)
            break
        need = max(need, (n - c) * tried / c)
    if left is not None:
        need = min(need, left)
    return need * per_doc


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    m = int(round(seconds / 60))
    return f"{m // 60}h{m % 60:02d}m"


def build_bench(docs: Sequence[BenchDoc] | Any, regimes: Sequence[str], n: int, *, seed: int,
                attempts_per_document: int = 1, log: Callable[[str], None] | None = None,
                total: int | None = None, progress_every_s: float = 60.0,
                checkpoint: Callable[[dict[str, Any]], None] | None = None, checkpoint_every_s: float = 300.0,
                state: dict[str, Any] | None = None,
                **kw: Any) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Counter]]:
    """Up to n verified instances per regime, drawn from docs in their order (the bench order), at most one document
    per entity per regime; a rejected draw is retried with new targets up to attempts_per_document times (this
    favours types that can be deleted cleanly: per-type acceptance is reported). docs may be a sequence or a callable
    returning a fresh iterator. Returns the instances and, per regime, the count of every outcome (ok and each
    rejection reason; same_entity for skipped snapshots).

    Long runs (CHANGELOG 2026-10-20/21: a 1 h 50 min Colab run could not be interrupted without losing it): every
    progress_every_s, log gets a line with a timestamp, the documents tried (of total), the instances per regime and an
    ETA (eta_seconds); every checkpoint_every_s (and after the last document), checkpoint receives the loop's whole
    state. Passing that state back as state= skips the documents already tried and continues: every (document,
    regime) draw has its own generator (instance_rng), so the result equals an uninterrupted run's."""
    for r in regimes:
        if r not in REGIMES:
            raise ValueError(f"unknown regime {r!r} ({', '.join(REGIMES)})")
    if state is not None:
        if list(state["out"]) != list(regimes) or state["n"] != n or state["seed"] != seed:
            raise ValueError("the checkpoint was made with other regimes, n or seed")
        out, used, traces = state["out"], {r: set(v) for r, v in state["used"].items()}, state["traces"]
        stats = {r: Counter(v) for r, v in state["stats"].items()}
        tried, elapsed0 = state["tried"], state.get("elapsed", 0.0)
    else:
        out = {r: [] for r in regimes}
        stats = {r: Counter() for r in regimes}
        used = {r: set() for r in regimes}
        traces = {r: [] for r in regimes}
        tried, elapsed0 = 0, 0.0
    t0 = time.perf_counter()
    last_log = last_ckpt = t0
    skip = tried

    def snapshot() -> dict[str, Any]:
        return {"n": n, "seed": seed, "tried": tried, "elapsed": elapsed0 + time.perf_counter() - t0, "out": out,
                "stats": {r: dict(c) for r, c in stats.items()}, "used": {r: sorted(v) for r, v in used.items()},
                "traces": traces}

    def progress(now: float) -> str:
        elapsed = elapsed0 + now - t0
        eta = eta_seconds({r: len(out[r]) for r in regimes}, n, tried=tried, total=total, elapsed=elapsed)
        return (f"{time.strftime('%Y-%m-%d %H:%M:%S')} docs {tried}/{total if total is not None else '?'} | "
                + " ".join(f"{r} {len(out[r])}/{n}" for r in regimes)
                + f" | {60 * tried / elapsed if elapsed else 0:.1f} docs/min, elapsed {_clock(elapsed)}, eta {_clock(eta)}")

    for d in (docs() if callable(docs) else docs):
        if skip:
            skip -= 1
            continue
        open_regimes = [r for r in regimes if len(out[r]) < n]
        if not open_regimes:
            break
        for r in open_regimes:
            ent = d.entity or d.id
            if ent in used[r]:
                stats[r]["same_entity"] += 1
                continue
            rng = instance_rng(seed, r, d.id)
            for _ in range(attempts_per_document if "targets" not in kw else 1):
                inst, reason = make_instance(d, r, rng, trace=traces[r], **kw)
                stats[r][reason] += 1
                if inst is not None or reason in ("too_few_types", "no_sections"):
                    break
            if inst is not None:
                inst["instance_id"] = f"{r}:{d.id}"
                out[r].append(inst)
                used[r].add(ent)
        tried += 1
        now = time.perf_counter()
        if log and now - last_log >= progress_every_s:
            log(progress(now))
            last_log = now
        if checkpoint and now - last_ckpt >= checkpoint_every_s:
            checkpoint(snapshot())
            last_ckpt = now
    if log:
        log(progress(time.perf_counter()) + " (done)")
    if checkpoint:
        checkpoint(snapshot())
    for r in regimes:  # per type: how often a draw including it was accepted / rejected, by reason
        stats[r + "/by_type"] = Counter(f"{t}|{reason}" for drawn, reason in traces[r] for t in drawn)
    return out, stats
