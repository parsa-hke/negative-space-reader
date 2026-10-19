"""Peer sentences and counts for each absence (plan §0.2 #14; §5.5).

Evidence for "N% of peers like this one state X; this document does not" comes from the cohort itself: for each peer
that states the type, the section where it does so most clearly (highest stored section probability,
packs/<genre>/section_probs), and inside it the sentence the presence model scores highest for the type (a scoring
callback: ColBERT pre-ranks a section's sentences, the linear heads and cascade score the best few); a peer whose best
sentence stays below the type's threshold is not quoted. When fewer than k peers can be quoted, the type's prototype sentences fill
up the list, marked `source: prototype` (they come from the labelled set, not from this cohort). Nothing that is not
a quote is produced here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


def best_sections(section_probs: Any, types: Sequence[str]) -> dict[str, np.ndarray]:
    """doc_id -> (T,) section_idx of the section with the highest probability for each type."""
    cols = [f"prob_{t}" for t in types]
    out = {}
    for d, g in section_probs.groupby("doc_id", sort=False):
        P = g[cols].to_numpy()
        out[d] = g["section_idx"].to_numpy()[P.argmax(0)]
    return out


class EvidenceStore:
    """Precomputed evidence (distil stage sentence_evidence): per type, doc_id -> its best sentence
    (section_idx, sentence_idx, p, text), kept only when p reaches the type's threshold."""

    def __init__(self, by_type: dict[str, dict[str, tuple[int, int, float, str]]]):
        self.by_type = by_type

    @classmethod
    def load(cls, folder: str | Path, thresholds: dict[str, float]) -> "EvidenceStore":
        import pandas as pd

        parts = sorted(Path(folder).glob("part-*.parquet"))
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame(
            columns=["doc_id", "type", "section_idx", "sentence_idx", "p", "text"])
        df = df[df["p"] >= df["type"].map(thresholds).fillna(1.1)]
        by_type: dict[str, dict[str, tuple[int, int, float, str]]] = {}
        for t, g in df.groupby("type", sort=False):
            by_type[t] = dict(zip(g["doc_id"], zip(g["section_idx"].astype(int), g["sentence_idx"].astype(int),
                                                   g["p"].astype(float), g["text"])))
        return cls(by_type)

    def get(self, type_id: str, doc_id: str) -> tuple[int, int, float, str] | None:
        return self.by_type.get(type_id, {}).get(doc_id)


class PeerEvidence:
    """sentence_score(groups, t) -> one array of P(sentence states type t) per group of sentences; called once per
    type with every candidate peer's sentences (in the pipeline: ColBERT pre-ranks, the presence model scores)."""

    def __init__(self, best: dict[str, np.ndarray], section_text: Callable[[str, int], str | None],
                 sentence_score: Callable[[list[list[str]], int], list[np.ndarray]] | None,
                 prototypes: dict[str, list[dict[str, Any]]], types: Sequence[str], *, store: EvidenceStore | None = None):
        self.best, self.section_text, self.sentence_score = best, section_text, sentence_score
        self.prototypes, self.types, self.store = prototypes, list(types), store

    def for_type(self, type_id: str, peers: Sequence[str], *, k: int = 5, min_prob: float | None = None,
                 candidates: int | None = None) -> list[dict[str, Any]]:
        """Up to k quotes for type_id from the peers that state it (in the order given, e.g. most confident first):
        from each peer's best section, its best sentence, kept only when its probability reaches min_prob. At most
        `candidates` peers (default 2k) are read."""
        if self.store is not None:
            return self._from_store(type_id, peers, k)
        from data.segment import split_sentences

        t = self.types.index(type_id)
        cand: list[tuple[str, int, list[str]]] = []
        for d in peers:
            if len(cand) >= (candidates or 2 * k):
                break
            if d not in self.best:
                continue
            idx = int(self.best[d][t])
            text = self.section_text(d, idx)
            sents = split_sentences(text) if text else []
            if sents:
                cand.append((d, idx, sents))
        out: list[dict[str, Any]] = []
        if cand:
            scores = self.sentence_score([c[2] for c in cand], t) if self.sentence_score else [np.ones(len(c[2])) for c in cand]
            for (d, idx, sents), sc in zip(cand, scores):
                j = int(np.argmax(sc))
                if min_prob is not None and sc[j] < min_prob:
                    continue
                out.append({"doc_id": d, "section_idx": idx, "text": sents[j], "source": "cohort", "p": round(float(sc[j]), 4)})
                if len(out) >= k:
                    break
        return self._fill(type_id, out, k)

    def _from_store(self, type_id: str, peers: Sequence[str], k: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for d in peers:
            if len(out) >= k:
                break
            hit = self.store.get(type_id, d)
            if hit is None or hit[3] in seen:  # shared boilerplate: one quote, not five copies
                continue
            seen.add(hit[3])
            out.append({"doc_id": d, "section_idx": hit[0], "sentence_idx": hit[1], "text": hit[3], "source": "cohort",
                        "p": round(hit[2], 4)})
        return self._fill(type_id, out, k)

    def _fill(self, type_id: str, out: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        for p in self.prototypes.get(type_id, []):
            if len(out) >= k:
                break
            out.append({"doc_id": p["doc_id"], "section_idx": p.get("section_idx"), "text": p["text"], "source": "prototype"})
        return out
