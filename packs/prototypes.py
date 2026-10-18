"""Prototype sentences per assertion type (plan 4.4): the evidence that the cascade matches sections against.

Candidates are the teacher's verbatim evidence spans for each type on training sections only (labelled positives;
annotation labels carry no spans), unique after normalising case and spacing. They are encoded with the
late-interaction model (engine/late_interaction.py, as queries), clustered with k-means over their mean-pooled token
vectors, and the candidate nearest each centroid is kept: n_prototypes(n) = clip(round(2 sqrt(n)), k_min, k_max), the
plan's 10-30, or every candidate when there are fewer. Each prototype keeps its document id and section, so it doubles
as peer evidence (plan §0.2 #14).
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np

_WS = re.compile(r"\s+")


def n_prototypes(n: int, k_min: int, k_max: int) -> int:
    if n <= k_min:
        return n
    return int(min(k_max, max(k_min, round(2 * math.sqrt(n)))))


def select_prototypes(vectors: np.ndarray, *, k: int, seed: int) -> list[int]:
    """Indices of the members nearest each of k k-means centroids (all indices when there are at most k)."""
    from sklearn.cluster import KMeans

    n = len(vectors)
    if n <= k:
        return list(range(n))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(vectors)
    picked = []
    for c in range(k):
        members = np.flatnonzero(km.labels_ == c)
        d = np.linalg.norm(vectors[members] - km.cluster_centers_[c], axis=1)
        picked.append(int(members[np.argmin(d)]))
    return sorted(picked)


def span_candidates(rows: dict[tuple[str, int], dict[str, Any]], types: Iterable[str],
                    keys: set[tuple[str, int]]) -> dict[str, list[dict[str, Any]]]:
    """Grounded evidence spans per type from the given (training) sections, unique by normalised text."""
    out: dict[str, list[dict[str, Any]]] = {t: [] for t in types}
    seen: dict[str, set[str]] = {t: set() for t in out}
    for key in sorted(k for k in rows if k in keys):
        r = rows[key]
        for t, span in (r.get("evidence") or {}).items():
            if t not in out or not (r.get("grounded") or {}).get(t) or not span:
                continue
            norm = _WS.sub(" ", span).strip().lower()
            if norm in seen[t]:
                continue
            seen[t].add(norm)
            out[t].append({"text": span.strip(), "doc_id": key[0], "section_idx": key[1]})
    return out
