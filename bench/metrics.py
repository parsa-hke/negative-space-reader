"""recall@k, MRR, nDCG, false-alarm rate, ECE, bootstrap CIs over documents (plan 6.2, §0.2 #15).

A ranking is the list of a document's absent types, best first; ground truth is the set of types deleted from it.
recall@k = |ground truth in the top k| / |ground truth|; MRR from the first ground-truth type; nDCG@k with binary
gains. The false-alarm rate of an untouched document is the share of its top-k reported absences that the reference
labels as present (the document-level silver labels of decision record 006). ECE uses equal-width bins. Bootstrap
intervals resample documents (one instance per document and regime, so documents and instances coincide); the
gate (6.6) uses a paired bootstrap of the per-document difference between two systems.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np


def _check(gt: set[str]) -> None:
    if not gt:
        raise ValueError("an instance needs at least one ground-truth type")


def recall_at_k(ranked: Sequence[str], gt: set[str], k: int) -> float:
    _check(gt)
    return len(gt & set(ranked[:k])) / len(gt)


def reciprocal_rank(ranked: Sequence[str], gt: set[str]) -> float:
    return next((1 / (i + 1) for i, t in enumerate(ranked) if t in gt), 0.0)


def ndcg_at_k(ranked: Sequence[str], gt: set[str], k: int) -> float:
    _check(gt)
    dcg = sum(1 / math.log2(i + 2) for i, t in enumerate(ranked[:k]) if t in gt)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gt))))
    return dcg / ideal


def instance_metrics(ranked: Sequence[str], gt: set[str], *, ks: Sequence[int] = (1, 5, 10),
                     ndcg_k: int = 10) -> dict[str, Any]:
    first = next((i + 1 for i, t in enumerate(ranked) if t in gt), None)
    return {**{f"recall@{k}": recall_at_k(ranked, gt, k) for k in ks}, "mrr": reciprocal_rank(ranked, gt),
            f"ndcg@{ndcg_k}": ndcg_at_k(ranked, gt, ndcg_k), "rank_first": first}


def false_alarm_rate(ranked: Sequence[str], present: set[str], k: int) -> float | None:
    """Share of the top-k reported absences that are in fact present; None when nothing is reported."""
    top = list(ranked[:k])
    return sum(t in present for t in top) / len(top) if top else None


def ece(p: np.ndarray, y: np.ndarray, *, bins: int = 15) -> float:
    p, y = np.asarray(p, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[b == i].mean() - y[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))


def within_document_auc(E: np.ndarray, Y: np.ndarray) -> float | None:
    """Mean over documents of the AUC with which an expectation E (documents x types) ranks a document's present
    types (Y = 1) above its absent ones; documents with no present or no absent type are skipped (None if all are).
    The label-free check of how much a cohort knows about the documents it is retrieved for."""
    out = []
    for e, y in zip(np.asarray(E, dtype=float), np.asarray(Y, dtype=int)):
        pos, neg = e[y == 1], e[y == 0]
        if len(pos) and len(neg):
            out.append(float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    return float(np.mean(out)) if out else None


def _resample(n: int, rng: np.random.Generator, groups: Sequence[Any] | None) -> np.ndarray:
    """Row indices of one bootstrap resample: of the rows, or of whole groups (e.g. every version of an entity)."""
    if groups is None:
        return rng.integers(0, n, size=n)
    g = np.asarray(groups)
    members = [np.flatnonzero(g == u) for u in np.unique(g)]
    pick = rng.integers(0, len(members), size=len(members))
    return np.concatenate([members[i] for i in pick])


def bootstrap(stat: Callable[[np.ndarray], float], n: int, *, n_boot: int = 1000, ci: float = 0.95,
              seed: int = 0, groups: Sequence[Any] | None = None) -> tuple[float, float, float]:
    """(statistic on all n documents, percentile interval over n_boot resamples of the documents, or of whole
    groups of rows when groups is given)."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(_resample(n, rng, groups)) for _ in range(n_boot)])
    a = (1 - ci) / 2
    return float(stat(np.arange(n))), float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def bootstrap_mean(x: Sequence[float], **kw: Any) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    return bootstrap(lambda idx: float(x[idx].mean()), len(x), **kw)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, n_boot: int = 1000, ci: float = 0.95, seed: int = 0,
                     groups: Sequence[Any] | None = None) -> dict[str, float]:
    """Mean of a - b over the same documents, its interval, and the share of resamples with a difference <= 0."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired values differ in shape: {a.shape} vs {b.shape}")
    d = a - b
    rng = np.random.default_rng(seed)
    means = np.array([d[_resample(len(d), rng, groups)].mean() for _ in range(n_boot)])
    q = (1 - ci) / 2
    return {"diff": float(d.mean()), "ci_low": float(np.quantile(means, q)), "ci_high": float(np.quantile(means, 1 - q)),
            "p_not_better": float((means <= 0).mean()), "n": int(len(d))}
