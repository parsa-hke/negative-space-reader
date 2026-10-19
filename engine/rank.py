"""Omission scoring, thresholds and top-N (plan §0.2 #1, #13; §5.5).

score(t) = P(t expected | d, C) x (1 - P(t present in d)) x w(t), for the types not present in d; w is the schema's
severity weight when rank.use_severity_weights is on (default 1). Ties are broken by cohort prevalence. Output: the
top_k (default 10) or every type above a threshold.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def rank_absences(types: Sequence[str], expected: np.ndarray, p_present: np.ndarray, present_bits: np.ndarray, *,
                  weights: np.ndarray | None = None, prevalence: np.ndarray | None = None, top_k: int | None = 10,
                  threshold: float | None = None) -> list[dict[str, Any]]:
    expected, p_present = np.asarray(expected, float), np.asarray(p_present, float)
    w = np.ones(len(types)) if weights is None else np.asarray(weights, float)
    prev = expected if prevalence is None else np.asarray(prevalence, float)
    score = expected * (1 - p_present) * w
    rows = [{"type": t, "score": float(score[k]), "expected": float(expected[k]), "p_present": float(p_present[k]),
             "weight": float(w[k]), "prevalence": float(prev[k])}
            for k, t in enumerate(types) if not present_bits[k]]
    rows.sort(key=lambda r: (-r["score"], -r["prevalence"], r["type"]))
    if threshold is not None:
        rows = [r for r in rows if r["score"] >= threshold]
    return rows[:top_k] if top_k else rows
