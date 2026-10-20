"""OmitBench-revision: revision-added types judged against a time-windowed cohort (plan 6.3b, §0.2 #16, #19).

For a version pair (n, n+1) of one entity, the added types are those present in n+1 and absent in n (document-level
presence, bits at 0.5). The pair is kept when at least one added type has prevalence >= min_prevalence among the
peers of version n dated before it (the plan's rule: the cohort's own prevalence); the instance is version n with
the qualifying added types as ground truth. min_prevalence=None keeps every added type (a system-neutral variant:
the plan's filter selects with the level-1 quantity, which favours level 1).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def added_types(p_old: np.ndarray, p_new: np.ndarray, *, threshold: float = 0.5) -> list[int]:
    return np.flatnonzero((np.asarray(p_new) >= threshold) & (np.asarray(p_old) < threshold)).tolist()


def revision_instances(pairs: Sequence[dict[str, Any]], p_doc: dict[str, np.ndarray],
                       expected: Callable[[dict[str, Any]], np.ndarray], types: Sequence[str], *,
                       min_prevalence: float | None = 0.5) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """pairs: revisions.jsonl rows; p_doc: doc_id -> document presence; expected(pair) -> prevalence of every type
    among version n's time-windowed peers (called only for pairs that add a type)."""
    stats = {"pairs": len(pairs), "missing_version": 0, "no_added_type": 0, "below_prevalence": 0, "instances": 0}
    out = []
    for pr in pairs:
        old, new = pr["old_doc_id"], pr["new_doc_id"]
        if old not in p_doc or new not in p_doc:
            stats["missing_version"] += 1
            continue
        added = added_types(p_doc[old], p_doc[new])
        if not added:
            stats["no_added_type"] += 1
            continue
        prev = np.asarray(expected(pr), dtype=float)
        gt = [k for k in added if min_prevalence is None or prev[k] >= min_prevalence]
        if not gt:
            stats["below_prevalence"] += 1
            continue
        stats["instances"] += 1
        date = pr.get("old_date")
        out.append({"instance_id": f"revision:{old}", "doc_id": old, "new_doc_id": new, "entity_id": pr["entity_id"],
                    "old_date": date, "new_date": pr.get("new_date"), "year": int(date[:4]) if date else None,
                    "added": [types[k] for k in added], "ground_truth": [types[k] for k in gt],
                    "added_prevalence": {types[k]: round(float(prev[k]), 4) for k in added}})
    return out, stats
