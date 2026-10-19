"""Level 1 (cohort prevalence, §0.2 #11, §5.4) and level 2 (MAM inference, §8.3) expectation.

Level 1: P(t expected | d) = the share of d's cohort whose profile has t present (`use="bit"`, the plan's bitvector
mean; the counts behind "N% of peers state X"), or the cohort's mean document probability (`use="prob"`, the expected
share under the calibrated document model; an ablation). Each prevalence carries a Wilson score interval, so a small
or fallback cohort does not produce an overconfident expectation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


def wilson_interval(successes: np.ndarray, n: float, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for successes out of n (successes may be fractional: weighted or soft counts)."""
    s = np.asarray(successes, dtype=float)
    if n <= 0:
        return np.zeros_like(s), np.ones_like(s)
    p = s / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(np.maximum(p * (1 - p) / n + z * z / (4 * n * n), 0)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


class ProfileStore:
    """Document profiles of a pack (packs/<genre>/profiles/part-*.parquet) as matrices: bits and probabilities."""

    def __init__(self, doc_ids: Sequence[str], bits: np.ndarray, probs: np.ndarray, types: Sequence[str]):
        self.doc_ids, self.types = list(doc_ids), list(types)
        self.row = {d: i for i, d in enumerate(self.doc_ids)}
        self.bits, self.probs = bits, probs

    @classmethod
    def load(cls, folder: str | Path, types: Sequence[str]) -> "ProfileStore":
        from packs.profiles import read_profiles

        df = read_profiles(Path(folder))
        return cls(df["doc_id"].tolist(), df[[f"bit_{t}" for t in types]].to_numpy(np.int8),
                   df[[f"prob_{t}" for t in types]].to_numpy(np.float32), types)

    def rows(self, doc_ids: Sequence[str]) -> list[int]:
        return [self.row[d] for d in doc_ids if d in self.row]


def level1(store: ProfileStore, peer_ids: Sequence[str], *, z: float = 1.96, use: str = "bit",
           weights: np.ndarray | None = None) -> dict[str, Any]:
    """Prevalence of every type over the peers: prevalence, count (weighted where weights are given), n, Wilson
    interval. Peers without a profile are left out and counted in `missing`."""
    rows = store.rows(peer_ids)
    present = [i for i, d in enumerate(peer_ids) if d in store.row]
    M = (store.bits if use == "bit" else store.probs)[rows].astype(float)
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=float)[present]
    n = float(w.sum())
    count = (w[:, None] * M).sum(0) if len(rows) else np.zeros(len(store.types))
    prevalence = count / n if n else np.zeros(len(store.types))
    lo, hi = wilson_interval(count, n, z)
    return {"types": store.types, "prevalence": prevalence, "count": np.round(count, 6), "n": len(rows),
            "weight": n, "ci_low": lo, "ci_high": hi, "missing": len(peer_ids) - len(rows), "use": use}
