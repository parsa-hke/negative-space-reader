"""Presence classifier, lean head (plan §0.2 #7, task 4.3): one logistic head per type on frozen section embeddings.

    heads = LinearHeads(type_ids, C=1.0, class_weight="balanced")
    heads.fit_type(t, X_t, y_t)          # each type may have its own rows (e.g. human labels for some types,
                                         # teacher labels for others)
    heads.calibrate(X_fit, Y_fit)        # temperature per type on held-out labels: P(present) is a probability
    heads.tune_thresholds(X_fit, Y_fit)  # per-type cut-off that maximises F1 on the same held-out labels
    P = heads.predict_proba(X)           # [n, T]

A type with fewer than `min_positives` training positives gets a constant prior instead of a head. Metrics for 4.3
and 4.6: prf_table (per-type P/R/F1 at thresholds) and ece (expected calibration error over all section-type pairs).
reference_split puts whole documents of the reference set into a fitting half (calibration, thresholds, the cascade's
combination layer) and an evaluation half (ablations, gate), so no number is measured on the data it was tuned on.
`setfit` and `finetune` (presence.head) are separate heads, built only if the linear head misses gate 4.8.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class LinearHeads:
    def __init__(self, types: Sequence[str], *, C: float = 1.0, class_weight: str | None = "balanced",
                 min_positives: int = 3, max_iter: int = 2000):
        self.types = list(types)
        self.C, self.class_weight, self.min_positives, self.max_iter = C, class_weight, min_positives, max_iter
        self.coef: np.ndarray | None = None
        self.intercept = np.zeros(len(self.types))
        self.temperature = np.ones(len(self.types))
        self.threshold = np.full(len(self.types), 0.5)
        self.info: dict[str, dict[str, Any]] = {t: {"mode": "prior", "positives": 0, "rows": 0} for t in self.types}

    def fit_type(self, t: str, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        j = self.types.index(t)
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y).astype(int)
        if self.coef is None:
            self.coef = np.zeros((len(self.types), X.shape[1]))
        pos = int(y.sum())
        self.info[t] = {"positives": pos, "rows": int(len(y))}
        if pos < self.min_positives or pos == len(y):
            prior = (pos + 0.5) / (len(y) + 1.0)
            self.coef[j] = 0.0
            self.intercept[j] = np.log(prior / (1 - prior))
            self.info[t]["mode"] = "prior"
            return
        m = LogisticRegression(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter)
        m.fit(X, y)
        self.coef[j], self.intercept[j] = m.coef_[0], m.intercept_[0]
        self.info[t]["mode"] = "head"

    def logits(self, X: np.ndarray) -> np.ndarray:
        if self.coef is None:  # no type was ever fitted: priors of 0.5
            return np.zeros((len(X), len(self.types)))
        return np.asarray(X, dtype=np.float32) @ self.coef.T + self.intercept

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logits(X) / self.temperature)

    def calibrate(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Temperature per type by minimising the log loss on held-out labels Y [n, T] (types without both classes
        in Y, and prior-only types, keep T = 1)."""
        from scipy.optimize import minimize_scalar

        Z = self.logits(X)
        for j, t in enumerate(self.types):
            y = Y[:, j]
            if self.info[t]["mode"] != "head" or y.min() == y.max():
                continue

            def nll(log_t: float, z: np.ndarray = Z[:, j], y: np.ndarray = y) -> float:
                p = np.clip(_sigmoid(z / np.exp(log_t)), 1e-7, 1 - 1e-7)
                return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

            self.temperature[j] = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))

    def tune_thresholds(self, X: np.ndarray, Y: np.ndarray, P: np.ndarray | None = None) -> None:
        """Per-type cut-off maximising F1 on held-out labels (0.5 when a type has no held-out positive)."""
        P = self.predict_proba(X) if P is None else P
        self.threshold = tune_thresholds(Y, P)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "heads.npz", coef=self.coef if self.coef is not None else np.zeros((len(self.types), 0)),
                 intercept=self.intercept, temperature=self.temperature, threshold=self.threshold)
        (path / "heads.json").write_text(json.dumps({"types": self.types, "C": self.C, "class_weight": self.class_weight,
                                                     "min_positives": self.min_positives, "info": self.info}, indent=1),
                                         encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearHeads":
        path = Path(path)
        meta = json.loads((path / "heads.json").read_text(encoding="utf-8"))
        h = cls(meta["types"], C=meta["C"], class_weight=meta["class_weight"], min_positives=meta["min_positives"])
        z = np.load(path / "heads.npz")
        h.coef = z["coef"] if z["coef"].size else None
        h.intercept, h.temperature, h.threshold, h.info = z["intercept"], z["temperature"], z["threshold"], meta["info"]
        return h


def tune_thresholds(Y: np.ndarray, P: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 37) if grid is None else grid
    out = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if y.sum() == 0:
            continue
        best, best_f = 0.5, -1.0
        for th in grid:
            pred = P[:, j] >= th
            tp = int((pred & (y == 1)).sum())
            f = 2 * tp / max(1, int(pred.sum()) + int(y.sum()))
            if f > best_f:
                best, best_f = th, f
        out[j] = best
    return out


def prf_table(Y: np.ndarray, P: np.ndarray, types: Sequence[str], *, thresholds: np.ndarray) -> list[dict[str, Any]]:
    """Per-type counts and P/R/F1 at the thresholds (None where undefined)."""
    rows = []
    for j, t in enumerate(types):
        pred, y = P[:, j] >= thresholds[j], Y[:, j] == 1
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f = (2 * tp / (2 * tp + fp + fn)) if (tp + fn) else None
        rows.append({"type_id": t, "positives": int(y.sum()), "predicted": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": r, "f1": f, "threshold": round(float(thresholds[j]), 3)})
    return rows


def summary(rows: list[dict[str, Any]], *, min_positives: int = 5) -> dict[str, Any]:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    f1s = [r["f1"] for r in rows if r["positives"] >= min_positives and r["f1"] is not None]
    return {"micro_precision": round(tp / max(1, tp + fp), 3), "micro_recall": round(tp / max(1, tp + fn), 3),
            "micro_f1": round(2 * tp / max(1, 2 * tp + fp + fn), 3),
            "macro_f1": round(float(np.mean(f1s)), 3) if f1s else None, "types_scored": len(f1s)}


def ece(Y: np.ndarray, P: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over all section-type pairs (equal-width probability bins)."""
    y, p = np.asarray(Y, dtype=float).ravel(), np.asarray(P, dtype=float).ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(round(err, 6))


def reference_split(keys: Sequence[tuple[str, int]], *, seed: int) -> dict[tuple[str, int], str]:
    """Whole documents of the reference set into 'fit' and 'eval' halves (by document count), seeded and order-free."""
    docs = sorted({d for d, _ in keys}, key=lambda d: hashlib.sha1(f"{seed}:{d}".encode()).hexdigest())
    fit = set(docs[: len(docs) // 2])
    return {k: ("fit" if k[0] in fit else "eval") for k in keys}
