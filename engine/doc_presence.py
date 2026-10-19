"""Document-level presence (Phase 5 first job, 2026-10-19): P(type t is stated somewhere in document d) from the
presence probabilities of every section of d.

The v1 profiles (packs/profiles.py) mark t present when the best section passes a threshold tuned for section-level
F1; across ~10-20 sections small per-section false-positive rates compound, so presence is inflated (papers:
`states_own_limitations` in 96.6% of documents). Here a document-level model is fitted on whole labelled documents
(a document states t when some labelled section does): features of the section probabilities per (document, type),
shared slopes across types and a regularised intercept per type (partial pooling: about 75 fitting documents per
genre cannot support a free model per type), C chosen by cross-validation grouped by document. Its output is a
calibrated probability, which the omission score of §0.2 #1 needs as P(present in d). No genre-specific code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_EPS = 1e-4
FEATURES = ("logit_max", "logit_second", "log_sum", "log_n")
ALL_FEATURES = FEATURES + ("logit_noisy_or",)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def doc_features(P: np.ndarray, names: Sequence[str] = FEATURES) -> np.ndarray:
    """(n_sections, T) section probabilities -> (T, len(names)) document features."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or len(P) == 0:
        raise ValueError("a document needs at least one section")
    srt = -np.sort(-P, axis=0)
    second = srt[1] if len(P) > 1 else np.full(P.shape[1], _EPS)
    cols = {"logit_max": _logit(srt[0]), "logit_second": _logit(second), "log_sum": np.log(P.sum(0) + _EPS),
            "log_n": np.full(P.shape[1], np.log(len(P))),
            "logit_noisy_or": _logit(1 - np.exp(np.log1p(-np.clip(P, 0, 1 - _EPS)).sum(0)))}
    return np.stack([cols[n] for n in names], axis=1)


def split_documents(ids: Sequence[str], *, seed: int, fit_share: float = 0.5) -> tuple[list[str], list[str]]:
    """Whole documents -> (fit, eval), seeded and independent of input order."""
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    k = int(round(fit_share * len(ids)))
    return sorted(ids[i] for i in order[:k]), sorted(ids[i] for i in order[k:])


class DocPresence:
    """Shared-slope logistic model over document features with a per-type intercept (one-hot, L2-shrunk)."""

    def __init__(self, features: Sequence[str] = FEATURES):
        self.features = tuple(features)
        self.T = 0
        self.mean = self.std = self.coef = None  # type: ignore[assignment]
        self.intercept = 0.0
        self.C = None

    def _design(self, F: np.ndarray) -> np.ndarray:
        """F: (n_docs, T, n_features) -> (n_docs * T, n_features + T)."""
        n = F.shape[0]
        Z = ((F - self.mean) / self.std).reshape(n * self.T, -1)
        onehot = np.tile(np.eye(self.T), (n, 1))
        return np.hstack([Z, onehot])

    def _feats(self, docs: Sequence[np.ndarray]) -> np.ndarray:
        idx = [ALL_FEATURES.index(f) for f in self.features]
        return np.stack([doc_features(P, ALL_FEATURES)[:, idx] for P in docs])

    def fit(self, docs: Sequence[np.ndarray], Y: np.ndarray, *, C_grid: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0),
            folds: int = 5, seed: int = 0) -> "DocPresence":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss

        Y = np.asarray(Y, dtype=int)
        F = self._feats(docs)
        self.T = F.shape[1]
        self.mean, self.std = F.reshape(-1, F.shape[2]).mean(0), F.reshape(-1, F.shape[2]).std(0) + 1e-6
        X, y = self._design(F), Y.reshape(-1)
        fold = np.random.default_rng(seed).permutation(len(docs)) % folds
        row_fold = np.repeat(fold, self.T)
        best = None
        for C in C_grid:  # cross-validation grouped by document
            losses = []
            for k in range(folds):
                tr, te = row_fold != k, row_fold == k
                if len(set(y[tr])) < 2 or not te.any():
                    continue
                m = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
                losses.append(log_loss(y[te], m.predict_proba(X[te])[:, 1], labels=[0, 1]))
            score = float(np.mean(losses)) if losses else np.inf
            if best is None or score < best[0]:
                best = (score, C)
        self.C, self.cv_log_loss = best[1], best[0]
        m = LogisticRegression(C=self.C, max_iter=2000).fit(X, y)
        self.coef, self.intercept = m.coef_[0], float(m.intercept_[0])
        return self

    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        """(n_sections, T) section probabilities of one document -> (T,) P(type stated in the document)."""
        P = np.asarray(P)
        if P.ndim != 2 or P.shape[1] != self.T:
            raise ValueError(f"expected section probabilities over {self.T} types, got shape {P.shape}")
        X = self._design(self._feats([P]))
        return 1 / (1 + np.exp(-(X @ self.coef + self.intercept)))

    def save(self, path: Path, **meta: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"features": self.features, "T": self.T, "mean": self.mean.tolist(),
                                          "std": self.std.tolist(), "coef": self.coef.tolist(),
                                          "intercept": self.intercept, "C": self.C, **meta}, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> "DocPresence":
        d = json.loads(Path(path).read_text())
        m = cls(d["features"])
        m.T, m.C, m.intercept = d["T"], d["C"], d["intercept"]
        m.mean, m.std, m.coef = np.array(d["mean"]), np.array(d["std"]), np.array(d["coef"])
        return m


def evaluate(Y: np.ndarray, P: np.ndarray, B: np.ndarray, *, min_positives: int = 5, bins: int = 10) -> dict[str, Any]:
    """Document-level scores: micro / macro F1 of the bits B (macro over types with >= min_positives labelled
    documents), ECE and Brier of the probabilities P, and the mean absolute error of each type's prevalence (share of
    documents with the bit set, and mean probability) against the labelled prevalence."""
    Y, P, B = np.asarray(Y, dtype=int), np.asarray(P, dtype=float), np.asarray(B, dtype=int)
    tp, fp, fn = (B & Y).sum(0), (B & (1 - Y)).sum(0), ((1 - B) & Y).sum(0)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(1, 2 * tp + fp + fn), np.nan)
    keep = Y.sum(0) >= min_positives
    p, y = P.reshape(-1), Y.reshape(-1)
    edges = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(p[edges == b].mean() - y[edges == b].mean()) * (edges == b).mean() for b in range(bins) if (edges == b).any())
    return {"documents": int(len(Y)), "micro_f1": round(float(2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum())), 4),
            "macro_f1": round(float(np.nanmean(f1[keep])), 4) if keep.any() else None, "macro_types": int(keep.sum()),
            "ece": round(float(ece), 4), "brier": round(float(((p - y) ** 2).mean()), 4),
            "prevalence_abs_error": round(float(np.abs(B.mean(0) - Y.mean(0)).mean()), 4),
            "prob_prevalence_abs_error": round(float(np.abs(P.mean(0) - Y.mean(0)).mean()), 4),
            "types_present_per_document": round(float(B.sum(1).mean()), 2),
            "labelled_types_per_document": round(float(Y.sum(1).mean()), 2)}
