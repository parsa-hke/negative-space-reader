"""Presence cascade (plan §0.2 #8, task 4.5), for every type and section:

  1. prototype matching: the best late-interaction match between the section's sentences and the type's prototypes
     (PrototypeScorer; engine/late_interaction.py);
  2. a score below the band (low) is a clear miss, above it (high) a clear hit: both stop here;
  3. the band in between goes to a cross-encoder reranker (Reranker) with the type's definition as the query.
Classifier probability (4.3) and the cascade's scores are combined by a learned logistic layer (Cascade), fitted on
held-out labels, with a per-type threshold. The band is fixed per pack from the prototype-score distributions of
labelled-positive and labelled-negative sections (band_thresholds: their overlap, robust quantiles).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

FEATURES = ["classifier_logit", "prototype", "reranker", "in_band", "above_band"]


def band_thresholds(pos: np.ndarray, neg: np.ndarray, *, q_low: float = 0.05, q_high: float = 0.95) -> tuple[float, float]:
    """(low, high): from the q_low quantile of positives' scores to the q_high quantile of negatives' scores. When they
    do not overlap the band is empty (low = high)."""
    low, high = float(np.quantile(pos, q_low)), float(np.quantile(neg, q_high))
    if high < low:
        mid = (low + high) / 2
        return mid, mid
    return low, high


def rerank_pairs(proto: np.ndarray, *, low: float, high: float) -> list[tuple[int, int]]:
    """(section, type) pairs inside the band [low, high)."""
    rows, cols = np.nonzero((proto >= low) & (proto < high))
    return list(zip(rows.tolist(), cols.tolist()))


def type_query(t: dict[str, Any]) -> str:
    examples = " ".join(t.get("positive_examples", [])[:2])
    return f"{t['definition']} For example: {examples}" if examples else t["definition"]


class PrototypeScorer:
    """Scores sections against every type's prototypes (packs/<genre>/prototypes/tokens.npz)."""

    def __init__(self, colbert: Any, tokens: np.ndarray, lengths: np.ndarray, type_idx: np.ndarray, n_types: int, *,
                 min_sentence_chars: int = 15):
        import torch

        self.colbert, self.n_types, self.min_chars = colbert, n_types, min_sentence_chars
        self.lengths, self.type_idx = np.asarray(lengths), np.asarray(type_idx)
        self.Q = torch.from_numpy(np.asarray(tokens, dtype=np.float32)).to(colbert.device)

    @classmethod
    def from_pack(cls, colbert: Any, path: str | Path, n_types: int, **kw: Any) -> "PrototypeScorer":
        z = np.load(Path(path) / "tokens.npz")
        return cls(colbert, z["tokens"], z["lengths"], z["type_idx"], n_types, **kw)

    def score(self, texts: Sequence[str]) -> np.ndarray:
        from data.segment import split_sentences
        from engine.late_interaction import score_matrix

        sents = [split_sentences(t, min_chars=self.min_chars) or [t] for t in texts]
        flat = [s for ss in sents for s in ss]
        enc = self.colbert.encode(flat, mode="doc")
        out, pos = np.zeros((len(texts), self.n_types), dtype=np.float32), 0
        for i, ss in enumerate(sents):
            out[i] = score_matrix(enc[pos:pos + len(ss)], self.Q, self.lengths, self.type_idx, self.n_types,
                                  device=self.colbert.device)
            pos += len(ss)
        return out


class Reranker:
    """Cross-encoder P(type stated | section) for (type query, section text) pairs; `predict` may be injected."""

    def __init__(self, model_id: str | None = None, *, revision: str | None = None, device: str = "cpu",
                 max_length: int = 512, batch_size: int = 16, predict: Callable[[list[tuple[str, str]]], np.ndarray] | None = None):
        self.model_id, self.batch_size = model_id, batch_size
        if predict is None:
            import torch
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_id, revision=revision, device=device, max_length=max_length,
                                 activation_fn=torch.nn.Sigmoid())
            predict = lambda pairs: np.asarray(model.predict(pairs, batch_size=batch_size,  # noqa: E731
                                                             show_progress_bar=len(pairs) > 2000))
        self._predict = predict

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray(self._predict(pairs), dtype=np.float32) if pairs else np.zeros(0, dtype=np.float32)


class Cascade:
    """Logistic combination of classifier probability and cascade scores, pooled over the pack's types, plus a
    per-type threshold. rr holds reranker probabilities for band pairs and NaN elsewhere."""

    def __init__(self, types: Sequence[str], *, low: float, high: float):
        self.types, self.low, self.high = list(types), float(low), float(high)
        self.coef = np.zeros(len(FEATURES))
        self.intercept = 0.0
        self.threshold = np.full(len(self.types), 0.5)

    def features(self, clf: np.ndarray, proto: np.ndarray, rr: np.ndarray) -> np.ndarray:
        if not (clf.shape == proto.shape == rr.shape) or clf.shape[1] != len(self.types):
            raise ValueError(f"shapes differ or do not match {len(self.types)} types: {clf.shape} {proto.shape} {rr.shape}")
        p = np.clip(clf, 1e-6, 1 - 1e-6)
        in_band = (proto >= self.low) & (proto < self.high)
        return np.stack([np.log(p / (1 - p)), proto, np.where(np.isnan(rr), 0.0, rr), in_band.astype(float),
                         (proto >= self.high).astype(float)], axis=-1).reshape(-1, len(FEATURES))

    def fit(self, clf: np.ndarray, proto: np.ndarray, rr: np.ndarray, Y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        from engine.classifier import tune_thresholds

        F = self.features(clf, proto, rr)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(F, np.asarray(Y).reshape(-1))
        self.coef, self.intercept = m.coef_[0], float(m.intercept_[0])
        self.threshold = tune_thresholds(np.asarray(Y), self.predict(clf, proto, rr, thresholded=False))

    def predict(self, clf: np.ndarray, proto: np.ndarray, rr: np.ndarray, *, thresholded: bool = False) -> np.ndarray:
        z = self.features(clf, proto, rr) @ self.coef + self.intercept
        P = (1 / (1 + np.exp(-z))).reshape(clf.shape)
        return (P >= self.threshold).astype(int) if thresholded else P

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"types": self.types, "low": self.low, "high": self.high, "features": FEATURES,
                                          "coef": self.coef.tolist(), "intercept": self.intercept,
                                          "threshold": self.threshold.tolist()}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Cascade":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        c = cls(d["types"], low=d["low"], high=d["high"])
        c.coef, c.intercept, c.threshold = np.array(d["coef"]), d["intercept"], np.array(d["threshold"])
        return c
