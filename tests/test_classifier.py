"""engine/classifier.py: per-type logistic heads, temperature scaling, thresholds, metrics (plan 4.3)."""

import numpy as np

from engine.classifier import LinearHeads, ece, prf_table, reference_split


def toy(n=400, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    ya = (X[:, 0] > 0.3).astype(int)          # learnable from feature 0
    yb = (X[:, 1] + 0.3 * rng.normal(size=n) > 1.2).astype(int)  # rarer, noisy
    return X, {"a": ya, "b": yb, "rare": np.r_[1, np.zeros(n - 1, dtype=int)]}


def test_heads_learn_calibrate_and_round_trip(tmp_path):
    X, Y = toy()
    heads = LinearHeads(["a", "b", "rare", "never"], C=1.0, class_weight="balanced")
    for t in ("a", "b", "rare"):
        heads.fit_type(t, X[:300], Y[t][:300])
    P = heads.predict_proba(X[300:])
    assert P.shape == (100, 4) and np.all((P >= 0) & (P <= 1))
    assert heads.info["rare"]["mode"] == "prior" and heads.info["never"]["mode"] == "prior"  # too few positives
    acc = ((P[:, 0] > 0.5) == Y["a"][300:]).mean()
    assert acc > 0.9
    Yfit = np.stack([Y[t][300:] for t in ("a", "b", "rare")] + [np.zeros(100, int)], axis=1)
    heads.calibrate(X[300:], Yfit)
    assert heads.temperature.shape == (4,) and heads.temperature[3] == 1.0  # no positives: left alone
    heads.tune_thresholds(X[300:], Yfit)
    assert 0 < heads.threshold[0] < 1
    heads.save(tmp_path / "h")
    again = LinearHeads.load(tmp_path / "h")
    assert np.allclose(again.predict_proba(X[:5]), heads.predict_proba(X[:5]))
    assert again.types == heads.types and np.allclose(again.threshold, heads.threshold)


def test_prf_table_and_ece():
    Y = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    P = np.array([[0.9, 0.2], [0.4, 0.1], [0.6, 0.8], [0.1, 0.1]])
    rows = prf_table(Y, P, ["a", "b"], thresholds=np.array([0.5, 0.5]))
    a, b = rows
    assert (a["tp"], a["fp"], a["fn"]) == (1, 1, 1) and a["f1"] == 0.5
    assert (b["tp"], b["fp"], b["fn"], b["f1"]) == (1, 0, 0, 1.0)
    assert 0 <= ece(Y, P) <= 1 and ece(np.array([[1], [0]]), np.array([[1.0], [0.0]])) == 0.0


def test_reference_split_is_by_document_and_balanced():
    keys = [(f"d{i // 3}", i % 3) for i in range(60)]  # 20 documents, 3 sections each
    split = reference_split(keys, seed=1234)
    docs = {}
    for (d, _), s in split.items():
        docs.setdefault(d, set()).add(s)
    assert all(len(v) == 1 for v in docs.values())  # a document is wholly in one half
    halves = [sum(1 for v in docs.values() if s in v) for s in ("fit", "eval")]
    assert halves == [10, 10] and reference_split(keys, seed=1234) == split
