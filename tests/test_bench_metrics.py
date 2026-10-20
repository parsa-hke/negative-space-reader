"""bench/metrics.py: ranking metrics, false alarms, ECE and bootstrap intervals over documents (plan 6.2)."""

import math

import numpy as np
import pytest

from bench.metrics import (bootstrap, bootstrap_mean, ece, false_alarm_rate, instance_metrics, ndcg_at_k,
                           paired_bootstrap, recall_at_k, reciprocal_rank)

RANKED = ["x", "y", "z", "w"]
GT = {"y", "w"}


def test_ranking_metrics_by_hand():
    assert recall_at_k(RANKED, GT, 1) == 0 and recall_at_k(RANKED, GT, 2) == 0.5 and recall_at_k(RANKED, GT, 10) == 1
    assert reciprocal_rank(RANKED, GT) == 0.5 and reciprocal_rank(RANKED, {"q"}) == 0
    want = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, GT, 10) == pytest.approx(want)
    assert ndcg_at_k(["y", "w", "x"], GT, 10) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, GT, 1) == 0.0
    m = instance_metrics(RANKED, GT, ks=(1, 5), ndcg_k=10)
    assert m == {"recall@1": 0.0, "recall@5": 1.0, "mrr": 0.5, "ndcg@10": pytest.approx(want), "rank_first": 2}
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 5)


def test_false_alarms_are_the_share_of_the_top_k_that_is_present():
    assert false_alarm_rate(RANKED, {"x", "z"}, 2) == 0.5
    assert false_alarm_rate(RANKED, set(), 5) == 0.0
    assert false_alarm_rate(["x"], {"x"}, 5) == 1.0  # fewer absences than k: share of those reported
    assert false_alarm_rate([], {"x"}, 5) is None


def test_ece_by_hand():
    p, y = np.array([0.1, 0.1, 0.9, 0.9]), np.array([0, 1, 1, 1])
    assert ece(p, y, bins=10) == pytest.approx(0.4 * 0.5 + 0.1 * 0.5)
    assert ece(np.array([1.0, 0.0]), np.array([1, 0]), bins=15) == pytest.approx(0.0)


def test_bootstrap_intervals_are_seeded_and_sensible():
    rng = np.random.default_rng(0)
    x = rng.normal(0.4, 0.1, size=400)
    mean, lo, hi = bootstrap_mean(x, n_boot=500, ci=0.95, seed=1)
    assert lo < mean < hi and mean == pytest.approx(x.mean()) and hi - lo == pytest.approx(4 * 0.1 / 20, rel=0.25)
    assert bootstrap_mean(x, n_boot=500, seed=1) == (mean, lo, hi)
    assert bootstrap_mean(np.full(10, 0.3), n_boot=50) == pytest.approx((0.3, 0.3, 0.3))
    # any statistic of resampled documents, e.g. ECE over (document, type) rows
    P, Y = rng.uniform(size=(50, 4)), rng.integers(0, 2, size=(50, 4))
    v, lo, hi = bootstrap(lambda idx: ece(P[idx].ravel(), Y[idx].ravel(), bins=10), 50, n_boot=200, seed=0)
    assert lo <= v <= hi and v == pytest.approx(ece(P.ravel(), Y.ravel(), bins=10))


def test_paired_bootstrap_of_a_difference():
    rng = np.random.default_rng(0)
    b = rng.uniform(size=300)
    d = paired_bootstrap(b + 0.1, b, n_boot=300, seed=0)
    assert d["diff"] == pytest.approx(0.1) and d["ci_low"] == pytest.approx(0.1) and d["ci_high"] == pytest.approx(0.1)
    noise = paired_bootstrap(rng.uniform(size=300), rng.uniform(size=300), n_boot=300, seed=0)
    assert noise["ci_low"] < 0 < noise["ci_high"]
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(4))


def test_grouped_bootstrap_resamples_whole_groups():
    # 20 entities with 10 identical-valued instances each: resampling instances understates the uncertainty
    groups = np.repeat(np.arange(20), 10)
    x = np.repeat(np.random.default_rng(0).uniform(size=20), 10)
    _, lo_i, hi_i = bootstrap_mean(x, n_boot=400, seed=0)
    v, lo_g, hi_g = bootstrap_mean(x, n_boot=400, seed=0, groups=groups)
    assert v == pytest.approx(x.mean()) and (hi_g - lo_g) > 2 * (hi_i - lo_i)
    d = paired_bootstrap(x + 0.05, x, n_boot=100, seed=0, groups=groups)
    assert d["diff"] == pytest.approx(0.05) and d["ci_low"] == pytest.approx(0.05)
