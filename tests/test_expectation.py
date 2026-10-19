"""engine/expectation.py (plan 5.4): level-1 expectation = prevalence over the cohort's profiles, with Wilson intervals."""

import numpy as np
import pandas as pd

from engine.expectation import ProfileStore, level1, wilson_interval


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(np.array([0, 5, 10]), 10, z=1.96)
    assert np.allclose(lo, [0.0, 0.2366, 0.7225], atol=1e-3) and np.allclose(hi, [0.2775, 0.7634, 1.0], atol=1e-3)
    lo, hi = wilson_interval(np.array([3]), 0)  # empty cohort: no information
    assert lo[0] == 0.0 and hi[0] == 1.0


def test_small_cohorts_get_wider_intervals():
    small = wilson_interval(np.array([8]), 10)
    big = wilson_interval(np.array([160]), 200)
    assert (small[1] - small[0])[0] > 2 * (big[1] - big[0])[0]


def test_profile_store_and_level1(tmp_path):
    types = ["a", "b"]
    df = pd.DataFrame({"doc_id": [f"d{i}" for i in range(6)], "split": "train",
                       "prob_a": [0.9, 0.8, 0.7, 0.1, 0.2, 0.6], "prob_b": [0.1] * 6,
                       "bit_a": [1, 1, 1, 0, 0, 1], "bit_b": [0] * 6})
    (tmp_path / "profiles").mkdir()
    df.to_parquet(tmp_path / "profiles" / "part-00000.parquet", index=False)
    store = ProfileStore.load(tmp_path / "profiles", types)
    e = level1(store, ["d0", "d1", "d3", "d5"], z=1.96)
    assert e["n"] == 4 and np.allclose(e["prevalence"], [0.75, 0.0]) and e["count"].tolist() == [3, 0]
    assert np.all(e["ci_low"] <= e["prevalence"]) and np.all(e["prevalence"] <= e["ci_high"])
    soft = level1(store, ["d0", "d1", "d3", "d5"], use="prob")  # mean probability: the expected share
    assert np.allclose(soft["prevalence"], [0.6, 0.1])
    w = level1(store, ["d0", "d3"], weights=np.array([3.0, 1.0]))
    assert np.allclose(w["prevalence"], [0.75, 0.0])
    missing = level1(store, ["d0", "zzz"])
    assert missing["n"] == 1 and missing["missing"] == 1
