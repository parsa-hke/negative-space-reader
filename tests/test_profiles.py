"""packs/profiles.py: document profiles from section probabilities, in resumable chunks (plan 4.7)."""

import numpy as np
import pandas as pd

from packs.profiles import aggregate, build_profiles, read_profiles


def test_aggregate_takes_the_best_section_and_thresholds():
    P = np.array([[0.1, 0.7], [0.6, 0.2], [0.3, 0.3]])
    prob, bits = aggregate(P, thresholds=np.array([0.5, 0.8]))
    assert np.allclose(prob, [0.6, 0.7]) and bits.tolist() == [1, 0]


def test_build_profiles_resumes_by_chunk(tmp_path):
    docs = [{"id": f"d{i}", "split": "train", "category": "c", "date": "2020-01-01"} for i in range(7)]
    sections = {f"d{i}": [f"text {i} a", f"text {i} b"] for i in range(7)}
    calls = []

    def score(texts):
        calls.append(len(texts))
        return np.array([[float(t.endswith("a")), 0.2] for t in texts])

    types, th = ["x", "y"], np.array([0.5, 0.5])
    out = tmp_path / "profiles"
    m = build_profiles(docs, lambda d: sections[d], score, types, th, out, chunk_documents=3)
    assert m["parts_written"] == 3 and sum(calls) == 14
    calls.clear()
    m = build_profiles(docs, lambda d: sections[d], score, types, th, out, chunk_documents=3)
    assert m["parts_written"] == 0 and calls == []  # every chunk already done
    df = read_profiles(out)
    assert list(df["doc_id"]) == [f"d{i}" for i in range(7)] and df["bit_x"].tolist() == [1] * 7
    assert df["prob_y"].round(2).tolist() == [0.2] * 7 and df["sections"].tolist() == [2] * 7
    assert isinstance(df, pd.DataFrame)
