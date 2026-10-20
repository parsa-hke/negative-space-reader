"""bench/probs.py: the bench's order of test documents and every test sentence's presence probabilities (plan 6.1),
stored outside the pack so that the sealed test split can never reach the index."""

import numpy as np
import pytest

from bench.probs import bench_order, build_sentence_probs, check_outside_packs, read_sentence_probs


def test_bench_order_is_seeded_and_independent_of_input_order():
    ids = [f"d{i}" for i in range(50)]
    a, b = bench_order(ids, seed=7), bench_order(list(reversed(ids)), seed=7)
    assert a == b and sorted(a) == sorted(ids) and a != sorted(ids)
    assert bench_order(ids, seed=8) != a


def test_sentence_probabilities_are_stored_per_sentence_and_resume(tmp_path):
    docs = [{"id": f"d{i}"} for i in range(5)]
    sents = {f"d{i}": [(0, 0, f"s {i} a"), (0, 1, f"s {i} b"), (1, 0, f"s {i} c")] for i in range(5)}
    calls = []

    def score(texts):
        calls.append(len(texts))
        return np.array([[float(t.endswith("a")), 0.25] for t in texts])

    out = tmp_path / "sentence_probs"
    m = build_sentence_probs(docs, lambda d: sents[d], score, ["x", "y"], out, chunk_documents=2)
    assert m["parts_written"] == 3 and sum(calls) == 15 and m["sentences_scored"] == 15
    calls.clear()
    assert build_sentence_probs(docs, lambda d: sents[d], score, ["x", "y"], out, chunk_documents=2)["parts_written"] == 0
    assert calls == []
    df = read_sentence_probs(out)
    assert list(df.columns[:3]) == ["doc_id", "section_idx", "sentence_idx"] and len(df) == 15
    assert df["prob_x"].tolist() == [1.0, 0.0, 0.0] * 5 and df["prob_y"].round(3).tolist() == [0.25] * 15
    assert df["prob_x"].dtype == np.float16  # 685k test sentences x ~50 types stay small


def test_outputs_may_not_go_into_a_pack(tmp_path):
    packs = tmp_path / "packs"
    with pytest.raises(ValueError, match="pack"):
        check_outside_packs(packs / "privacy" / "test_probs", packs)
    check_outside_packs(tmp_path / "runs" / "bench_probs_s0" / "privacy", packs)


def test_presence_model_refuses_the_test_split_unless_asked():
    from experiments.distil import wanted_documents

    split = {"a": "train", "b": "dev", "c": "test"}
    assert wanted_documents(split, ["train", "dev"]) == {"a", "b"}
    with pytest.raises(ValueError, match="sealed test split"):
        wanted_documents(split, ["test"])
    assert wanted_documents(split, ["test"], allow_test=True) == {"c"}
