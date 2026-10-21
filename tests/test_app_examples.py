"""app/examples.py (plan 7.1): corpus examples from the test split, benchmark examples from an OmitBench-synthetic
run with the chat-model lists of recorded 7.3 runs, written to and read back from serve.examples_dir."""

import csv
import json

import pytest

from data.schema import Document, Section, write_jsonl


@pytest.fixture
def cfg(tmp_path):
    from experiments.config import load_config

    c = load_config("configs/base.yaml", [])
    processed = tmp_path / "processed" / "g"
    processed.mkdir(parents=True)
    docs = [Document(id=f"t{i}", genre="g", source="s", entity_id=f"te{i}", title=f"Policy {i}", category="c",
                     date="2020-01-01") for i in range(3)]
    write_jsonl(processed / "documents.jsonl", docs)
    write_jsonl(processed / "sections.jsonl", [Section(doc_id=d.id, idx=j, heading=f"H{j}", text=f"Text {j} of {d.id}.")
                                               for d in docs for j in range(6 + int(d.id[1]))])
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "g.json").write_text(json.dumps({"doc_ids": {"train": [], "dev": [], "test": [d.id for d in docs]}}))
    synth = tmp_path / "synth" / "g"
    synth.mkdir(parents=True)
    with open(synth / "instances.jsonl", "w") as f:
        f.write(json.dumps({"doc_id": "t9", "entity_id": "te9", "regime": "single_type", "category": "c", "date": "2019-05-05",
                            "ground_truth": ["cookies"], "targets": ["cookies"]}) + "\n")
    with open(synth / "edited.jsonl", "w") as f:
        f.write(json.dumps({"instance_id": "single_type:t9", "title": "Edited policy", "sections": [
            {"section_idx": 0, "heading": "Sharing", "text": "We share data."}, {"section_idx": 1, "heading": None, "text": "Bye."}]}) + "\n")
    for run, ranked, first in (("opus", "cookies|sharing", "1"), ("qwen", "sharing", "")):
        d = tmp_path / run / "g"
        d.mkdir(parents=True)
        with open(d / "per_instance.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instance_id", "system", "rank_first", "ranked"])
            w.writerow(["single_type:t9", run, first, ranked])
    c["paths"].update(data_processed=str(tmp_path / "processed"), splits=str(tmp_path / "splits"))
    c["baselines"]["synth_run"] = str(tmp_path / "synth" / "{genre}")
    c["run"]["seed"] = 0
    c["genres"] = ["g"]
    c["serve"].update(examples={"g": ["t2"]}, bench_examples={"g": ["single_type:t9"]}, examples_dir=str(tmp_path / "ex"),
                      chat_runs=[{"run": str(tmp_path / "opus"), "label": "Opus"}, {"run": str(tmp_path / "qwen"), "label": "Qwen"}])
    return c


def test_corpus_examples_named_or_drawn(cfg):
    from app.examples import corpus_examples

    named = corpus_examples(cfg, "g", ["t2"])
    assert [e["doc_id"] for e in named] == ["t2"] and named[0]["kind"] == "corpus" and "H0" in named[0]["markdown"]
    assert named[0]["entity_id"] == "te2" and named[0]["category"] == "c"
    drawn = corpus_examples(cfg, "g", None, per_genre=2)
    assert len(drawn) == 2 and all(e["pack"] == "g" for e in drawn)
    with pytest.raises(KeyError):
        corpus_examples(cfg, "g", ["nope"])


def test_benchmark_examples_carry_the_removed_types_and_the_chat_lists(cfg):
    from app.examples import benchmark_examples
    from engine.pipeline import segment

    ex = benchmark_examples(cfg, "g", ["single_type:t9"], cfg["serve"]["chat_runs"])
    e = ex[0]
    assert e["kind"] == "benchmark" and e["removed"] == ["cookies"] and e["entity_id"] == "te9" and e["date"] == "2019-05-05"
    assert e["chat"] == {"Opus": {"ranked": ["cookies", "sharing"], "rank_first": 1}, "Qwen": {"ranked": ["sharing"], "rank_first": None}}
    assert [h for h, _ in segment(e["markdown"], cfg["segment"])] == ["Sharing", None]  # the edited sections, as stored
    with pytest.raises(KeyError):
        benchmark_examples(cfg, "g", ["single_type:t8"], [])


def test_write_then_read_and_the_app_loader_prefers_the_files(cfg):
    from app.examples import load_examples, read_examples, write_examples

    assert read_examples(cfg["serve"]["examples_dir"], ["g"]) is None
    assert write_examples(cfg, cfg["serve"]["examples_dir"]) == {"g": 2}
    ex = read_examples(cfg["serve"]["examples_dir"], ["g"])
    assert [e["kind"] for e in ex] == ["corpus", "benchmark"]
    cfg["paths"]["data_processed"] = "/nowhere"  # the files are enough
    assert [e["name"] for e in load_examples(cfg, ["g"])] == [e["name"] for e in ex]
