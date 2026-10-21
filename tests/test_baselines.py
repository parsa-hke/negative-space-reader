"""experiments/baselines.py, baseline (e) candidate list (plan §0.2 #17, 7.3): a seeded sample of OmitBench-synthetic
instances, one prompt per edited document (prompts/llm_gap_candidate.md), the model's ranked absent types scored with
bench/metrics.py and paired with the recorded level-1 and genre-uniform rankings of the same instances. dry_run
renders the prompts and estimates the cost without calling any model."""

import csv
import json

import pytest

TYPES = ["retention", "sharing", "cookies", "children"]


def synth_run(tmp_path, n=6):
    d = tmp_path / "synth" / "g"
    d.mkdir(parents=True)
    with open(d / "instances.jsonl", "w") as f, open(d / "edited.jsonl", "w") as e:
        for i in range(n):
            for regime in ("single_type", "three_types"):
                gt = ["retention"] if i % 2 == 0 else ["cookies"]
                f.write(json.dumps({"doc_id": f"d{i}", "regime": regime, "ground_truth": gt, "targets": gt}) + "\n")
                e.write(json.dumps({"instance_id": f"{regime}:d{i}", "title": f"Policy {i}", "sections": [
                    {"section_idx": 0, "heading": "Sharing", "text": f"We share data with partners ({i})."}]}) + "\n")
    rec = tmp_path / "eval" / "g"
    rec.mkdir(parents=True)
    with open(rec / "per_instance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "regime", "system", "candidates", "ground_truth", "recall@1", "recall@5", "recall@10",
                    "mrr", "ndcg@10", "rank_first", "top5"])
        for i in range(n):
            for system, first in (("genre_uniform", 3), ("level1", 1)):
                r1, r5 = float(first == 1), 1.0
                w.writerow([f"single_type:d{i}", "single_type", system, 3, 1, r1, r5, r5, 1 / first, 0.5, first, ""])
    return tmp_path / "synth" / "{genre}", tmp_path / "eval"


@pytest.fixture
def cfg(tmp_path):
    from experiments.config import load_config

    c = load_config("configs/baselines.yaml", [])
    synth, ev = synth_run(tmp_path)
    schema_dir = tmp_path / "packs" / "g"
    schema_dir.mkdir(parents=True)
    import yaml

    (schema_dir / "schema.yaml").write_text(yaml.safe_dump({"version": 1, "types": [
        {"id": t, "name": t, "definition": f"States its {t} practice."} for t in TYPES]}))
    c["paths"]["packs"] = str(tmp_path / "packs")
    c["genre_info"]["g"] = {"document_kind": "test policy", "author_role": "the company"}
    c["pack"] = "g"
    c["run"]["seed"] = 0
    c["baselines"].update(synth_run=str(synth), compare_to=str(ev), llm_gap_instances=4, regimes=["single_type"])
    return c


class FakeTeacher:
    route = "fake:model"

    def __init__(self):
        self.prompts = []

    def generate_structured_batch(self, prompts, model, **kw):
        self.prompts += prompts
        return [model(absent_types=["cookies", "retention", "cookies"]) for _ in prompts]


def test_dry_run_renders_prompts_and_estimates_cost_without_a_model(cfg, tmp_path):
    from experiments.baselines import run_candidate

    cfg["baselines"]["dry_run"] = True
    m = run_candidate(cfg, tmp_path / "run", teacher=None)
    assert m["g/instances"] == 4 and m["g/estimated_usd"] > 0 and m["dry_run"] is True
    rows = [json.loads(x) for x in open(tmp_path / "run" / "g" / "requests.jsonl")]
    assert len(rows) == 4 and all(r["instance_id"].startswith("single_type:") for r in rows)
    p = rows[0]["prompt"]
    assert "retention: States its retention practice." in p and "We share data with partners" in p
    assert "deleted" not in p.lower() and "removed" not in p.lower() and "<!--" not in p
    again = run_candidate(cfg, tmp_path / "run2", teacher=None)
    assert [json.loads(x)["instance_id"] for x in open(tmp_path / "run2" / "g" / "requests.jsonl")] == \
        [r["instance_id"] for r in rows]  # the sample is seeded
    assert again["g/instances"] == 4


def test_answers_are_scored_and_paired_with_the_recorded_systems(cfg, tmp_path):
    from experiments.baselines import run_candidate

    cfg["baselines"]["tables_dir"] = str(tmp_path / "tables")
    t = FakeTeacher()
    m = run_candidate(cfg, tmp_path / "run", teacher=t)
    assert len(t.prompts) == 4
    rows = list(csv.DictReader(open(tmp_path / "run" / "g" / "per_instance.csv")))
    assert {r["system"] for r in rows} == {"fake:model"} and len(rows) == 4
    assert all(r["ranked"] == "cookies|retention" for r in rows)  # duplicates dropped, order kept
    table = list(csv.DictReader(open(tmp_path / "tables" / "llm_gap_candidate_g.csv")))
    systems = {r["system"] for r in table}
    assert {"fake:model", "level1", "genre_uniform", "fake:model - level1", "fake:model - genre_uniform"} <= systems
    llm = next(r for r in table if r["system"] == "fake:model")
    assert float(llm["recall@5"]) == 1.0 and int(llm["n"]) == 4
    assert m["g/fake:model/recall@5"] == 1.0


def test_answers_outside_the_catalogue_are_dropped_and_counted(cfg, tmp_path):
    from experiments.baselines import run_candidate

    class Odd(FakeTeacher):
        def generate_structured_batch(self, prompts, model, **kw):
            return [model.model_construct(absent_types=["nonsense", "children"]) for _ in prompts]

    m = run_candidate(cfg, tmp_path / "run", teacher=Odd())
    assert m["g/invalid_ids"] == 4
    rows = list(csv.DictReader(open(tmp_path / "run" / "g" / "per_instance.csv")))
    assert all(r["ranked"] == "children" and float(r["recall@5"]) == 0.0 for r in rows)


def test_a_second_model_keeps_the_first_models_rows(cfg, tmp_path):
    from experiments.baselines import run_candidate

    class Other(FakeTeacher):
        route = "other:model"

    cfg["baselines"]["tables_dir"] = str(tmp_path / "tables")
    run_candidate(cfg, tmp_path / "a", teacher=FakeTeacher())
    run_candidate(cfg, tmp_path / "b", teacher=Other())
    run_candidate(cfg, tmp_path / "c", teacher=Other())  # a rerun replaces its own rows
    systems = [r["system"] for r in csv.DictReader(open(tmp_path / "tables" / "llm_gap_candidate_g.csv"))]
    assert sorted(systems) == sorted(["fake:model", "other:model", "level1", "genre_uniform", "fake:model - level1",
                                      "fake:model - genre_uniform", "other:model - level1", "other:model - genre_uniform"])
    assert systems.index("level1") < systems.index("fake:model - level1")


def test_a_prompt_the_route_cannot_answer_is_unanswered_not_truncated(cfg, tmp_path):
    from experiments.baselines import run_candidate
    from models.teacher import StructuredOutputError

    class Short(FakeTeacher):
        def generate_structured_batch(self, prompts, model, **kw):
            if any("(1)" in p for p in prompts):  # the document of d1 does not fit
                raise StructuredOutputError("1 output failed", "")
            return super().generate_structured_batch(prompts, model, **kw)

    cfg["baselines"]["llm_gap_instances"] = 6  # every instance, d1 included
    m = run_candidate(cfg, tmp_path / "run", teacher=Short())
    rows = list(csv.DictReader(open(tmp_path / "run" / "g" / "per_instance.csv")))
    assert m["g/unanswered"] == 1 and len(rows) == 6
    assert all(r["answered"] == "False" and r["ranked"] == "" for r in rows if r["instance_id"].endswith(":d1"))
