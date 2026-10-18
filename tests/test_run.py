"""experiments/config.py and experiments/run.py: inheritance, overlays, --set, seeds, run logging (plan §1.5)."""

import csv
import json
import random

import numpy as np
import pytest
import torch
import yaml

from experiments import run as runner
from experiments.config import ConfigError, apply_set, deep_merge, load_config


def write(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


# ------------------------------------------------------------------------------------ config
def test_deep_merge_replaces_lists_and_merges_dicts():
    out = deep_merge({"a": {"x": 1, "y": [1, 2]}, "b": 1}, {"a": {"y": [3]}, "b": None})
    assert out == {"a": {"x": 1, "y": [3]}, "b": None}


def test_inherit_chain_overlay_and_set(tmp_path):
    write(tmp_path / "root.yaml", {"run": {"seeds": [0, 1, 2]}, "cohort": {"k": 200}, "m": {"id": "big"}})
    write(tmp_path / "mid.yaml", {"inherit": "root.yaml", "cohort": {"k": 100}})
    write(tmp_path / "task.yaml", {"inherit": ["mid.yaml"], "task": "bench"})
    write(tmp_path / "ov.yaml", {"m": {"id": "tiny"}, "run": {"seeds": [0]}})
    cfg = load_config(tmp_path / "task.yaml", [str(tmp_path / "ov.yaml")], ["cohort.k=50", "+extra.flag=true"])
    assert cfg == {"run": {"seeds": [0]}, "cohort": {"k": 50}, "m": {"id": "tiny"}, "task": "bench",
                   "extra": {"flag": True}}


def test_set_rejects_unknown_keys():
    with pytest.raises(ConfigError):
        apply_set({"cohort": {"k": 1}}, "cohort.kk=5")
    with pytest.raises(ConfigError):
        apply_set({"cohort": {"k": 1}}, "cohort.k")


def test_inherit_cycle_is_detected(tmp_path):
    write(tmp_path / "a.yaml", {"inherit": "b.yaml"})
    write(tmp_path / "b.yaml", {"inherit": "a.yaml"})
    with pytest.raises(ConfigError, match="cycle"):
        load_config(tmp_path / "a.yaml")


LEAN = {"tier": "lean", "embedder": "Qwen/Qwen3-Embedding-0.6B", "late": "answerdotai/answerai-colbert-small-v1",
        "head": "linear", "explain": "template", "real": "revision", "labels": 2000, "rag_peers": 5}
FULL = {"tier": "full", "embedder": "Qwen/Qwen3-Embedding-8B", "late": "jinaai/jina-colbert-v2",
        "head": "finetune", "explain": "teacher", "real": "annotation", "labels": 5000, "rag_peers": 20}


@pytest.mark.parametrize("config", ["data", "embed", "pack_privacy", "pack_papers", "induce", "distil", "mam", "bench",
                                    "baselines", "drift", "serve"])
@pytest.mark.parametrize("overlays", [[], ["local"], ["local", "smoke"], ["gpu"], ["openrouter"],
                                      ["full"], ["full", "gpu"], ["full", "local"]])
def test_repo_configs_resolve(config, overlays):
    cfg = load_config(f"configs/{config}.yaml", overlays)
    assert cfg["task"] in runner.TASKS
    assert cfg["cohort"]["k"] > 0 and cfg["models"]["teacher"]["id"]
    band = cfg["presence"]["band"]  # fixed per pack in Phase 4.5 (configs/pack_<genre>.yaml), computed elsewhere
    if config.startswith("pack_"):
        assert 0 < band["low"] <= band["high"] < 1
    else:
        assert band["low"] is None and band["high"] is None and band["quantiles"] == [0.05, 0.95]
    want = FULL if "full" in overlays else LEAN  # plan §0.2 #25: one overlay flips every tiered key
    assert cfg["tier"] == want["tier"]
    assert cfg["models"]["embedder"]["id"] == ("BAAI/bge-small-en-v1.5" if "smoke" in overlays else want["embedder"])
    assert cfg["models"]["late_interaction"]["id"] == want["late"]
    assert cfg["presence"]["head"] == want["head"]
    assert cfg["explain"]["mode"] == want["explain"]
    assert cfg["bench"]["real"]["source"] == want["real"]
    assert cfg["distil"]["label_sections_per_genre"] == (200 if "smoke" in overlays else want["labels"])
    assert cfg["baselines"]["rag_peers"] == want["rag_peers"]
    assert cfg["corpus"]["max_documents_per_genre"] == (200 if "smoke" in overlays else 25000)
    teacher = cfg["models"]["teacher"]
    if "smoke" in overlays:
        assert (teacher["backend"], teacher["id"]) == ("transformers", "Qwen/Qwen2.5-1.5B-Instruct")
    elif "openrouter" in overlays:
        assert (teacher["backend"], teacher["api_key_env"]) == ("openai", "OPENROUTER_API_KEY")
        assert teacher["provider"]["allow_fallbacks"] is False and "api_key" not in teacher
    elif "local" in overlays:
        assert (teacher["backend"], teacher["id"]) == ("mlx", "mlx-community/Qwen2.5-7B-Instruct-4bit")
    else:
        assert teacher["backend"] == "vllm" and teacher["id"].startswith("Qwen/Qwen2.5-72B-Instruct")


def test_no_secrets_in_configs():
    import re
    from pathlib import Path

    for path in Path("configs").glob("*.yaml"):
        text = path.read_text()
        assert not re.search(r"sk-[A-Za-z0-9]{10,}|api_key:\s*\S", text), f"{path} holds a key-like value"


# ------------------------------------------------------------------------------------ runs
@pytest.fixture
def fake_task(monkeypatch):
    seen = {}

    def task(cfg, run_dir):
        seen[cfg["run"]["seed"]] = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        return {"score": 0.5, "seed": cfg["run"]["seed"]}

    def broken(cfg, run_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "resolve_task", lambda name: {"ok": task, "bad": broken}[name])
    monkeypatch.setitem(runner.TASKS, "ok", "x:y")
    monkeypatch.setitem(runner.TASKS, "bad", "x:y")
    return seen


def read_log(runs_dir):
    with open(runs_dir / "log.csv") as f:
        return list(csv.DictReader(f))


def test_run_writes_config_metrics_and_log_per_seed(tmp_path, fake_task):
    runs_dir = tmp_path / "runs"
    code = runner.main(["--config", "configs/bench.yaml", "--task", "ok", "--name", "t",
                        "--set", "run.seeds=[3, 4]", "--set", "run.device=cpu", "--runs-dir", str(runs_dir)])
    assert code == 0
    for seed in (3, 4):
        run_dir = runs_dir / f"t_s{seed}"
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert cfg["run"]["seed"] == seed
        assert metrics["status"] == "ok" and metrics["metrics"]["seed"] == seed
        assert metrics["tier"] == "lean" and metrics["teacher"]["calls"] == 0
    rows = read_log(runs_dir)
    assert [r["status"] for r in rows] == ["ok", "ok"] and rows[0]["tier"] == "lean"
    assert rows[0]["run_id"] == "t_s3" and json.loads(rows[0]["metrics"])["score"] == 0.5
    assert rows[0]["config_hash"] == rows[1]["config_hash"]  # seed is not part of the config hash


def test_full_tier_is_recorded(tmp_path, fake_task):
    runs_dir = tmp_path / "runs"
    runner.main(["--config", "configs/bench.yaml", "--overlay", "full", "--task", "ok", "--seed", "0",
                 "--set", "run.device=cpu", "--runs-dir", str(runs_dir)])
    assert read_log(runs_dir)[0]["tier"] == "full"
    assert json.loads((runs_dir / "bench_s0" / "metrics.json").read_text())["tier"] == "full"


def test_seeds_make_runs_reproducible(tmp_path, fake_task):
    def launch(seed):
        runner.main(["--config", "configs/bench.yaml", "--task", "ok", "--seed", str(seed),
                     "--set", "run.device=cpu", "--runs-dir", str(tmp_path / "runs")])
        return fake_task[seed]

    first = launch(7)
    random.random(), np.random.rand(), torch.rand(1)  # disturb global RNG state between runs
    assert launch(7) == first
    assert launch(8) != first
    # a repeated run name never overwrites the earlier directory
    assert len([p for p in (tmp_path / "runs").iterdir() if p.name.startswith("bench_s7")]) == 2


def test_failed_run_is_logged(tmp_path, fake_task):
    runs_dir = tmp_path / "runs"
    code = runner.main(["--config", "configs/bench.yaml", "--task", "bad", "--seed", "0",
                        "--set", "run.device=cpu", "--runs-dir", str(runs_dir)])
    assert code == 1
    metrics = json.loads((runs_dir / "bench_s0" / "metrics.json").read_text())
    assert metrics["status"] == "failed" and "boom" in metrics["error"] and "Traceback" in metrics["traceback"]
    assert read_log(runs_dir)[0]["status"] == "failed"


def test_unimplemented_stub_is_logged_as_failed(tmp_path):
    runs_dir = tmp_path / "runs"
    assert runner.main(["--config", "configs/mam.yaml", "--seed", "0", "--set", "run.device=cpu",
                        "--runs-dir", str(runs_dir)]) == 1
    row = read_log(runs_dir)[0]
    assert row["task"] == "train_mam" and row["status"] == "failed" and "Phase 8" in row["error"]


def test_device_info(capsys):
    info = runner.device_info()
    assert info["torch"] == torch.__version__ and info["default_device"] in {"cuda", "mps", "cpu"}
    assert runner.main(["--device-info"]) == 0
    assert "torch" in capsys.readouterr().out
