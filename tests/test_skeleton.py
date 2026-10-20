"""Every module in the §0.4 layout exists and imports (plan §1.7)."""

import importlib
import pkgutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["data", "packs", "engine", "models", "experiments", "bench", "app", "analysis"]

# Files named in plan §0.4 (code modules only; artefact directories are checked separately).
LAYOUT = """
data/schema.py data/segment.py data/dedup.py data/splits.py data/revisions.py data/build.py
data/loaders/privacy_pl.py data/loaders/opp115.py data/loaders/unarxive.py data/loaders/arxiv_versions.py
packs/spec.py packs/schema_induction.py packs/prototypes.py packs/profiles.py packs/builder.py
engine/embed.py engine/index.py engine/cohort.py engine/presence.py engine/classifier.py
engine/expectation.py engine/rank.py engine/evidence.py engine/explain.py engine/pipeline.py engine/onthefly.py
models/teacher.py models/mam.py models/reranker.py
experiments/run.py experiments/build_pack.py experiments/induce.py experiments/distil.py
experiments/train_mam.py experiments/bench.py experiments/baselines.py experiments/drift.py experiments/embed.py
bench/synth.py bench/revision.py bench/metrics.py app/gradio_app.py app/api.py
analysis/tables.py analysis/figures.py analysis/error_analysis.py
configs/base.yaml configs/smoke.yaml configs/pack_privacy.yaml configs/pack_papers.yaml
configs/induce.yaml configs/distil.yaml configs/mam.yaml configs/bench.yaml configs/baselines.yaml
configs/gpu.yaml configs/local.yaml configs/openrouter.yaml configs/full.yaml configs/data.yaml configs/lab.yaml configs/embed.yaml
""".split()
# data/raw and data/processed are gitignored (created by downloads and the build), so a fresh clone has neither.
DIRS = ["packs/privacy", "packs/papers", "bench/real", "app/web",
        "prompts", "reports/figures", "reports/tables", "reports/paper", "reports/decisions", "runs"]


def all_modules() -> list[str]:
    names = []
    for pkg in PACKAGES:
        names.append(pkg)
        module = importlib.import_module(pkg)
        names += [m.name for m in pkgutil.walk_packages(module.__path__, prefix=f"{pkg}.")]
    return names


@pytest.mark.parametrize("path", LAYOUT)
def test_layout_file_exists(path):
    assert (ROOT / path).is_file(), f"missing {path} (plan §0.4)"


@pytest.mark.parametrize("path", DIRS)
def test_layout_dir_exists(path):
    assert (ROOT / path).is_dir(), f"missing directory {path} (plan §0.4)"


def test_every_module_imports():
    modules = all_modules()
    assert len(modules) >= len([p for p in LAYOUT if p.endswith(".py")])
    for name in modules:
        importlib.import_module(name)


def test_every_task_target_is_importable():
    from experiments.run import TASKS, resolve_task

    assert set(TASKS) == {"build_data", "build_pack", "embed", "induce", "distil", "train_mam", "bench", "analyse", "baselines",
                          "drift", "serve"}
    for task in TASKS:
        assert callable(resolve_task(task))


def test_root_files():
    for name in ["CLAUDE.md", "NEGATIVE_SPACE_PLAN.md", "CHANGELOG.md", "README.md", "pyproject.toml",
                 "requirements.txt", ".gitignore", "data/README.md", "reports/decisions/001-lean-architecture.md"]:
        assert (ROOT / name).is_file(), name
