"""notebooks/colab_runner.ipynb: valid notebook, and no secret is ever put into a URL or a command line (a failed
command prints its arguments, which once showed the GitHub token in the cell output)."""

import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "colab_runner.ipynb"


def cells():
    return ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"]]


def test_notebook_parses_and_keeps_its_steps():
    src = cells()
    assert any("hub_sync pull" in s for s in src) and any("experiments.run" in s for s in src)


def test_no_token_in_urls():
    for s in cells():
        assert not re.search(r"https://[^\s\"']*\{os\.environ\[", s), "a secret is interpolated into a URL"


def test_git_errors_are_masked():
    clone = next(s for s in cells() if "git" in s and "clone" in s)
    assert "extraheader" in clone and '"***"' in clone and "capture_output=True" in clone


# Colab's own scientific stack: replacing it underneath a running session breaks imports (numpy `_slice` error) and
# google-colab itself (it pins pandas); found 2026-10-19 on a Python 3.13 runtime
COLAB_OWNED = ("numpy", "pandas", "pyarrow", "numba", "torch", "scipy")


def test_colab_requirements_leave_colabs_stack_alone():
    req = (NB.parent.parent / "requirements-colab.txt").read_text()
    names = [l.split("==")[0].split(">=")[0].strip().lower() for l in req.splitlines()
             if l.strip() and not l.startswith("#")]
    assert not set(names) & set(COLAB_OWNED), set(names) & set(COLAB_OWNED)


def test_libraries_cell_does_not_install_the_package():
    # pyproject pins the Mac's Python (3.11); on Colab the code runs from the clone, which is the working directory
    lib = next(s for s in cells() if "requirements-colab.txt" in s)
    assert "install -e" not in lib


def _jobs():
    import yaml

    return yaml.safe_load((NB.parent.parent / "configs" / "colab_jobs.yaml").read_text())


def test_colab_jobs_are_runnable_as_written():
    import shlex

    import yaml

    root = NB.parent.parent
    spec = _jobs()
    bundles = yaml.safe_load((root / "configs" / "base.yaml").read_text())["hub"]["bundles"]
    assert spec["next"] in spec["jobs"]
    for name, job in spec["jobs"].items():
        assert {"what", "pull", "run", "push"} <= set(job), name
        args = shlex.split(job["run"])
        assert (root / args[args.index("--config") + 1]).is_file(), name
        assert "colab" in [args[i + 1] for i, a in enumerate(args) if a == "--overlay"], name
        assert args[args.index("--name") + 1] == name, f"{name}: the run's --name is the job's name"
        for b in (job["pull"] + "," + job["push"]).split(","):
            assert b.strip() in bundles, f"{name}: unknown bundle {b}"


def test_notebook_runs_the_job_from_the_jobs_file():
    src = "\n".join(cells())
    assert "colab_jobs.yaml" in src and "colab.research.google.com/github/parsa-hke/negative-space-reader" in src
    assert "RUN = \"--config" not in src  # run lines live in the jobs file, not in the notebook
