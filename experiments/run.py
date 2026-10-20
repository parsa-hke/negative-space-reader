"""Single entry point for every experiment (plan §0.2 #23, §1.5).

    python -m experiments.run --device-info
    python -m experiments.run --config configs/induce.yaml --overlay smoke --overlay local --set pack=papers
    python -m experiments.run --config configs/bench.yaml --set cohort.k=50 --seed 0
    python -m experiments.run --config configs/bench.yaml --print-config

For each seed in run.seeds (or --seed) this writes runs/<name>_s<seed>/{config.yaml,metrics.json} and
appends one line to runs/log.csv, including for failed and interrupted runs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import yaml

from experiments.config import config_hash, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

# task name -> "module:function"; every function has the signature (cfg, run_dir) -> metrics dict
TASKS: dict[str, str] = {
    "build_data": "data.build:main",
    "build_pack": "experiments.build_pack:main",
    "embed": "experiments.embed:main",
    "induce": "experiments.induce:main",
    "distil": "experiments.distil:main",
    "train_mam": "experiments.train_mam:main",
    "bench": "experiments.bench:main",
    "analyse": "experiments.analyse:main",
    "baselines": "experiments.baselines:main",
    "drift": "experiments.drift:main",
    "serve": "app.gradio_app:main",
}

LOG_FIELDS = [
    "timestamp", "run_id", "task", "pack", "tier", "config", "overlays", "sets", "seed", "status",
    "duration_s", "config_hash", "git_commit", "device", "prompt_hashes", "teacher_calls", "metrics", "error",
]


# ---------------------------------------------------------------------------------------- devices
def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _gib(n_bytes: int) -> float:
    return round(n_bytes / 2**30, 1)


def _chip_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or platform.machine()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or platform.machine()


def device_info() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    import psutil
    import torch

    info: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _chip_name(),
        "cpu_count": os.cpu_count(),
        "ram_gib": _gib(psutil.virtual_memory().total),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "mps_available": torch.backends.mps.is_available(),
        "default_device": resolve_device("auto"),
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i, "type": "cuda", "name": p.name,
                "vram_gib": _gib(p.total_memory),
                "compute_capability": f"{p.major}.{p.minor}",
            })
        info["total_vram_gib"] = round(sum(d["vram_gib"] for d in info["devices"]), 1)
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            info["nvidia_driver"] = out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, IndexError):
            info["nvidia_driver"] = None
    if torch.backends.mps.is_available():
        # Unified memory: MPS can use up to recommended_max_memory of system RAM.
        info["devices"].append({
            "index": 0, "type": "mps", "name": info["processor"],
            "recommended_max_memory_gib": _gib(torch.mps.recommended_max_memory()),
        })
    info["packages"] = {}
    for dist in ("transformers", "sentence-transformers", "xgrammar", "outlines", "vllm", "openai"):
        try:
            info["packages"][dist] = version(dist)
        except PackageNotFoundError:  # vllm is not installable on macOS
            info["packages"][dist] = None
    return info


def print_device_info(info: dict[str, Any]) -> None:
    print(f"host            {info['host']}  ({info['platform']})")
    print(f"cpu / ram       {info['processor']}, {info['cpu_count']} cores, {info['ram_gib']} GiB")
    print(f"python / torch  {info['python']} / {info['torch']}")
    print(f"cuda            available={info['cuda_available']} version={info['cuda_version']} "
          f"cudnn={info['cudnn_version']} driver={info.get('nvidia_driver')}")
    print(f"mps             available={info['mps_available']}")
    print(f"default device  {info['default_device']}")
    for d in info["devices"]:
        if d["type"] == "cuda":
            print(f"  [cuda:{d['index']}] {d['name']}  VRAM={d['vram_gib']} GiB  sm_{d['compute_capability']}")
        else:
            print(f"  [mps:0] {d['name']}  unified memory usable by MPS={d['recommended_max_memory_gib']} GiB")
    if "total_vram_gib" in info:
        print(f"total VRAM      {info['total_vram_gib']} GiB across {len(info['devices'])} GPU(s)")
    print("packages        " + ", ".join(f"{k}={v}" for k, v in info["packages"].items()))


# ---------------------------------------------------------------------------------------- seeds
def set_seeds(seed: int, deterministic: bool = False) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses only
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------------------- logging
def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "none"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.TimeoutExpired):
        return "none"


def append_log(log_path: Path, row: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists() or log_path.stat().st_size == 0
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def make_run_dir(runs_dir: Path, run_id: str) -> Path:
    run_dir = runs_dir / run_id
    if run_dir.exists():  # never overwrite an earlier run
        run_dir = runs_dir / f"{run_id}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    run_dir.mkdir(parents=True)
    return run_dir


def resolve_task(task: str) -> Callable[[dict[str, Any], Path], dict[str, Any]]:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")
    module, fn = TASKS[task].split(":")
    return getattr(importlib.import_module(module), fn)


# ---------------------------------------------------------------------------------------- runs
def run_one(cfg: dict[str, Any], seed: int, *, config_path: str, overlays: list[str], sets: list[str],
            runs_dir: Path) -> tuple[str, Path]:
    """Run one (config, seed). Returns (status, run_dir). Never raises except on KeyboardInterrupt."""
    from models.teacher import PROMPTS, TEACHER_STATS  # prompt hashes and teacher calls are recorded per run

    cfg = json.loads(json.dumps(cfg))  # private copy
    cfg["run"]["seed"] = seed
    name = cfg["run"].get("name") or Path(config_path).stem
    task = cfg.get("task")
    run_dir = make_run_dir(runs_dir, f"{name}_s{seed}")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    PROMPTS.reset_usage()
    TEACHER_STATS.reset()
    status, error, metrics, tb = "ok", "", {}, None
    device = "unknown"
    t0 = time.perf_counter()
    try:
        device = resolve_device(cfg["run"].get("device", "auto"))
        set_seeds(seed, cfg["run"].get("deterministic", False))
        fn = resolve_task(task)
        metrics = fn(cfg, run_dir) or {}
    except KeyboardInterrupt:
        status, error, tb = "interrupted", "KeyboardInterrupt", traceback.format_exc()
        raise
    except Exception as e:  # every failure is logged, then the next seed runs
        status, error, tb = "failed", f"{type(e).__name__}: {e}", traceback.format_exc()
    finally:
        duration = round(time.perf_counter() - t0, 3)
        prompt_hashes = PROMPTS.usage()
        record = {
            "status": status, "task": task, "pack": cfg.get("pack"), "seed": seed,
            "duration_s": duration, "device": device, "git_commit": git_commit(),
            "config_hash": config_hash({k: v for k, v in cfg.items() if k != "run"}),
            "tier": cfg.get("tier"), "prompt_hashes": prompt_hashes, "teacher": TEACHER_STATS.summary(),
            "metrics": metrics,
        }
        if tb:
            record["error"], record["traceback"] = error, tb
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        append_log(runs_dir / "log.csv", {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "run_id": run_dir.name, "task": task, "pack": cfg.get("pack") or "", "tier": cfg.get("tier") or "",
            "config": config_path, "overlays": "+".join(overlays), "sets": " ".join(sets),
            "seed": seed, "status": status, "duration_s": duration,
            "config_hash": record["config_hash"], "git_commit": record["git_commit"], "device": device,
            "prompt_hashes": json.dumps(prompt_hashes, sort_keys=True) if prompt_hashes else "",
            "teacher_calls": json.dumps(record["teacher"]["served_by"], sort_keys=True) if record["teacher"]["calls"] else "",
            "metrics": json.dumps(metrics, sort_keys=True, default=str) if metrics else "",
            "error": error.replace("\n", " ")[:500],
        })
    print(f"[{status}] {run_dir}" + (f"  {error}" if error else ""), file=sys.stderr)
    return status, run_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiments.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="task config, e.g. configs/induce.yaml")
    p.add_argument("--overlay", action="append", default=[], help="overlay name or path (repeatable)")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                   help="override a dotted key (repeatable); +KEY=VALUE adds a new key")
    p.add_argument("--task", help="override the config's task")
    p.add_argument("--seed", type=int, action="append", help="run only these seeds (repeatable)")
    p.add_argument("--name", help="run name (default: config stem)")
    p.add_argument("--runs-dir", help="default: paths.runs from the config")
    p.add_argument("--device-info", action="store_true", help="print devices, VRAM, CUDA, torch and exit")
    p.add_argument("--json", action="store_true", help="with --device-info: print JSON")
    p.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    args = p.parse_args(argv)

    if args.device_info:
        info = device_info()
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print_device_info(info)
        return 0
    if not args.config:
        p.error("--config is required (or use --device-info)")

    cfg = load_config(args.config, args.overlay, args.sets)
    if args.task:
        cfg["task"] = args.task
    if args.name:
        cfg["run"]["name"] = args.name
    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False), end="")
        return 0
    if cfg.get("task") not in TASKS:
        p.error(f"config task {cfg.get('task')!r} is not one of {sorted(TASKS)}")

    runs_dir = Path(args.runs_dir or cfg["paths"]["runs"])
    if not runs_dir.is_absolute():
        runs_dir = REPO_ROOT / runs_dir
    seeds = args.seed or cfg["run"]["seeds"]
    statuses = [run_one(cfg, s, config_path=args.config, overlays=args.overlay, sets=args.sets,
                        runs_dir=runs_dir)[0] for s in seeds]
    return 0 if all(s == "ok" for s in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
