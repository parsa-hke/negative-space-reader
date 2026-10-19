"""Build a genre pack from a config (plan §5.7): runs the stages in build.stages (packs/builder.py STAGES), per genre.

    python -m experiments.run --config configs/pack_privacy.yaml --name index_privacy --set 'build.stages=[index,cohort_report]'
"""

from pathlib import Path
from typing import Any


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from packs.builder import STAGES

    genres = [cfg["pack"]] if cfg.get("pack") else list(cfg["genres"])
    stages = (cfg.get("build") or {}).get("stages") or list(STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown build stages {unknown} (known: {list(STAGES)})")
    metrics: dict[str, Any] = {}
    for genre in genres:
        for stage in stages:
            print(f"[build_pack {genre}] {stage}", flush=True)
            metrics.update({f"{genre}/{stage}/{k}": v for k, v in STAGES[stage](cfg, genre).items()})
    return metrics
