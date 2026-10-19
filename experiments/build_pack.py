"""Build a genre pack from a config (plan §5.7): packs/builder.py pack_stages in order, per genre, skipping every
stage whose inputs, settings and model ids are unchanged; the manifest is packs/<genre>/pack.json.

    python -m experiments.run --config configs/pack_privacy.yaml --name pack_privacy           # every stage
    python -m experiments.run --config configs/pack_privacy.yaml --name x --set 'build.stages=[index,cohort_report]'

build.adopt: record artefacts that exist but have no manifest entry instead of rebuilding them. A stale GPU stage on a
machine without CUDA stops the build and names its Colab job.
"""

from pathlib import Path
from typing import Any


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from packs.builder import _path, build, pack_stages

    b = cfg.get("build") or {}
    genres = [cfg["pack"]] if cfg.get("pack") else list(cfg["genres"])
    stages = pack_stages(cfg)
    names = [s.name for s in stages]
    wanted = b.get("stages") or names
    unknown = [s for s in wanted if s not in names]
    if unknown:
        raise ValueError(f"unknown build stages {unknown} (known: {names})")
    metrics: dict[str, Any] = {}
    for genre in genres:
        rep = build(cfg, genre, [s for s in stages if s.name in wanted],
                    manifest_path=_path(cfg["paths"]["packs"]) / genre / "pack.json", run_dir=run_dir,
                    adopt=bool(b.get("adopt")), force=b.get("force"))
        print(f"[build_pack {genre}] ran {rep['ran']}, skipped {rep['skipped']}, adopted {rep['adopted']}", flush=True)
        metrics.update({f"{genre}/{k}": v for k, v in rep.items()})
    return metrics
