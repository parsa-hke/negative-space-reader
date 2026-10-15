"""YAML config loading: `inherit:` chains, overlays, and `--set key=value` overrides (plan §1.5).

Resolution order (later wins):
  1. every file named in `inherit:` (recursively, in list order; paths relative to the inheriting file)
  2. the config file itself
  3. each `--overlay` in order (a name like `smoke` means configs/smoke.yaml)
  4. each `--set dotted.key=value` (value parsed as YAML; key must already exist unless prefixed with `+`)

Dicts merge recursively; lists and scalars are replaced. A key set to null overrides a value.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class ConfigError(ValueError):
    pass


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(ref: str | Path, relative_to: Path | None = None) -> Path:
    """`smoke` -> configs/smoke.yaml; `x.yaml` -> relative to the referring file, then configs/."""
    ref = Path(ref)
    candidates = []
    if ref.suffix not in (".yaml", ".yml"):
        candidates.append(CONFIG_DIR / f"{ref}.yaml")
    else:
        if ref.is_absolute():
            candidates.append(ref)
        if relative_to is not None:
            candidates.append(relative_to / ref)
        candidates += [Path.cwd() / ref, CONFIG_DIR.parent / ref, CONFIG_DIR / ref]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise ConfigError(f"config not found: {ref} (looked in {[str(c) for c in candidates]})")


def load_file(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load one YAML file (an existing path) and resolve its `inherit:` chain."""
    path = Path(path).resolve()
    if path in _stack:
        chain = " -> ".join(p.name for p in (*_stack, path))
        raise ConfigError(f"inherit cycle: {chain}")
    with open(path, encoding="utf-8") as f:  # explicit: Windows defaults to cp1252
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parents = raw.pop("inherit", None) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: dict[str, Any] = {}
    for parent in parents:
        merged = deep_merge(merged, load_file(resolve_path(parent, path.parent), (*_stack, path)))
    return deep_merge(merged, raw)


def apply_set(cfg: dict[str, Any], assignment: str) -> dict[str, Any]:
    """Apply one `dotted.key=value` override in place. `+dotted.key=value` may create new keys."""
    if "=" not in assignment:
        raise ConfigError(f"--set expects key=value, got {assignment!r}")
    key, raw_value = assignment.split("=", 1)
    allow_new = key.startswith("+")
    parts = key.lstrip("+").strip().split(".")
    value = yaml.safe_load(raw_value) if raw_value.strip() else None
    node = cfg
    for i, part in enumerate(parts[:-1]):
        if part not in node:
            if not allow_new:
                raise ConfigError(f"--set {key}: unknown key {'.'.join(parts[: i + 1])!r} (use +{key} to add)")
            node[part] = {}
        node = node[part]
        if not isinstance(node, dict):
            raise ConfigError(f"--set {key}: {'.'.join(parts[: i + 1])!r} is not a mapping")
    if parts[-1] not in node and not allow_new:
        raise ConfigError(f"--set {key}: unknown key {key!r} (use +{key} to add)")
    node[parts[-1]] = value
    return cfg


def load_config(
    path: str | Path,
    overlays: list[str] | None = None,
    sets: list[str] | None = None,
) -> dict[str, Any]:
    cfg = load_file(resolve_path(path))
    for overlay in overlays or []:
        cfg = deep_merge(cfg, load_file(resolve_path(overlay)))
    for assignment in sets or []:
        apply_set(cfg, assignment)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of a resolved config (seed-independent if the caller strips the seed)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
