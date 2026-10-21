"""Assemble the Hugging Face Space (plan 7.1, "hostable on a CPU Space") in `space/` (gitignored): the code the app
imports, the configs, the pinned examples and results files, and the subset of each pack the app reads, plus the
Space's `app.py`, `README.md` (Space header), `requirements.txt` and `.gitattributes` (Git LFS for the vectors).

    python -m app.space_bundle [--variant cpu|zerogpu] [--out space] [--clean]

It copies; it never uploads. What is left out, on purpose: the corpora (`data/processed`), runs, section_probs and
profiles_v1 (training artefacts the app does not read), schema drafts, label sets, tests, the lab and Colab tooling.
A pack with a stored evidence set loads without the corpus (packs/spec.py section_texts_or_none).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

CODE_DIRS = ["app", "engine", "packs", "data", "experiments", "models", "bench", "analysis"]
PACK_FILES = ["schema.yaml", "pack.json", "cascade.json", "doc_presence.json"]
PACK_DIRS = ["classifier", "prototypes", "profiles", "evidence"]
INDEX_FILES = ["index.json", "payloads.jsonl", "vectors.npy", "best_sections.npz"]  # not sections.npz: no corpus on the Space
# every pattern the Hub stores as LFS and the bundle ships must be here: a Space checkout smudges only the patterns in
# the repo's own .gitattributes (first build, 2026-10-21: .npz and .parquet arrived as pointer files and np.load failed)
LFS_PATTERNS = ["*.npy", "*.npz", "*.parquet"]

APP_PY = '''"""Negative-Space Reader on a Hugging Face Space: the demo of app/gradio_app.py with the overlays __OVERLAYS__ applied."""
__SPACES_IMPORT__
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from experiments.config import load_config  # noqa: E402

from app.gradio_app import build_from_config  # noqa: E402

cfg = load_config("configs/serve.yaml", __OVERLAYS__)
cfg["run"]["seed"] = 0
demo, launch, record, info = build_from_config(cfg)
print(f"[space] {', '.join(info['packs'])} loaded in {info['load_seconds']} s on {info['device']}; {info['examples']} examples", flush=True)
demo.launch(**launch)
'''

README_MD = '''---
title: Negative-Space Reader
emoji: 👻
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: {gradio_version}
python_version: "{python_version}"
app_file: app.py
pinned: false
license: apache-2.0
short_description: What a document conspicuously doesn't say
---

# Negative-Space Reader

Reads what a document conspicuously doesn't say: for every kind of statement its genre peers usually make, how many of
its 200 nearest peers make it, with the peers' own sentences as evidence. Prevalence, never a verdict. Two genres
(privacy policies, computer-science papers); lean tier, CPU only; documents are analysed in memory and never stored.

Built from the project repository by `python -m app.space_bundle`; the "About the project" tab explains the method,
the results and the limits. Numbers come from the recorded result tables (`app/examples/results.json`).
'''

REQUIREMENTS = '''{torch_index}torch=={torch}{spaces_req}
sentence-transformers=={sentence_transformers}
transformers=={transformers}
faiss-cpu=={faiss}
gradio=={gradio}
numpy=={numpy}
pandas=={pandas}
pyarrow=={pyarrow}
PyYAML=={pyyaml}
pydantic>=2.11.10,<=2.12.5
scikit-learn=={sklearn}
'''


# ZeroGPU Spaces build on Python 3.10 whatever README.md's python_version says (build log, 2026-10-21): the latest
# release of each pin that still supports 3.10 (checked against PyPI requires_python) replaces this environment's.
PY310_PINS = {"numpy": "2.2.6", "pandas": "2.3.3", "pyarrow": "25.0.1", "sklearn": "1.7.2"}
# pydantic is a range, not a pin: the Space builder installs gradio[oauth,mcp], whose mcp extra caps pydantic at 2.12.5.


def _pins(*, py310: bool = False) -> dict[str, str]:
    import importlib.metadata as md

    want = {"torch": "torch", "sentence_transformers": "sentence-transformers", "transformers": "transformers", "faiss": "faiss-cpu",
            "gradio": "gradio", "numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow", "pyyaml": "PyYAML", "pydantic": "pydantic",
            "sklearn": "scikit-learn"}
    pins = {k: md.version(v) for k, v in want.items()}
    if py310:
        pins.update(PY310_PINS)
    return pins


CODE_EXTS = {"app": (".py", ".css", ".js", ".json", ".md")}  # app/ carries its static assets and example files
SKIP_DIRS = {"processed", "processed_smoke", "raw", "embeddings", "__pycache__"}  # never corpus or caches


def _copy_tree(src: Path, dst: Path, *, only_py: bool, exts: tuple[str, ...] = (".py",)) -> int:
    n = 0
    for f in src.rglob("*"):
        if not f.is_file() or SKIP_DIRS & set(f.relative_to(src).parts[:-1]) or f.suffix == ".pyc":
            continue
        if only_py and f.suffix not in exts:
            continue
        rel = f.relative_to(src)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / rel)
        n += 1
    return n


def bundle(root: Path, out: Path, cfg: dict[str, Any], *, clean: bool = False, variant: str = "cpu") -> dict[str, Any]:
    """Copy what the Space needs into `out`; return a manifest with file counts and sizes per part. variant: cpu (the
    default and the deployed one, 2026-10-21: no per-visitor GPU quota) or zerogpu (CUDA torch, the `spaces` package, both
    overlays; faster, but ZeroGPU charges GPU time to each visitor's own allowance, which anonymous visitors exhaust fast)."""
    if variant not in ("zerogpu", "cpu"):
        raise ValueError(f"variant must be zerogpu or cpu, not {variant!r}")
    root, out = Path(root), Path(out)
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    genres = list((cfg.get("serve") or {}).get("packs") or cfg["genres"])
    parts: dict[str, dict[str, int]] = {}

    def size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else (p.stat().st_size if p.exists() else 0)

    for d in CODE_DIRS:
        src = root / d
        if not src.is_dir():
            continue
        if d == "packs":  # the Python only; each pack's artefact subset is copied below
            n = sum(1 for f in src.glob("*.py") if (out / d).mkdir(parents=True, exist_ok=True) or shutil.copy2(f, out / d / f.name))
        else:
            n = _copy_tree(src, out / d, only_py=True, exts=CODE_EXTS.get(d, (".py",)))
        parts[d] = {"files": n, "bytes": size(out / d)}
    n = 0
    for f in (root / "configs").glob("*.yaml"):
        (out / "configs").mkdir(exist_ok=True)
        shutil.copy2(f, out / "configs" / f.name)
        n += 1
    parts["configs"] = {"files": n, "bytes": size(out / "configs")}
    (out / "prompts").mkdir(exist_ok=True)
    n = 0
    for f in (root / "prompts").glob("*.md"):
        shutil.copy2(f, out / "prompts" / f.name)
        n += 1
    parts["prompts"] = {"files": n, "bytes": size(out / "prompts")}
    missing: list[str] = []
    for g in genres:
        src, dst = root / "packs" / g, out / "packs" / g
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in PACK_FILES:
            if (src / name).is_file():
                shutil.copy2(src / name, dst / name)
                n += 1
            elif name != "doc_presence.json":
                missing.append(f"packs/{g}/{name}")
        for d in PACK_DIRS:
            if (src / d).is_dir():
                n += _copy_tree(src / d, dst / d, only_py=False)
            else:
                missing.append(f"packs/{g}/{d}/")
        (dst / "index").mkdir(exist_ok=True)
        for name in INDEX_FILES:
            if (src / "index" / name).is_file():
                shutil.copy2(src / "index" / name, dst / "index" / name)
                n += 1
            else:
                missing.append(f"packs/{g}/index/{name}")
        parts[f"packs/{g}"] = {"files": n, "bytes": size(dst)}
    zero = variant == "zerogpu"
    pins = _pins(py310=zero)
    overlays = ["configs/space.yaml"] + (["configs/space_zerogpu.yaml"] if zero else [])
    app_py = APP_PY.replace("__OVERLAYS__", json.dumps(overlays)).replace(  # the template holds f-strings: no str.format
        "__SPACES_IMPORT__", "\nimport spaces  # noqa: F401  must come before torch on a ZeroGPU Space\n" if zero else "")
    (out / "app.py").write_text(app_py, encoding="utf-8")
    (out / "README.md").write_text(README_MD.format(gradio_version=pins["gradio"], python_version="3.10" if zero else "3.11"), encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS.format(**pins, torch_index="" if zero else "--extra-index-url https://download.pytorch.org/whl/cpu\n",
                                                              spaces_req="\nspaces" if zero else ""), encoding="utf-8")
    (out / ".gitattributes").write_text("".join(f"{p} filter=lfs diff=lfs merge=lfs -text\n" for p in LFS_PATTERNS), encoding="utf-8")
    (out / ".gitignore").write_text("__pycache__/\n*.pyc\n.cache/\nruns/\n", encoding="utf-8")
    total = size(out)
    lfs = sum(f.stat().st_size for pat in LFS_PATTERNS for f in out.rglob(pat))
    manifest = {"out": str(out), "variant": variant, "overlays": overlays, "genres": genres, "parts": parts, "total_bytes": total,
                "lfs_bytes": lfs, "missing": missing, "pins": pins}
    (out / "BUNDLE.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    import argparse

    from experiments.config import load_config

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/serve.yaml")
    p.add_argument("--overlay", action="append", default=["configs/space.yaml"])
    p.add_argument("--out", default="space")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--variant", choices=["zerogpu", "cpu"], default="cpu")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.overlay)
    m = bundle(Path("."), Path(a.out), cfg, clean=a.clean, variant=a.variant)
    print(f"variant {m['variant']}: overlays {m['overlays']}")
    for k, v in m["parts"].items():
        print(f"{k:16s} {v['files']:5d} files {v['bytes'] / 1e6:8.1f} MB")
    print(f"total {m['total_bytes'] / 1e6:.1f} MB, of which Git LFS ({', '.join(LFS_PATTERNS)}) {m['lfs_bytes'] / 1e6:.1f} MB; missing: {m['missing'] or 'none'}")
    return 1 if m["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
