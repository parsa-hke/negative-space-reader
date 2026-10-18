"""Encode an embedding bundle on another machine (Colab), self-contained: no imports from this repository.

The bundle comes from `embed.mode: export` (experiments/embed.py): manifest.json (model id, Hugging Face snapshot,
max length, normalisation, truncation) and one gzip JSON-lines file per (genre, unit) with rows {sha1, text}: only the
texts the local store does not hold. This script encodes them exactly as engine/embed.Embedder does (same model
kwargs, left padding, max_seq_length, normalisation; fp16 on CUDA) and writes, per file, vectors_<name>.f16.npy and
rows_<name>.jsonl (sha1 per row, in order) plus meta.json, then zips them. `embed.mode: import` merges the zip into
the local store by sha1.

    python colab_embed.py --bundle <unzipped bundle dir> --out colab_out
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


def load_model(manifest: dict[str, Any], device: str) -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = SentenceTransformer(manifest["model"], device=device, revision=manifest.get("revision"),
                                model_kwargs={"dtype": dtype}, processor_kwargs={"padding_side": "left"},
                                truncate_dim=manifest.get("truncate_dim"))
    model.max_seq_length = manifest["max_length"]
    return model


def encode_bundle(bundle: Path, out: Path, encode: Callable[[Sequence[str]], np.ndarray],
                  info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Encode every file listed in bundle/manifest.json with encode(texts) -> unit rows; write outputs to out."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"manifest": manifest, **(info or {}), "files": {}}
    for name, entry in manifest["files"].items():
        with gzip.open(bundle / entry["file"], "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        for r in rows:  # the text must be the one whose hash the local store will look up
            if hashlib.sha1(r["text"].encode("utf-8")).hexdigest() != r["sha1"]:
                raise ValueError(f"{name}: text does not match its sha1 {r['sha1']}")
        t0 = time.perf_counter()
        X = np.asarray(encode([r["text"] for r in rows]), dtype=np.float32)
        seconds = time.perf_counter() - t0
        if X.shape[0] != len(rows) or not np.isfinite(X).all():
            raise ValueError(f"{name}: bad output {X.shape}")
        np.save(out / f"vectors_{name}.f16.npy", X.astype(np.float16))
        with open(out / f"rows_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"sha1": r["sha1"]}) + "\n")
        meta["files"][name] = {"rows": len(rows), "dim": int(X.shape[1]), "seconds": round(seconds, 1),
                               "rows_per_second": round(len(rows) / max(seconds, 1e-6), 1)}
        print(f"[colab_embed] {name}: {len(rows)} rows in {seconds:.0f}s", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def main() -> None:
    import sentence_transformers
    import torch
    import transformers

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--out", default="colab_out")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    bundle, out = Path(args.bundle), Path(args.out)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[colab_embed] WARNING: no GPU found (Runtime > Change runtime type > T4 GPU); this will be slow")
    model = load_model(manifest, device)
    info = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "dtype": "float16" if device == "cuda" else "float32", "torch": torch.__version__,
            "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__,
            "python": platform.python_version()}
    for key in ("sentence_transformers", "transformers"):
        if manifest.get("versions", {}).get(key) and manifest["versions"][key] != info[key]:
            print(f"[colab_embed] WARNING: {key} {info[key]} differs from the local {manifest['versions'][key]}")

    def encode(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), batch_size=args.batch_size, normalize_embeddings=manifest["normalize"],
                            show_progress_bar=True, convert_to_numpy=True)

    encode_bundle(bundle, out, encode, info)
    shutil.make_archive(str(out), "zip", out)
    print(f"[colab_embed] wrote {out}.zip")


if __name__ == "__main__":
    main()
