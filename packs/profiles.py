"""Document profiles (plan 4.7): presence over every section of the indexed documents, aggregated per document.

A document's profile has, per type, prob_<type> = the highest section probability (a statement is present in the
document if some section states it) and bit_<type> = whether that reaches the type's threshold. build_profiles writes
chunks of `chunk_documents` documents as Parquet parts (part-00000.parquet, ...) and skips parts that exist, so an
overnight run can be stopped and resumed; read_profiles concatenates them. The scoring function is the pack's presence
model (engine/presence.py), passed in, so the same code serves every variant and genre.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


def aggregate(P: np.ndarray, *, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prob = P.max(axis=0) if len(P) else np.zeros(P.shape[1])
    return prob, (prob >= thresholds).astype(np.int8)


def build_profiles(docs: Sequence[dict[str, Any]], sections_of: Callable[[str], list[str]],
                   score: Callable[[list[str]], np.ndarray], types: Sequence[str], thresholds: np.ndarray, out: Path, *,
                   chunk_documents: int = 500, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    import pandas as pd

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written, t0, n_sections = 0, time.perf_counter(), 0
    n_parts = (len(docs) + chunk_documents - 1) // chunk_documents
    for part in range(n_parts):
        path = out / f"part-{part:05d}.parquet"
        if path.exists():
            continue
        chunk = docs[part * chunk_documents:(part + 1) * chunk_documents]
        texts, owner = [], []
        for i, d in enumerate(chunk):
            secs = sections_of(d["id"])
            texts += secs
            owner += [i] * len(secs)
        P = score(texts) if texts else np.zeros((0, len(types)))
        owner_arr = np.array(owner)
        rows = []
        for i, d in enumerate(chunk):
            prob, bits = aggregate(P[owner_arr == i] if len(owner_arr) else np.zeros((0, len(types))),
                                   thresholds=thresholds)
            rows.append({"doc_id": d["id"], **{k: d.get(k) for k in ("split", "category", "date")},
                         "sections": int((owner_arr == i).sum()) if len(owner_arr) else 0,
                         **{f"prob_{t}": float(p) for t, p in zip(types, prob)},
                         **{f"bit_{t}": int(b) for t, b in zip(types, bits)}})
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        tmp.rename(path)  # a part exists only once complete
        written += 1
        n_sections += len(texts)
        if log:
            rate = n_sections / max(1e-6, time.perf_counter() - t0)
            log(f"part {part + 1}/{n_parts}: {len(chunk)} documents, {len(texts)} sections ({rate:.1f} sections/s)")
    seconds = time.perf_counter() - t0
    return {"parts": n_parts, "parts_written": written, "sections_scored": n_sections, "seconds": round(seconds, 1),
            "sections_per_second": round(n_sections / seconds, 1) if n_sections and seconds else None}


def build_section_probs(docs: Sequence[dict[str, Any]], sections_of: Callable[[str], list[tuple[int, str]]],
                        score: Callable[[list[str]], np.ndarray], types: Sequence[str], out: Path, *,
                        chunk_documents: int = 500, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Every section's presence probabilities (doc_id, split, section_idx, prob_<type> as float32), in resumable
    Parquet parts like build_profiles. Document-level presence is fitted on these (engine/doc_presence.py), so the
    GPU pass runs once and calibration can be refitted on the Mac. sections_of(doc_id) -> [(section_idx, text)]."""
    import pandas as pd

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written, t0, n_sections = 0, time.perf_counter(), 0
    n_parts = (len(docs) + chunk_documents - 1) // chunk_documents
    for part in range(n_parts):
        path = out / f"part-{part:05d}.parquet"
        if path.exists():
            continue
        chunk = docs[part * chunk_documents:(part + 1) * chunk_documents]
        keys, texts = [], []
        for d in chunk:
            for idx, text in sections_of(d["id"]):
                keys.append((d["id"], d.get("split"), idx))
                texts.append(text)
        P = np.asarray(score(texts), dtype=np.float32) if texts else np.zeros((0, len(types)), dtype=np.float32)
        df = pd.DataFrame({"doc_id": [k[0] for k in keys], "split": [k[1] for k in keys],
                           "section_idx": np.array([k[2] for k in keys], dtype=np.int32)})
        for j, t in enumerate(types):
            df[f"prob_{t}"] = P[:, j] if len(P) else np.zeros(0, dtype=np.float32)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.rename(path)  # a part exists only once complete
        written += 1
        n_sections += len(texts)
        if log:
            rate = n_sections / max(1e-6, time.perf_counter() - t0)
            log(f"part {part + 1}/{n_parts}: {len(chunk)} documents, {len(texts)} sections ({rate:.1f} sections/s)")
    seconds = time.perf_counter() - t0
    return {"parts": n_parts, "parts_written": written, "sections_scored": n_sections, "seconds": round(seconds, 1),
            "sections_per_second": round(n_sections / seconds, 1) if n_sections and seconds else None}


def read_profiles(out: Path) -> Any:
    import pandas as pd

    parts = sorted(Path(out).glob("part-*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()
