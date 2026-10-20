"""Presence probabilities of the sealed test split for OmitBench-synthetic (plan 6.1; stage `bench.stage: probs`).

The bench needs, for every test document, each section's probabilities (document-level presence before and after a
deletion) and each sentence's probabilities (which sentences carry a type). Both come from the pack's lean presence
model (embedder + linear heads, ColBERT prototypes, cascade) in one GPU pass (Colab job `bench_probs_v1`) and are
written to the run folder, never into a pack: the test split is never indexed.

    bench_order(ids, seed)            the bench's fixed order of test documents (instances are drawn in this order,
                                      so a limited run scores exactly the documents a full run would draw first)
    build_sentence_probs(...)         doc_id, section_idx, sentence_idx, prob_<type> (float16), resumable Parquet parts
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


def bench_order(ids: Sequence[str], *, seed: int) -> list[str]:
    ids = sorted(ids)
    return [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]


def check_outside_packs(out: Path, packs: Path) -> None:
    out, packs = Path(out).resolve(), Path(packs).resolve()
    if out == packs or packs in out.parents:
        raise ValueError(f"{out}: test-split probabilities may not be written into a pack ({packs})")


def build_sentence_probs(docs: Sequence[dict[str, Any]],
                         sentences_of: Callable[[str], list[tuple[int, int, str]]],
                         score: Callable[[list[str]], np.ndarray], types: Sequence[str], out: Path, *,
                         chunk_documents: int = 500, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """sentences_of(doc_id) -> [(section_idx, sentence_idx, text)]; score(texts) -> (n, T) probabilities."""
    import pandas as pd

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written, t0, n = 0, time.perf_counter(), 0
    n_parts = (len(docs) + chunk_documents - 1) // chunk_documents
    for part in range(n_parts):
        path = out / f"part-{part:05d}.parquet"
        if path.exists():
            continue
        keys, texts = [], []
        for d in docs[part * chunk_documents:(part + 1) * chunk_documents]:
            for s, j, text in sentences_of(d["id"]):
                keys.append((d["id"], s, j))
                texts.append(text)
        P = np.asarray(score(texts), dtype=np.float16) if texts else np.zeros((0, len(types)), dtype=np.float16)
        df = pd.DataFrame({"doc_id": [k[0] for k in keys], "section_idx": np.array([k[1] for k in keys], dtype=np.int32),
                           "sentence_idx": np.array([k[2] for k in keys], dtype=np.int32)})
        for j, t in enumerate(types):
            df[f"prob_{t}"] = P[:, j] if len(P) else np.zeros(0, dtype=np.float16)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.rename(path)  # a part exists only once complete
        written += 1
        n += len(texts)
        if log:
            log(f"part {part + 1}/{n_parts}: {len(texts)} sentences ({n / max(1e-6, time.perf_counter() - t0):.0f} sentences/s)")
    seconds = time.perf_counter() - t0
    return {"parts": n_parts, "parts_written": written, "sentences_scored": n, "seconds": round(seconds, 1),
            "sentences_per_second": round(n / seconds, 1) if n and seconds else None}


def read_sentence_probs(out: Path) -> Any:
    from packs.profiles import read_profiles

    return read_profiles(out)
