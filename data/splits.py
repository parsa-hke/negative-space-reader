"""Entity-disjoint train / dev / test splits (plan §2.7).

  train  the indexable corpus (cohorts, profiles, MAM training)
  dev    schema-induction samples, calibration, presence-head training data (OPP-115 entities live here)
  test   OmitBench documents; never indexed. Every entity with a revision pair used by OmitBench-revision is here.

Assignment is a pure function of (seed, entity_id), so it is stable when the corpus grows or is re-sampled.
Entities can be forced into a split (e.g. OPP-115 sites into dev). Near-duplicate clusters that span splits are
resolved so no test document has a near-copy in train or dev: members outside the highest-priority split of their
cluster (test > dev > train) are dropped.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Hashable, Iterable

SPLITS = ("train", "dev", "test")
PRIORITY = {"test": 2, "dev": 1, "train": 0}


def assign_split(entity_id: str, *, seed: int, fractions: dict[str, float], forced: dict[str, str] | None = None) -> str:
    if forced and entity_id in forced:
        return forced[entity_id]
    h = int.from_bytes(hashlib.sha256(f"{seed}:{entity_id}".encode("utf-8")).digest()[:8], "big") / 2**64
    edge = 0.0
    for name in SPLITS:
        edge += fractions[name]
        if h < edge:
            return name
    return SPLITS[-1]


def resolve_cross_split_duplicates(cluster_of: dict[Hashable, Hashable], split_of: dict[Hashable, str]) -> set[Hashable]:
    """Doc keys to drop so that no near-duplicate cluster spans two splits."""
    members: dict[Hashable, list[Hashable]] = defaultdict(list)
    for key, cluster in cluster_of.items():
        members[cluster].append(key)
    drop: set[Hashable] = set()
    for keys in members.values():
        splits = {split_of[k] for k in keys}
        if len(splits) <= 1:
            continue
        keep = max(splits, key=PRIORITY.__getitem__)
        drop.update(k for k in keys if split_of[k] != keep)
    return drop


def manifest_hash(doc_ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(doc_ids)).encode("utf-8")).hexdigest()


def write_manifest(path: str | Path, *, genre: str, seed: int, fractions: dict[str, float],
                   doc_split: dict[str, str], entity_of: dict[str, str], extra: dict | None = None) -> dict:
    """Write data/processed/splits/<genre>.json and return its summary (with the sealed test hash)."""
    by_split: dict[str, list[str]] = {s: [] for s in SPLITS}
    for doc_id, split in doc_split.items():
        by_split[split].append(doc_id)
    entities = {s: sorted({entity_of[d] for d in ids}) for s, ids in by_split.items()}
    overlap = set(entities["test"]) & (set(entities["train"]) | set(entities["dev"]))
    if overlap:
        raise ValueError(f"{len(overlap)} entities appear in test and another split, e.g. {sorted(overlap)[:3]}")
    summary = {
        "genre": genre, "seed": seed, "fractions": fractions,
        "documents": {s: len(v) for s, v in by_split.items()},
        "entities": {s: len(v) for s, v in entities.items()},
        "test_sha256": manifest_hash(by_split["test"]),
        **(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**summary, "doc_ids": {s: sorted(v) for s, v in by_split.items()}}, indent=1),
                    encoding="utf-8")
    return summary


def load_manifest(path: str | Path) -> dict[str, str]:
    """doc_id -> split."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {d: s for s, ids in data["doc_ids"].items() for d in ids}


def split_counts(doc_split: dict[str, str]) -> Counter:
    return Counter(doc_split.values())
