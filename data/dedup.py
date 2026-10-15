"""Near-duplicate detection (MinHash) and version-chain collapsing (plan §2.6).

  exact_key(text)                    sha1 of the normalised text: identical documents
  signatures(texts)                  MinHash signatures over word shingles
  near_duplicate_clusters(sigs)      LSH candidates verified by estimated Jaccard >= threshold, union-find
  collapse_chain(versions)           consecutive versions of one entity whose Jaccard >= same_threshold are one
                                     version (the earliest snapshot of the run is kept)

Thresholds come from configs (`dedup.*`). Normalisation lower-cases and keeps alphanumeric tokens only, so
whitespace, punctuation and Markdown differences never count as changes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np
from datasketch import MinHash, MinHashLSH

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def exact_key(text: str) -> str:
    return hashlib.sha1(" ".join(tokens(text)).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[bytes]:
    toks = tokens(text)
    if len(toks) < k:
        return {" ".join(toks).encode("utf-8")} if toks else set()
    return {" ".join(toks[i:i + k]).encode("utf-8") for i in range(len(toks) - k + 1)}


def signature(text: str, *, num_perm: int = 128, k: int = 5, seed: int = 1) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=seed)
    sh = shingles(text, k)
    if sh:
        m.update_batch(list(sh))
    return m


def signatures(texts: Iterable[tuple[Hashable, str]], *, num_perm: int = 128, k: int = 5) -> dict[Hashable, MinHash]:
    return {key: signature(text, num_perm=num_perm, k=k) for key, text in texts}


def jaccard(a: MinHash, b: MinHash) -> float:
    return float(a.jaccard(b))


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Hashable, Hashable] = {}

    def find(self, x: Hashable) -> Hashable:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: Hashable, b: Hashable) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb, key=str)] = min(ra, rb, key=str)


def near_duplicate_clusters(sigs: dict[Hashable, MinHash], *, threshold: float = 0.85,
                            num_perm: int = 128) -> dict[Hashable, Hashable]:
    """key -> cluster id (the smallest key in its cluster). Singletons map to themselves."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for key, m in sigs.items():
        lsh.insert(key, m)
    uf = _UnionFind()
    for key, m in sigs.items():
        uf.find(key)
        for other in lsh.query(m):
            if other != key and jaccard(m, sigs[other]) >= threshold:
                uf.union(key, other)
    return {key: uf.find(key) for key in sigs}


@dataclass
class Version:
    doc_id: str
    date: str | None
    text: str


def collapse_chain(versions: Sequence[Version], *, same_threshold: float = 0.95, num_perm: int = 128,
                   k: int = 5) -> list[tuple[Version, int, float]]:
    """Collapse consecutive near-identical snapshots. Returns (version, n_snapshots_collapsed, jaccard_to_previous_kept)
    in date order; the first kept version has jaccard 0.0."""
    ordered = sorted(versions, key=lambda v: (v.date or "", v.doc_id))
    kept: list[tuple[Version, int, float]] = []
    prev_sig, prev_key = None, None
    for v in ordered:
        key = exact_key(v.text)
        if prev_key is not None and key == prev_key:
            kept[-1] = (kept[-1][0], kept[-1][1] + 1, kept[-1][2])
            continue
        sig = signature(v.text, num_perm=num_perm, k=k)
        if prev_sig is not None:
            sim = jaccard(sig, prev_sig)
            if sim >= same_threshold:
                kept[-1] = (kept[-1][0], kept[-1][1] + 1, kept[-1][2])
                prev_sig, prev_key = sig, key  # compare the next snapshot with the latest text, not the first
                continue
        else:
            sim = 0.0
        kept.append((v, 1, sim))
        prev_sig, prev_key = sig, key
    return kept


def duplicate_rates(cluster_of: dict[Hashable, Hashable]) -> dict[str, float]:
    sizes = np.bincount(np.unique(list(map(str, cluster_of.values())), return_inverse=True)[1])
    n = len(cluster_of)
    return {"documents": n, "clusters": int(len(sizes)),
            "in_multi_member_clusters": int(sizes[sizes > 1].sum()),
            "near_duplicate_rate": float(1 - len(sizes) / n) if n else 0.0}
