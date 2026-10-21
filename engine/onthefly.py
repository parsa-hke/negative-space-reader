"""Pack detection and temporary schema induction (plan §0.2 #20; §10.2).

detect_pack (used by the app's "auto" pack, plan 7.1): embed the upload once per embedder (packs that share an
embedder and segmentation settings share the embedding), build each pack's document vector, and pick the pack whose
index holds the nearest neighbours (mean similarity of the top n, no filters). The margin over the next pack is
reported; below `threshold` (onthefly.pack_similarity_threshold) no pack matches, which is where the on-the-fly mode
of Phase 10 takes over. Temporary schema induction: Phase 10.2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def detect_pack(text_or_path: str | Path, packs: dict[str, Any], *, neighbours: int = 10,
                threshold: float | None = None, title: str | None = None) -> dict[str, Any]:
    from engine.pipeline import document_vector_of, embed_upload

    embedded: dict[str, Any] = {}
    shared: dict[tuple[Any, str], Any] = {}
    scores: dict[str, float] = {}
    for name, pack in packs.items():
        key = ((pack.info or {}).get("embedder") or id(pack.embedder), json.dumps(pack.cfg["segment"], sort_keys=True))
        if key not in shared:
            shared[key] = embed_upload(text_or_path, pack, title=title)
        embedded[name] = shared[key]
        hits = pack.index.search(document_vector_of(embedded[name], pack), neighbours, exact=True)
        scores[name] = round(sum(h.score for h in hits) / len(hits), 4) if hits else 0.0
    order = sorted(scores, key=lambda n: -scores[n])
    best = order[0] if order else None
    matched = best is not None and (threshold is None or scores[best] >= threshold)
    margin = round(scores[order[0]] - scores[order[1]], 4) if matched and len(order) > 1 else None
    return {"pack": best if matched else None, "matched": matched, "scores": scores, "margin": margin,
            "neighbours": neighbours, "threshold": threshold, "embedded": embedded}
