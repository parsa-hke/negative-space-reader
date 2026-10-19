"""Peer retrieval with category and date filters (plan §0.2 #10, #19; §5.3).

The cohort of a document is its k nearest indexed documents by document embedding, restricted to the same category
where the pack has category metadata and, on request, to a date window. It never contains the query's own entity:
the corpus holds one snapshot per entity per year, so a site's other years would otherwise be its closest "peers".
When the category-filtered pool is smaller than k, the category filter is dropped (flagged in the result); the entity
exclusion and the date window always apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.index import Filters, Hit, Index


@dataclass
class Cohort:
    peers: list[Hit]
    fallback: bool
    filters_used: dict[str, Any]
    stats: dict[str, Any] = field(default_factory=dict)


class CohortRetriever:
    def __init__(self, index: Index, *, k: int = 200, category_filter: bool = True,
                 date_window: tuple[str | None, str | None] | None = None, fallback_unfiltered: bool = True,
                 exact: bool = True):
        self.index, self.k, self.category_filter = index, k, category_filter
        self.date_window, self.fallback_unfiltered, self.exact = date_window, fallback_unfiltered, exact

    @classmethod
    def from_config(cls, index: Index, ccfg: dict[str, Any]) -> "CohortRetriever":
        dw = ccfg.get("date_window")
        return cls(index, k=ccfg.get("k", 200), category_filter=ccfg.get("category_filter", True),
                   date_window=tuple(dw) if dw else None, fallback_unfiltered=ccfg.get("fallback_unfiltered", True))

    def retrieve(self, vector: np.ndarray, *, entity_id: str | None = None, doc_id: str | None = None,
                 category: str | None = None, k: int | None = None,
                 date_window: tuple[str | None, str | None] | None = None) -> Cohort:
        k = k or self.k
        lo, hi = date_window or self.date_window or (None, None)
        cat = [category] if (self.category_filter and category) else None
        base = dict(date_from=lo, date_to=hi, exclude_entities=[entity_id] if entity_id else None,
                    exclude_ids=[doc_id] if doc_id else None)
        peers = self.index.search(vector, k, Filters(category=cat, **base), exact=self.exact)
        fallback = False
        if cat and len(peers) < k and self.fallback_unfiltered:
            peers, cat, fallback = self.index.search(vector, k, Filters(**base), exact=self.exact), None, True
        sims = np.array([h.score for h in peers]) if peers else np.zeros(0)
        stats = {"k": k, "n": len(peers), "entities": len({h.payload.get("entity_id") for h in peers}),
                 "mean_similarity": round(float(sims.mean()), 4) if len(sims) else None,
                 "min_similarity": round(float(sims.min()), 4) if len(sims) else None,
                 "max_similarity": round(float(sims.max()), 4) if len(sims) else None}
        return Cohort(peers, fallback, {"category": cat[0] if cat else None, "date_window": (lo, hi),
                                        "excluded_entity": entity_id}, stats)
