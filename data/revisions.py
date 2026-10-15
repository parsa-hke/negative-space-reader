"""Version chains -> (version n, version n+1) pairs for OmitBench-revision (plan §2.6b, §0.2 #16).

Genre-agnostic: input is any entity's versions as ParsedDocuments. Consecutive near-identical versions are collapsed
first (data.dedup.collapse_chain), then a pair is kept when the change is material but the two texts are still the
same document:

    min_change <= 1 - Jaccard(version n, version n+1) <= max_change

(below min_change: cosmetic edits; above max_change: a different page, a site redesign or an extraction failure).
Which types were added is decided later by presence detection (plan §6.3b), not here.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from data.dedup import Version, collapse_chain
from data.schema import ParsedDocument, RevisionPair


def chain_pairs(versions: Iterable[ParsedDocument], *, same_threshold: float = 0.95, min_change: float = 0.05,
                max_change: float = 0.9, num_perm: int = 128, shingle_size: int = 5) -> Iterator[RevisionPair]:
    docs = {v.document.id: v for v in versions}
    if len(docs) < 2:
        return
    kept = collapse_chain([Version(d.document.id, d.document.date, d.text()) for d in docs.values()],
                          same_threshold=same_threshold, num_perm=num_perm, k=shingle_size)
    for (old, _, _), (new, _, sim) in zip(kept, kept[1:]):
        change = 1.0 - sim
        if min_change <= change <= max_change:
            o, n = docs[old.doc_id].document, docs[new.doc_id].document
            yield RevisionPair(entity_id=o.entity_id, genre=o.genre, old_doc_id=o.id, new_doc_id=n.id,
                               old_date=o.date, new_date=n.date, change_ratio=round(change, 4))
