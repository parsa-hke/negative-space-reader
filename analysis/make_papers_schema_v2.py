"""Papers pack version 2 (2026-10-18): two merges of near-duplicate types, decided from the presence labels.

    .venv/bin/python -m analysis.make_papers_schema_v2

Evidence (CHANGELOG 2026-10-17/18): on the 300 reference sections the teacher (Qwen2.5-72B) and the second judge
(Claude Haiku 5.5) split these types differently (teacher-judge F1: method_proposal 0.11; the comparison types 0.33,
0.58, 0.61), and the v1 cards point at each other (each one's near-miss is the other's positive example). Merging them
lifts papers agreement 0.671 -> 0.692 without relabelling. Parsa asked for the best fast approach with no cost that
keeps the novelty; the other confused pairs (own vs existing limitations, notation vs method details) carry meaning
the omission task needs and stay separate. v1 is kept as packs/papers/history/schema.v1.yaml; labels stay as made
under v1 and are read through `merged_from` (packs/schema_versions.py).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import yaml

from packs.schema_versions import merge_types

REPO = Path(__file__).resolve().parent.parent
PACK = REPO / "packs" / "papers"

MERGES = [
    {"id": "proposes_method", "from": ["proposes_method", "method_proposal"], "card": {
        "definition": "Announces or proposes the authors' new method, model, algorithm or framework, often by name, as "
                      "the paper's contribution or to address a particular problem.",
        "positive_examples": [
            "The authors propose a novel algorithm named ${\\tt LyOn}$ for solving constrained bandit optimization "
            "problems.",
            "The authors propose a new method called SpliMe to leverage static embedding models for temporal knowledge "
            "graphs."],
        "near_miss": "The authors develop a version of Q-learning for finite horizon Markov decision processes.",
        "section_hint": "under a heading like 'abstract' or 'introduction', in the first half of the document"}},
    {"id": "compares_with_other_methods",
     "from": ["performance_comparison", "performance_comparison_across_other_works", "method_comparison"], "card": {
        "definition": "Compares the authors' method, model or system with other methods, baselines or state-of-the-art "
                      "approaches, by performance on datasets or metrics or otherwise, including reports that it "
                      "outperforms them.",
        "positive_examples": [
            "The authors compare their method with state-of-the-art baselines.",
            "The authors report that RadNet outperforms benchmark models across all metrics for the RadSet dataset."],
        "near_miss": "The authors report the performance of the models on the test set for text input.",
        "section_hint": "in the second half of the document"}},
]


def main() -> None:
    v1_path, hist = PACK / "schema.yaml", PACK / "history" / "schema.v1.yaml"
    v1 = yaml.safe_load(v1_path.read_text(encoding="utf-8"))
    if v1["version"] != 1:
        raise SystemExit(f"packs/papers/schema.yaml is already version {v1['version']}")
    hist.parent.mkdir(exist_ok=True)
    shutil.copy(v1_path, hist)
    sha = hashlib.sha1(hist.read_bytes()).hexdigest()[:12]
    v2 = merge_types(v1, MERGES, reason="near-duplicates: the teacher and the second judge split them differently on "
                     "the reference sections (CHANGELOG 2026-10-18)",
                     decided_by="Claude, on Parsa's instruction (2026-10-18): the best fast approach with no cost "
                                "that keeps the novelty",
                     date="2026-10-18", v1_file="packs/papers/history/schema.v1.yaml", v1_sha1=sha)
    v1_path.write_text(yaml.safe_dump(v2, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    print(f"papers schema v2: {len(v1['types'])} -> {len(v2['types'])} types; v1 kept as {hist.relative_to(REPO)}")


if __name__ == "__main__":
    main()
