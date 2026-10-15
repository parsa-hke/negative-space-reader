"""Data for the guided schema-review page (plan task 3.4): the same inputs as the review workbook (a cluster run,
the naming runs, Claude's proposals, the pack's seeds), written as one JSON file the page loads.

    .venv/bin/python -m analysis.schema_review_page --cluster-run runs/cluster_v1_s0 \
        --name-run 'runs/name_72b_{genre}_s0' --out reports/schema_review/page/review_data.json

The page stores the reviewer's decisions in its own database; `apply_page_decisions` (packs/schema_review.py)
writes them into the workbook, which stays the record the `candidate` stage reads.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import quantiles

import numpy as np
import yaml

from experiments.induce import cluster_claims_input
from packs.schema_review import complete_proposals, nearest_clusters, proposed_actions

REPO = Path(__file__).resolve().parent.parent
LABELS = {"privacy": ("Privacy policies", "policies"), "papers": ("Research papers", "papers")}


def genre_data(genre: str, crun: Path, nrun: Path, proposals_path: Path, seeds_path: Path,
               typical: int = 10, extra: int = 10, noise_n: int = 40) -> dict:
    ccfg = yaml.safe_load((crun / "config.yaml").read_text(encoding="utf-8"))
    rows, _ = cluster_claims_input(ccfg, genre)
    claims = {c["claim_id"]: c for c in rows}
    n_docs = len({c["doc_id"] for c in rows})
    clusters = json.loads((crun / f"clusters_{genre}.json").read_text(encoding="utf-8"))
    centroids = np.load(crun / f"centroids_{genre}.npy")
    draft = yaml.safe_load((nrun / f"schema_draft_{genre}.yaml").read_text(encoding="utf-8"))
    names = {t["id"]: t for t in draft["types"]}
    proposals = complete_proposals(yaml.safe_load(proposals_path.read_text(encoding="utf-8")), clusters, names, centroids)
    seeds = yaml.safe_load(seeds_path.read_text(encoding="utf-8"))
    ids = [c["id"] for c in clusters]
    nn = nearest_clusters(centroids, ids, k=3)
    actions = proposed_actions(ids, proposals)
    rng = random.Random(0)
    text = lambda ref: claims[ref]["claim"] if ref in claims else (ref or "")  # noqa: E731

    out_clusters = []
    for c in clusters:
        nm = names[c["id"]]
        head, rest = c["members"][:typical], c["members"][typical:]
        action, types, reason = actions[c["id"]]
        out_clusters.append({
            "id": c["id"], "size": c["size"], "documents": c["documents"],
            "share": round(c["documents"] / n_docs, 3), "cohesion": c["cohesion"], "position": c["position_mean"],
            "headings": [h for h in c["heading_top"] if h[0] != "(no heading)"],
            "name": nm["name"], "definition": nm["definition"], "flags": nm["flags"],
            "split_note": nm.get("split_note") or "",
            "examples": [r["claim"] for r in nm["representative_claims"]],
            "typical": [text(m) for m in head], "random": [text(m) for m in rng.sample(rest, min(extra, len(rest)))],
            "labels": c["type_guess_top"], "nearest": nn[c["id"]],
            "proposal": {"action": action, "types": [t.strip() for t in types.split(",") if t.strip()],
                         "reason": reason}})
    out_types = []
    for t in proposals.get("types", []):
        out_types.append({
            "id": t["id"], "name": t.get("name", t["id"]), "definition": t.get("definition", ""),
            "positives": [text(x) for x in t.get("positive_examples", [])],
            "near_miss": text(t.get("near_miss")), "severity": t.get("severity_weight", 1.0),
            "section_hint": t.get("section_hint", ""), "label_source": t.get("label_source", []),
            "seeds": t.get("seeds", []), "clusters": t.get("clusters", []), "notes": t.get("notes", ""),
            "auto": t.get("auto", [])})
    seed_types: dict[str, list[str]] = {}
    for t in out_types:
        for s in t["seeds"]:
            seed_types.setdefault(s, []).append(t["id"])
    out_seeds = [{"id": s["id"], "title": s.get("title") or s.get("category", ""),
                  "types": seed_types.get(s["id"], []),
                  "exclude": ((proposals.get("seeds") or {}).get(s["id"]) or {}).get("exclude", "")}
                 for s in seeds["items"]]
    assigned = {json.loads(l)["claim_id"]: json.loads(l)["cluster"]
                for l in open(crun / f"assignments_{genre}.jsonl", encoding="utf-8")}
    noise_ids = sorted(i for i, c in assigned.items() if c == "noise")
    noise = [text(i) for i in random.Random(1).sample(noise_ids, min(noise_n, len(noise_ids)))]
    coh = [c["cohesion"] for c in clusters]
    q1, q2, q3 = quantiles(coh, n=4)
    return {"genre": genre, "label": LABELS[genre][0], "unit": LABELS[genre][1], "n_documents": n_docs,
            "n_claims": len(rows), "noise_rate": round(sum(1 for v in assigned.values() if v == "noise") / len(assigned), 3),
            "cohesion": {"q1": round(q1, 3), "median": round(q2, 3), "q3": round(q3, 3)},
            "legal_weight": genre == "privacy", "label_source": genre == "privacy",
            "clusters": out_clusters, "types": out_types, "seeds": out_seeds, "noise": noise}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-run", default="runs/cluster_v1_s0")
    ap.add_argument("--name-run", default="runs/name_72b_{genre}_s0")
    ap.add_argument("--proposals", default="reports/schema_review/proposals_{genre}.yaml")
    ap.add_argument("--seeds", default="packs/{genre}/seeds.yaml")
    ap.add_argument("--out", default="reports/schema_review/page/review_data.json")
    a = ap.parse_args()
    data = {"built_from": {"cluster_run": a.cluster_run, "name_run": a.name_run}, "genres": [
        genre_data(g, REPO / a.cluster_run, REPO / a.name_run.format(genre=g), REPO / a.proposals.format(genre=g),
                   REPO / a.seeds.format(genre=g)) for g in ("privacy", "papers")]}
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
