"""New pack versions of a frozen schema (CLAUDE.md: schemas are frozen after Phase 3; a change is a new pack version).

merge_types(schema, merges, ...)  the next version with some types merged: each merge names the new id, the old ids it
                                  absorbs (`merged_from`) and the new card (definition, examples, near-miss, hint);
                                  source clusters are united, the severity weight is the highest, the seed mapping is
                                  rewritten, and `derived_from` / `changes` record the previous version (kept on file).
projection(schema)                old type id -> new type id, from `merged_from` (empty when nothing was merged)
project_labels(row, projection)   a label row of the previous version read in the new one: a merged type is present when
                                  any type it absorbed is; found by the first pass when any of them was.

Labels are stored as produced (with the schema version they were made under); readers project them, so a merge never
needs a relabelling run. Nothing here names a genre.
"""

from __future__ import annotations

import copy
from typing import Any


def merge_types(schema: dict[str, Any], merges: list[dict[str, Any]], *, reason: str, decided_by: str, date: str,
                v1_file: str, v1_sha1: str) -> dict[str, Any]:
    old = {t["id"]: t for t in schema["types"]}
    seen: set[str] = set()
    for m in merges:
        unknown = [x for x in m["from"] if x not in old]
        if unknown:
            raise ValueError(f"merge {m['id']}: unknown type ids {unknown}")
        twice = seen & set(m["from"])
        if twice:
            raise ValueError(f"types merged more than once: {sorted(twice)}")
        seen |= set(m["from"])
    into = {x: m for m in merges for x in m["from"]}
    types, done = [], set()
    for t in schema["types"]:
        m = into.get(t["id"])
        if m is None:
            types.append(copy.deepcopy(t))
            continue
        if m["id"] in done:
            continue
        done.add(m["id"])
        members = [old[x] for x in m["from"]]
        merged = {"id": m["id"], "name": m["id"], **copy.deepcopy(m["card"]),
                  "severity_weight": max(x.get("severity_weight", 1.0) for x in members),
                  "label_source": sorted({s for x in members for s in (x.get("label_source") or [])}),
                  "source_clusters": [c for x in members for c in (x.get("source_clusters") or [])],
                  "merged_from": list(m["from"])}
        types.append(merged)
    seeds = {}
    for k, v in (schema.get("seed_mapping") or {}).items():
        v = copy.deepcopy(v)
        if "types" in v:
            v["types"] = list(dict.fromkeys(into[x]["id"] if x in into else x for x in v["types"]))
        seeds[k] = v
    out = {k: copy.deepcopy(v) for k, v in schema.items() if k not in ("types", "seed_mapping", "frozen", "version")}
    out.update({"version": schema["version"] + 1, "status": "frozen",
                "frozen": {"date": date, "decided_by": decided_by},
                "derived_from": {"version": schema["version"], "file": v1_file, "sha1": v1_sha1},
                "changes": [{"merged": list(m["from"]), "into": m["id"], "reason": reason} for m in merges],
                "types": types, "seed_mapping": seeds})
    return out


def projection(schema: dict[str, Any]) -> dict[str, str]:
    return {old: t["id"] for t in schema["types"] for old in (t.get("merged_from") or [])}


def project_labels(row: dict[str, Any], proj: dict[str, str]) -> dict[str, Any]:
    if not proj:
        return row
    out = dict(row)
    labels: dict[str, int] = {}
    for t, v in (row.get("labels") or {}).items():
        n = proj.get(t, t)
        labels[n] = max(labels.get(n, 0), int(v))
    out["labels"] = labels
    found: dict[str, str] = {}
    for t, v in (row.get("found_by") or {}).items():
        n = proj.get(t, t)
        found[n] = "first" if v == "first" or found.get(n) == "first" else v
    out["found_by"] = found
    for key in ("evidence", "grounded"):
        merged: dict[str, Any] = {}
        for t, v in (row.get(key) or {}).items():
            n = proj.get(t, t)
            merged[n] = (merged.get(n) or v) if key == "grounded" else merged.get(n, v)
        out[key] = merged
    return out
