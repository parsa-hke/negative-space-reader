"""Turn the reviewer's answers from the review page, plus the resolutions agreed afterwards, into the final reviewed
checklist per genre, in the proposals format (reports/schema_review/final_<genre>.yaml). The `review` stage then
builds the workbook from it and the `candidate` stage reads that workbook, so the gate checks are the usual ones.

    .venv/bin/python -m analysis.apply_review_decisions \
        --answers reports/schema_review/decisions/page_answers_2026-10-15.json \
        --resolutions reports/schema_review/decisions/resolutions_2026-10-15.yaml

Rules (each change is reported):
  - a group's items: a resolution (merge / keep as one) wins; else `accept` = Claude's proposal; `drop` = none;
    `own` / `split` = the names the reviewer gave. "Keep as one" names the item after the group.
  - an item the reviewer removed disappears; a group left with no item is pruned.
  - a proposed item whose groups all went to other items is replaced by those items: checklist rows that pointed
    at it are re-pointed, and (with carry_over) its severity weight and label source pass to the replacements.
  - edits the reviewer made to a proposed item's fields on the page override the proposal (a new item's fields on
    the page were defaults, not edits).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def slug(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def resolve(genre: str, page: dict, proposals: dict, answers: dict, res: dict) -> tuple[dict, list[str]]:
    log: list[str] = []
    clusters = page["clusters"]
    prop = {t["id"]: t for t in proposals["types"]}
    merge = (res.get("merge") or {}).get(genre, {})
    keep_one = set((res.get("keep_as_one") or {}).get(genre, []))
    item_rec = {k.split("--", 1)[1]: v for k, v in answers["items"].items() if k.startswith(genre + "--")}
    removed = {i for i, v in item_rec.items() if v.get("removed")}

    owners: dict[str, list[str]] = {}
    for c in clusters:
        d = answers["groups"][f"{genre}--{c['id']}"]
        if c["id"] in merge:
            items = [merge[c["id"]]]
            log.append(f"{c['id']}: merged into {items[0]} (resolution; page answer was {d['action']} {d.get('types', [])})")
        elif c["id"] in keep_one:
            items = [slug(c["name"])]
            log.append(f"{c['id']}: kept as one item {items[0]} (resolution; page answer was {d['action']} {d.get('types')})")
        elif d["action"] == "accept":
            items = [] if c["proposal"]["action"] == "prune" else list(c["proposal"]["types"])
        elif d["action"] == "drop":
            items = []
        else:
            items = list(d.get("types") or [])
        dropped = [i for i in items if i in removed]
        if dropped:
            log.append(f"{c['id']}: item(s) {dropped} removed by the reviewer")
        owners[c["id"]] = [i for i in items if i not in removed]

    by_item: dict[str, list[str]] = {}
    for cid, items in owners.items():
        for i in items:
            by_item.setdefault(i, []).append(cid)

    # proposed items that no longer own any group -> replaced by the items now owning their groups
    replaced: dict[str, list[str]] = {}
    for t in proposals["types"]:
        if t["clusters"] and t["id"] not in by_item:
            repl = sorted({i for c in t["clusters"] for i in owners.get(c, [])}, key=list(by_item).index)
            replaced[t["id"]] = repl
            log.append(f"proposed item {t['id']} replaced by {repl or 'nothing (its groups were removed)'}")

    final: list[dict] = []
    for t in proposals["types"]:  # surviving proposals, in proposal order
        if t["id"] in removed or (t["clusters"] and t["id"] not in by_item):
            continue
        t = dict(t)
        if t["clusters"]:
            t["clusters"] = by_item[t["id"]]
        final.append(t)
    carried = {}
    if res.get("carry_over"):
        for old, repl in replaced.items():
            for n in repl:
                carried.setdefault(n, []).append(prop[old])
    for item, cl in by_item.items():  # items the reviewer created
        if item in prop:
            continue
        src = next(c for c in clusters if c["id"] == cl[0])
        rec = item_rec.get(item, {})
        t = {"id": item, "name": item, "definition": rec.get("definition") or src["definition"], "clusters": cl,
             "severity_weight": 1.0, "seeds": [],
             "notes": f"kept as its own item by the reviewer (group {', '.join(cl)})"}
        if page["label_source"]:
            t["label_source"] = ["teacher"]
        for old in carried.get(item, []):
            if old.get("severity_weight", 1.0) > t["severity_weight"]:
                t["severity_weight"] = old["severity_weight"]
            labels = [l for l in old.get("label_source", []) if l != "teacher"]
            if page["label_source"] and labels:
                t["label_source"] = [l for l in t["label_source"] if l != "teacher"] + [l for l in labels if l not in t["label_source"]]
            t["notes"] += f"; severity and label source carried over from {old['id']}"
        if carried.get(item):
            log.append(f"{item}: severity {t['severity_weight']}, labels {t.get('label_source')} (carried over)")
        final.append(t)

    # the reviewer's field edits on the page
    by_id = {t["id"]: t for t in final}
    page_types = {t["id"]: t for t in page["types"]}
    for item, rec in item_rec.items():
        t = by_id.get(item)
        if not t or item not in page_types:
            continue  # a new item's fields on the page were defaults (teacher definition, weight 1.0), not edits
        pt = page_types[item]
        for f, k in (("definition", "definition"), ("section_hint", "section_hint")):
            if rec.get(f) and rec[f] != pt.get(f, t.get(k)) and rec[f] != t.get(k):
                t[k] = rec[f]
                log.append(f"{item}: {k} edited by the reviewer")
        if rec.get("near_miss") and rec["near_miss"] != pt.get("near_miss"):
            t["near_miss"] = rec["near_miss"]
            log.append(f"{item}: near_miss edited by the reviewer")
        if rec.get("severity") is not None and float(rec["severity"]) != float(pt.get("severity", t.get("severity_weight", 1.0))):
            t["severity_weight"] = float(rec["severity"])
            log.append(f"{item}: severity set to {rec['severity']} by the reviewer")

    # definitions widened after the coverage check (resolutions file)
    for item, text in ((res.get("definitions") or {}).get(genre) or {}).items():
        if item not in by_id:
            raise KeyError(f"{genre}: definition given for unknown item {item}")
        by_id[item]["definition"] = text
        by_id[item]["notes"] = (by_id[item].get("notes", "") + "; definition widened after the coverage check").lstrip("; ")
        log.append(f"{item}: definition widened (resolution)")

    # checklist rows, re-pointed through replacements
    for t in final:
        t["seeds"] = []
    seed_excl = {}
    for s in page["seeds"]:
        a = answers["checks"][f"{genre}--{s['id']}"]
        if a.get("exclude"):
            seed_excl[s["id"]] = {"exclude": a["exclude"]}
            continue
        mapped = []
        for i in a.get("types", []):
            if i in by_id:
                mapped.append(i)
            elif i in replaced:
                mapped += replaced[i]
                log.append(f"checklist {s['id']}: {i} -> {replaced[i]}")
            else:
                log.append(f"checklist {s['id']}: {i} was removed by the reviewer, dropped from the mapping")
        mapped = list(dict.fromkeys(mapped))
        if not mapped:
            log.append(f"checklist {s['id']}: NOTHING LEFT; needs a decision")
        for i in mapped:
            by_id[i]["seeds"].append(s["id"])

    prune = {cid: {"action": "prune", "reason": "dropped by the reviewer" if answers["groups"][f"{genre}--{cid}"]["action"] == "drop"
                   else "its item was removed by the reviewer"} for cid, items in owners.items() if not items}
    for cid in prune:
        log.append(f"{cid}: pruned ({prune[cid]['reason']})")
    for t in final:
        t.pop("auto", None)
    return {"types": final, "clusters": prune, "seeds": seed_excl}, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--resolutions", required=True)
    ap.add_argument("--page-data", default="reports/schema_review/page/review_data.json")
    ap.add_argument("--proposals", default="reports/schema_review/proposals_{genre}.yaml")
    ap.add_argument("--out", default="reports/schema_review/final_{genre}.yaml")
    a = ap.parse_args()
    answers = json.loads((REPO / a.answers).read_text(encoding="utf-8"))
    res = yaml.safe_load((REPO / a.resolutions).read_text(encoding="utf-8"))
    page = {g["genre"]: g for g in json.loads((REPO / a.page_data).read_text(encoding="utf-8"))["genres"]}
    for genre in ("privacy", "papers"):
        proposals = yaml.safe_load((REPO / a.proposals.format(genre=genre)).read_text(encoding="utf-8"))
        final, log = resolve(genre, page[genre], proposals, answers, res)
        out = REPO / a.out.format(genre=genre)
        header = (f"# Final reviewed checklist for {genre} (task 3.4), generated by analysis/apply_review_decisions.py from\n"
                  f"# {a.answers} (Parsa's answers on the review page) and {a.resolutions}. Do not edit by hand.\n")
        out.write_text(header + yaml.safe_dump(final, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
        print(f"== {genre}: {len(final['types'])} items, {len(final['clusters'])} pruned groups -> {out.relative_to(REPO)}")
        for line in log:
            print("   ", line)


if __name__ == "__main__":
    main()
