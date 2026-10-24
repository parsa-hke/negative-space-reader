"""Every table of the class report, copied from the recorded tables (plan 7.4: not one number typed by hand).

    python -m analysis.tables                      # reports/tables -> reports/tables/report_fragments/
    python -m analysis.tables --tables T --out F

Reads `reports/tables/*.csv` only (nothing is computed beyond sums, medians, shares and rounding) and writes Markdown
fragments the report includes or quotes, each naming its source tables in an HTML comment on its first line:

  corpus.md        corpus statistics (corpus_stats, dedup, revisions)
  presence.md      presence model vs the teacher ceiling, gate 4.8, document-level presence (presence_ablation,
                   presence_gate, doc_presence)
  synth.md         OmitBench-synthetic recall@5 per regime, system and genre, with 95% bootstrap CIs; false alarms@5
                   and ECE of P(expected) on the untouched labelled documents (bench_synth)
  gate66.md        the gate 6.6 quantity: paired recall@5 difference level 1 - genre-uniform on the synthetic regimes
                   and both OmitBench-revision variants (bench_synth, bench_revision)
  revision.md      OmitBench-revision: how pairs become instances, recall@5 per system (revision_bench, bench_revision)
  leak.md          decision record 007: within-document AUC per cohort variant and the deletion leak
                   (cohort_diagnosis; the test-selected post_hoc_best row is left out, it is not a gate result)
  llm_gap.md       the 7.3 frontier comparison (llm_gap_candidate)
  speed.md         gate 5.9 timings (analyse_gate)
  errors.md        error analysis: recurring false alarms, never-found types, gain by document length (errors)
  severity.md      severity-weight ablation, privacy (ablation_severity); an ablation, not a gate result
  values.json      every value the fragments show, keyed, for inline quotation (`<!-- values: key -->` in the report)

A missing table leaves its fragment out and is listed in `MISSING.md`; nothing is ever filled in by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

GENRES = ("privacy", "papers")
GENRE_NAMES = {"privacy": "privacy policies", "papers": "research papers"}
REGIMES = ("single_type", "three_types", "whole_section")
REGIME_NAMES = {"single_type": "one type", "three_types": "three types", "whole_section": "whole section",
                "revision": "revision (plan filter)", "revision_all_added": "revision (every added type)"}
SYSTEMS = ("genre_uniform", "level1", "random", "absence_only")
SYSTEM_NAMES = {"genre_uniform": "(a) genre-uniform prevalence", "level1": "(b) level-1 cohort prevalence",
                "random": "random order", "absence_only": "absence only (1 − P(present))",
                "anthropic:claude-opus-5-5": "Claude Opus 5.5 (candidate list)",
                "openai:qwen/qwen-2.5-72b-instruct": "Qwen2.5-72B-Instruct (candidate list)"}
DIFF = "level1 - genre_uniform"
_NUM = r"[+-]?\d+(?:\.\d+)?"


class Values:
    """The numbers a fragment shows, keyed, so the report can quote them with their key in a comment."""

    def __init__(self) -> None:
        self.d: "OrderedDict[str, Any]" = OrderedDict()

    def put(self, key: str, value: Any) -> Any:
        self.d[key] = value
        return value


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row(rows: list[dict[str, str]], **match: str) -> dict[str, str] | None:
    return next((r for r in rows if all(r.get(k) == v for k, v in match.items())), None)


def parse_interval(cell: Any) -> tuple[float | None, float, float] | None:
    """'+0.0110 [-0.0106, +0.0326]' -> (0.011, -0.0106, 0.0326); '[0.17, 0.18]' -> (None, 0.17, 0.18); '' -> None."""
    if cell is None or not str(cell).strip():
        return None
    m = re.fullmatch(rf"\s*({_NUM})?\s*\[\s*({_NUM})\s*,\s*({_NUM})\s*\]\s*", str(cell))
    if not m:
        raise ValueError(f"not an interval: {cell!r}")
    return (float(m.group(1)) if m.group(1) else None), float(m.group(2)), float(m.group(3))


def f3(x: Any) -> str:
    return "" if x in (None, "") else f"{float(x):.3f}"


def s3(x: Any) -> str:
    return "" if x in (None, "") else f"{float(x):+.3f}"


def ci(lo: Any, hi: Any, signed: bool = False) -> str:
    if lo in (None, "") or hi in (None, ""):
        return ""
    f = s3 if signed else f3
    return f"[{f(lo)}, {f(hi)}]"


def pct(x: Any, nd: int = 1) -> str:
    return "" if x in (None, "") else f"{100 * float(x):.{nd}f}%"


def table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _source(*names: str) -> str:
    return "<!-- generated by analysis/tables.py from " + ", ".join(names) + "; do not edit -->\n"


# ----------------------------------------------------------------------------------------------------------------- corpus


def corpus_md(tables: Path, V: Values) -> str | None:
    p = tables / "corpus_stats.csv"
    if not p.is_file():
        return None
    stats = _read(p)
    dedup = {r["genre"]: r for r in _read(tables / "dedup.csv")} if (tables / "dedup.csv").is_file() else {}
    revisions = _read(tables / "revisions.csv") if (tables / "revisions.csv").is_file() else []
    genres = [g for g in GENRES if _row(stats, genre=g, split="all")]
    rows: list[list[Any]] = []

    def line(label: str, key: str, fn: Any) -> None:
        rows.append([label] + [V.put(f"corpus/{g}/{key}", fn(g)) for g in genres])

    def split_docs(g: str) -> str:
        parts = [int(_row(stats, genre=g, split=s)["documents"]) for s in ("train", "dev", "test")]
        for s, n in zip(("train", "dev", "test"), parts):
            V.put(f"corpus/{g}/documents_{s}", n)
        V.put(f"corpus/{g}/documents_total", sum(parts))
        return f"{sum(parts):,} ({parts[0]:,} / {parts[1]:,} / {parts[2]:,})"

    line("documents (indexed train / dev / test)", "documents", split_docs)
    line("distinct entities (sites or arXiv ids)", "entities", lambda g: f"{int(_row(stats, genre=g, split='all')['entities']):,}")
    line("sections per document (median)", "sections_per_doc_median",
         lambda g: f"{float(_row(stats, genre=g, split='all')['sections_per_doc_median']):g}")
    line("sentences per section (mean)", "sentences_per_section_mean",
         lambda g: f"{float(_row(stats, genre=g, split='all')['sentences_per_section_mean']):.1f}")
    line("characters per document (median)", "chars_per_doc_median",
         lambda g: f"{int(float(_row(stats, genre=g, split='all')['chars_per_doc_median'])):,}")
    line("dates", "dates", lambda g: f"{_row(stats, genre=g, split='all')['date_min'][:7]} to {_row(stats, genre=g, split='all')['date_max'][:7]}")
    if dedup:
        line("near-duplicate rate in the sample", "near_duplicate_rate",
             lambda g: pct(dedup[g]["near_near_duplicate_rate"]) if g in dedup else "")
        line("documents dropped as cross-split twins", "cross_split_dropped",
             lambda g: f"{int(dedup[g]['cross_split_dropped']):,}" if g in dedup else "")
    if revisions:
        line("revision pairs (test entities)", "revision_pairs",
             lambda g: f"{sum(int(r['pairs']) for r in revisions if r['genre'] == g):,}")
    top = {}
    for g in genres:
        cats = json.loads(_row(stats, genre=g, split="all")["top_categories"])
        top[g] = ", ".join(f"{k} {v:,}" for k, v in list(cats.items())[:5])
    line("largest categories (documents)", "top_categories", lambda g: top[g])
    md = _source("corpus_stats.csv", "dedup.csv", "revisions.csv")
    md += table([""] + [GENRE_NAMES[g] for g in genres], rows)
    return md


# --------------------------------------------------------------------------------------------------------------- presence


def presence_md(tables: Path, V: Values) -> str | None:
    gate_p = tables / "presence_gate.csv"
    if not gate_p.is_file():
        return None
    gate = {r["genre"]: r for r in _read(gate_p)}
    genres = [g for g in GENRES if g in gate]
    md = _source("presence_ablation_<genre>.csv", "presence_gate.csv", "doc_presence_<genre>.csv")
    variants = ["classifier only", "prototypes only", "classifier + prototypes", "teacher labels (ceiling)"]
    header = ["section-level variant"] + [f"{GENRE_NAMES[g]}: macro-F1 / ECE / ms per section" for g in genres]
    rows = []
    for v in variants:
        cells = []
        for g in genres:
            p = tables / f"presence_ablation_{g}.csv"
            r = _row(_read(p), variant=v) if p.is_file() else None
            if r is None:
                cells.append("")
                continue
            key = f"presence/{g}/{v}"
            V.put(key + "/macro_f1", float(r["macro_f1"]))
            if r.get("ece"):
                V.put(key + "/ece", float(r["ece"]))
            if r.get("ms_per_section"):
                V.put(key + "/ms_per_section", float(r["ms_per_section"]))
            cells.append(" / ".join(x for x in (f3(r["macro_f1"]), f3(r.get("ece")) or "–",
                                                f"{float(r['ms_per_section']):.0f}" if r.get("ms_per_section") else "–")))
        rows.append([v] + cells)
    md += "**Section level** (evaluation half of the silver reference set; the ceiling is the teacher's own labels scored the same way):\n\n"
    md += table(header, rows)
    rows = []
    for g in genres:
        r = gate[g]
        V.put(f"presence/{g}/gate/macro_f1", float(r["macro_f1"]))
        V.put(f"presence/{g}/gate/teacher_ceiling", float(r["teacher_ceiling"]))
        V.put(f"presence/{g}/gate/share_of_ceiling", float(r["share_of_ceiling"]))
        V.put(f"presence/{g}/gate/result", r["f1_gate"])
        V.put(f"presence/{g}/gate/sections_per_second", float(r["sections_per_second"]))
        V.put(f"presence/{g}/gate/profiled_documents", int(r["profiled_documents"]))
        rows.append([GENRE_NAMES[g], r["presence_model"], f3(r["macro_f1"]), f3(r["teacher_ceiling"]),
                     f3(r["share_of_ceiling"]) + f" (gate ≥ {float(r['required_share']):.2f})", r["f1_gate"],
                     f"{int(r['profiled_documents']):,} of {int(r['indexable_documents']):,}",
                     f"{float(r['sections_per_second']):.0f} ({r['device']})"])
    md += "\n**Gate 4.8** (macro-F1 relative to the teacher ceiling, decision record 005):\n\n"
    md += table(["genre", "presence model", "macro-F1", "teacher ceiling", "share of ceiling", "gate",
                 "documents profiled", "sections / s"], rows)
    rows = []
    for g in genres:
        p = tables / f"doc_presence_{g}.csv"
        if not p.is_file():
            continue
        drows = _read(p)
        v1 = next((r for r in drows if r["variant"].startswith("v1")), None)
        fitted = [r for r in drows if r.get("cv_log_loss_fit")]
        best = min(fitted, key=lambda r: float(r["cv_log_loss_fit"])) if fitted else None
        for tag, r in (("v1", v1), ("v2", best)):
            if r is None:
                continue
            for k in ("macro_f1", "micro_f1", "ece", "prevalence_abs_error", "types_present_per_document"):
                V.put(f"doc_presence/{g}/{tag}/{k}", float(r[k]))
            V.put(f"doc_presence/{g}/labelled_types_per_document", float(r["labelled_types_per_document"]))
            rows.append([GENRE_NAMES[g], "best section ≥ section threshold (v1)" if tag == "v1" else "document model (v2)",
                         f3(r["macro_f1"]), f3(r["micro_f1"]), f3(r["ece"]), f3(r["prevalence_abs_error"]),
                         f"{float(r['types_present_per_document']):.1f} (labelled {float(r['labelled_types_per_document']):.1f})"])
    if rows:
        md += "\n**Document level** (decision record 006; evaluation half, 75 whole labelled documents per genre, against the judge):\n\n"
        md += table(["genre", "aggregation", "macro-F1", "micro-F1", "ECE", "prevalence error", "types per document"], rows)
    return md


# ------------------------------------------------------------------------------------------------------------------ synth


def synth_md(tables: Path, V: Values) -> str | None:
    data = {g: _read(tables / f"bench_synth_{g}.csv") for g in GENRES if (tables / f"bench_synth_{g}.csv").is_file()}
    if not data:
        return None
    genres = list(data)
    md = _source("bench_synth_<genre>.csv")
    md += "**Recall@5 on OmitBench-synthetic** (500 instances per regime and genre unless stated; 95% bootstrap CI over documents):\n\n"
    header = ["regime", "system"] + [GENRE_NAMES[g] for g in genres]
    rows = []
    for regime in REGIMES:
        for s in SYSTEMS:
            cells = []
            for g in genres:
                r = _row(data[g], regime=regime, system=s, metric="recall@5")
                if r is None:
                    cells.append("")
                    continue
                key = f"synth/{g}/{regime}/{s}/recall@5"
                V.put(key, float(r["value"]))
                V.put(key + "/ci", [float(r["ci_low"]), float(r["ci_high"])])
                V.put(f"synth/{g}/{regime}/n", int(float(r["n"])))
                n = f" (n = {int(float(r['n']))})" if int(float(r["n"])) != 500 else ""
                cells.append(f"{f3(r['value'])} {ci(r['ci_low'], r['ci_high'])}{n}")
            rows.append([REGIME_NAMES[regime] if s == SYSTEMS[0] else "", SYSTEM_NAMES[s]] + cells)
    md += table(header, rows)
    # other metrics, single_type, the two baselines
    rows = []
    for metric in ("recall@1", "recall@10", "mrr", "ndcg@10"):
        for s in ("genre_uniform", "level1"):
            cells = []
            for g in genres:
                r = _row(data[g], regime="single_type", system=s, metric=metric)
                if r is None:
                    cells.append("")
                    continue
                V.put(f"synth/{g}/single_type/{s}/{metric}", float(r["value"]))
                cells.append(f"{f3(r['value'])} {ci(r['ci_low'], r['ci_high'])}")
            rows.append([metric if s == "genre_uniform" else "", SYSTEM_NAMES[s]] + cells)
    md += "\n**Other ranking metrics, one-type regime:**\n\n" + table(["metric", "system"] + [GENRE_NAMES[g] for g in genres], rows)
    rows = []
    for metric, label in (("false_alarm@5", "false alarms@5 (share of the top 5 in fact present)"),
                          ("ece_expected", "ECE of P(expected) against the judge")):
        for s in ("genre_uniform", "level1", "random"):
            cells = []
            for g in genres:
                r = _row(data[g], regime="untouched_labelled", system=s, metric=metric)
                if r is None:
                    cells.append("")
                    continue
                V.put(f"synth/{g}/untouched/{s}/{metric}", float(r["value"]))
                V.put(f"synth/{g}/untouched/n", int(float(r["n"])))
                cells.append(f"{f3(r['value'])} {ci(r['ci_low'], r['ci_high'])}")
            if any(cells):
                rows.append([label if s == "genre_uniform" else "", SYSTEM_NAMES[s]] + cells)
    if rows:
        n = V.d.get(f"synth/{genres[0]}/untouched/n", "")
        md += f"\n**Untouched labelled documents** ({n} per genre, judge labels, out-of-fold document presence):\n\n"
        md += table(["measure", "system"] + [GENRE_NAMES[g] for g in genres], rows)
    return md


# ----------------------------------------------------------------------------------------------------------------- gate66


def gate66_md(tables: Path, V: Values) -> str | None:
    spec = [("bench_synth", r) for r in REGIMES] + [("bench_revision", "revision"), ("bench_revision", "revision_all_added")]
    genres = [g for g in GENRES if (tables / f"bench_synth_{g}.csv").is_file()]
    if not genres:
        return None
    md = _source("bench_synth_<genre>.csv", "bench_revision_<genre>.csv")
    md += "**Recall@5, level 1 − genre-uniform** (paired bootstrap over documents; over entities for the revision rows). The gate 6.6 criterion is the one-type row with a 95% CI above zero. ✓ marks a CI above zero, ✗ a CI below zero.\n\n"
    rows = []
    for tbl, regime in spec:
        cells = []
        for g in genres:
            p = tables / f"{tbl}_{g}.csv"
            r = _row(_read(p), regime=regime, system=DIFF, metric="recall@5") if p.is_file() else None
            if r is None:
                cells.append("")
                continue
            v, lo, hi = float(r["value"]), float(r["ci_low"]), float(r["ci_high"])
            key = f"gate66/{g}/{regime}"
            V.put(key, v)
            V.put(key + "/ci", [lo, hi])
            V.put(key + "/above_zero", lo > 0)
            mark = " ✓" if lo > 0 else (" ✗" if hi < 0 else "")
            cells.append(f"{s3(v)} {ci(lo, hi, signed=True)}{mark} (n = {int(float(r['n']))})")
        rows.append([REGIME_NAMES[regime]] + cells)
    md += table(["benchmark, regime"] + [GENRE_NAMES[g] for g in genres], rows)
    return md


# --------------------------------------------------------------------------------------------------------------- revision


def revision_md(tables: Path, V: Values) -> str | None:
    genres = [g for g in GENRES if (tables / f"bench_revision_{g}.csv").is_file()]
    if not genres:
        return None
    md = _source("revision_bench_<genre>.csv", "bench_revision_<genre>.csv")
    rows = []
    counts_ok = all((tables / f"revision_bench_{g}.csv").is_file() for g in genres)
    if counts_ok:
        for g in genres:
            c = {(r["set"], r["key"]): int(float(r["value"])) for r in _read(tables / f"revision_bench_{g}.csv") if r["kind"] == "pairs"}
            years = [int(r["key"]) for r in _read(tables / f"revision_bench_{g}.csv") if r["kind"] == "instances_per_year" and r["set"] == "revision"]
            for k in ("pairs", "no_added_type", "below_prevalence", "instances"):
                V.put(f"revision/{g}/{k}", c[("revision", k)])
            V.put(f"revision/{g}/instances_all_added", c[("revision_all_added", "instances")])
            V.put(f"revision/{g}/years", [min(years), max(years)] if years else None)
            rows.append([GENRE_NAMES[g], f"{c[('revision', 'pairs')]:,}", f"{c[('revision', 'no_added_type')]:,}",
                         f"{c[('revision', 'below_prevalence')]:,}", f"{c[('revision', 'instances')]:,}",
                         f"{c[('revision_all_added', 'instances')]:,}", f"{min(years)}–{max(years)}" if years else ""])
        md += "**From version pairs to instances** (a pair qualifies when a type added in version n+1 was stated by at least half of the peers dated before version n):\n\n"
        md += table(["genre", "pairs", "no added type", "no added type at ≥ 0.5 prevalence", "instances (plan filter)",
                     "instances (every added type)", "years"], rows)
    rows = []
    for regime in ("revision", "revision_all_added"):
        for s in SYSTEMS:
            cells = []
            for g in genres:
                r = _row(_read(tables / f"bench_revision_{g}.csv"), regime=regime, system=s, metric="recall@5")
                if r is None:
                    cells.append("")
                    continue
                key = f"revision/{g}/{regime}/{s}/recall@5"
                V.put(key, float(r["value"]))
                V.put(key + "/ci", [float(r["ci_low"]), float(r["ci_high"])])
                cells.append(f"{f3(r['value'])} {ci(r['ci_low'], r['ci_high'])}")
            rows.append([REGIME_NAMES[regime] if s == SYSTEMS[0] else "", SYSTEM_NAMES[s]] + cells)
    md += "\n**Recall@5 on OmitBench-revision** (time-windowed cohorts; genre-uniform uses the prevalence of the same date window; bootstrap over entities):\n\n"
    md += table(["variant", "system"] + [GENRE_NAMES[g] for g in genres], rows)
    return md


# ------------------------------------------------------------------------------------------------------------------- leak


def leak_md(tables: Path, V: Values) -> str | None:
    genres = [g for g in GENRES if (tables / f"cohort_diagnosis_{g}.csv").is_file()]
    if not genres:
        return None
    md = _source("cohort_diagnosis_<genre>.csv")
    md += ("**Cohort diagnosis** (decision record 007): how well cohort prevalence predicts which types an untouched test document states "
           "(within-document AUC), and on the one-type regime the recall@5 gain over genre-uniform with the cohort retrieved from the edited "
           "document (what the system sees) and from the original (before the deletion); the leak is their difference. "
           "The variant selected on the test bench (post_hoc_best) is not shown: it is not a gate result.\n\n")
    for g in genres:
        rows = []
        for r in _read(tables / f"cohort_diagnosis_{g}.csv"):
            if r.get("post_hoc") == "True":
                continue
            v = r["variant"]
            auc = float(r["within_doc_auc"]) if r.get("within_doc_auc") else None
            V.put(f"leak/{g}/{v}/within_doc_auc", auc)
            V.put(f"leak/{g}/{v}/brier", float(r["brier"]) if r.get("brier") else None)
            cells = [v.replace("_", " "), f"{f3(r['brier'])}", f"{f3(auc)} {ci(*parse_interval(r['within_doc_auc_ci'])[1:])}" if auc is not None else ""]
            for key in ("single_type/edited_minus_uniform", "single_type/original_minus_uniform", "single_type/leak_original_minus_edited"):
                iv = parse_interval(r.get(key))
                if iv is None:
                    cells.append("")
                    continue
                V.put(f"leak/{g}/{v}/{key.split('/')[1]}", iv[0])
                V.put(f"leak/{g}/{v}/{key.split('/')[1]}/ci", [iv[1], iv[2]])
                cells.append(f"{s3(iv[0])} {ci(iv[1], iv[2], signed=True)}")
            rows.append(cells)
        n = _read(tables / f"cohort_diagnosis_{g}.csv")[0].get("presence_documents", "")
        V.put(f"leak/{g}/presence_documents", int(n) if n else None)
        md += f"*{GENRE_NAMES[g].capitalize()}* (AUC and Brier on {n} untouched test documents; recall@5 differences on the one-type instances):\n\n"
        md += table(["cohort variant", "Brier", "within-document AUC", "edited − uniform", "original − uniform", "leak (original − edited)"], rows) + "\n"
    return md


# ---------------------------------------------------------------------------------------------------------------- llm gap


def llm_gap_md(tables: Path, V: Values) -> str | None:
    genres = [g for g in GENRES if (tables / f"llm_gap_candidate_{g}.csv").is_file()]
    if not genres:
        return None
    md = _source("llm_gap_candidate_<genre>.csv")
    data = {g: _read(tables / f"llm_gap_candidate_{g}.csv") for g in genres}
    systems = [r["system"] for r in data[genres[0]] if " - " not in r["system"]]
    diffs = [r["system"] for r in data[genres[0]] if " - " in r["system"]]
    n = data[genres[0]][0]["n"]
    V.put("llm_gap/n", int(n))
    md += f"**Candidate-list setting, the same {n} one-type instances per genre** (MRR cut at 10 for every system; 95% bootstrap CIs):\n\n"
    header = ["system", "metric"] + [GENRE_NAMES[g] for g in genres]
    rows = []
    for s in systems:
        for metric in ("recall@1", "recall@5", "recall@10", "mrr@10"):
            cells = []
            for g in genres:
                r = _row(data[g], system=s)
                if r is None or not r.get(metric):
                    cells.append("")
                    continue
                iv = parse_interval(r.get(f"{metric}_ci"))
                V.put(f"llm_gap/{g}/{s}/{metric}", float(r[metric]))
                if iv:
                    V.put(f"llm_gap/{g}/{s}/{metric}/ci", [iv[1], iv[2]])
                cells.append(f"{f3(r[metric])} " + (ci(iv[1], iv[2]) if iv else ""))
            rows.append([SYSTEM_NAMES.get(s, s) if metric == "recall@1" else "", metric] + cells)
    md += table(header, rows)
    rows = []
    for s in diffs:
        a, b = s.split(" - ")
        cells = []
        for g in genres:
            r = _row(data[g], system=s)
            if r is None:
                cells.append("")
                continue
            iv = parse_interval(r.get("recall@5_ci"))
            V.put(f"llm_gap/{g}/{s}/recall@5", float(r["recall@5"]))
            if iv:
                V.put(f"llm_gap/{g}/{s}/recall@5/ci", [iv[1], iv[2]])
            cells.append(f"{s3(r['recall@5'])} " + (ci(iv[1], iv[2], signed=True) if iv else ""))
        rows.append([f"{SYSTEM_NAMES.get(a, a).split(' (')[0]} − {SYSTEM_NAMES.get(b, b).split(') ')[-1]}"] + cells)
    md += "\n**Paired differences in recall@5** (chat model − corpus system):\n\n" + table(["difference"] + [GENRE_NAMES[g] for g in genres], rows)
    return md


# ------------------------------------------------------------------------------------------------------------------ speed


def speed_md(tables: Path, V: Values) -> str | None:
    genres = [g for g in GENRES if (tables / f"analyse_gate_{g}.csv").is_file()]
    if not genres:
        return None
    md = _source("analyse_gate_<genre>.csv")
    rows = []
    for g in genres:
        rs = _read(tables / f"analyse_gate_{g}.csv")
        totals = sorted(float(r["total"]) for r in rs)
        embed_share = sum(float(r["embed"]) for r in rs) / max(sum(totals), 1e-9)
        secs = [int(r["sections"]) for r in rs]
        V.put(f"speed/{g}/documents", len(rs))
        V.put(f"speed/{g}/max_seconds", totals[-1])
        V.put(f"speed/{g}/median_seconds", totals[len(totals) // 2])
        V.put(f"speed/{g}/embed_share", round(embed_share, 3))
        V.put(f"speed/{g}/sections", [min(secs), max(secs)])
        rows.append([GENRE_NAMES[g], len(rs), f"{min(secs)}–{max(secs)}", f"{totals[len(totals) // 2]:.2f}", f"{totals[-1]:.2f}", pct(embed_share, 0)])
    md += "**Gate 5.9, end-to-end `analyse()` time per held-out document** (template explanations, after one warm-up call; the machine is named in the text):\n\n"
    md += table(["genre", "documents", "sections", "median s", "max s", "embedding share of time"], rows)
    return md


# ----------------------------------------------------------------------------------------------------------------- errors


def errors_md(tables: Path, V: Values) -> str | None:
    genres = [g for g in GENRES if (tables / f"errors_{g}.csv").is_file()]
    if not genres:
        return None
    md = _source("errors_<genre>.csv")
    md += "**Error analysis of level 1** (run of stage `errors` on the recorded evaluation).\n\n"
    for g in genres:
        rows = _read(tables / f"errors_{g}.csv")
        fa = [r for r in rows if r["analysis"] == "false_alarm" and r["system"] == "level1" and r["metric"] == "false_alarms"]
        fa.sort(key=lambda r: -float(r["value"]))
        top = []
        for r in fa[:5]:
            shown = _row(rows, analysis="false_alarm", system="level1", group=r["group"], metric="in_top5")
            V.put(f"errors/{g}/false_alarm/{r['group']}", [int(float(r["value"])), int(float(shown["value"])) if shown else None])
            top.append([r["group"], f"{int(float(r['value']))} of {int(float(shown['value']))}" if shown else f"{int(float(r['value']))}"])
        md += f"*{GENRE_NAMES[g].capitalize()}, recurring false alarms* (type in the top 5 of an untouched labelled document although the judge finds it stated; shown of appearances):\n\n"
        md += table(["type", "false alarms of appearances in the top 5"], top) + "\n"
        missed = [r for r in rows if r["analysis"] == "missed_type" and r["regime"] == "all" and r["system"] == "level1" and r["metric"] == "recall@5"]
        never = [r for r in missed if float(r["value"]) == 0.0]
        prev = {r["group"]: float(r["value"]) for r in rows if r["analysis"] == "missed_type" and r["regime"] == "all" and r["system"] == "level1" and r["metric"] == "corpus_prevalence"}
        V.put(f"errors/{g}/types_scored", len(missed))
        V.put(f"errors/{g}/types_never_in_top5", len(never))
        V.put(f"errors/{g}/never_types", sorted(r["group"] for r in never))
        if missed:
            med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
            V.put(f"errors/{g}/median_prevalence_never", med([prev[r["group"]] for r in never if r["group"] in prev]))
            V.put(f"errors/{g}/median_prevalence_all", med([prev[r["group"]] for r in missed if r["group"] in prev]))
        md += (f"Types never reaching the top 5 (all regimes): {len(never)} of {len(missed)} scored; median corpus prevalence "
               f"{f3(V.d.get(f'errors/{g}/median_prevalence_never'))} against {f3(V.d.get(f'errors/{g}/median_prevalence_all'))} over all scored types.\n\n")
        length = [r for r in rows if r["analysis"] == "by_length" and r["regime"] == "single_type" and r["system"] == DIFF and r["metric"] == "recall@5"]
        lrows = []
        for i, r in enumerate(length, 1):
            lo, hi = float(r["ci_low"]), float(r["ci_high"])
            V.put(f"errors/{g}/by_length/q{i}", float(r["value"]))
            V.put(f"errors/{g}/by_length/q{i}/ci", [lo, hi])
            lrows.append([f"Q{i}: {r['group'].split(':')[1]} chars", f"{s3(r['value'])} {ci(lo, hi, signed=True)}{' ✓' if lo > 0 else ''}", int(float(r["n"]))])
        if lrows:
            md += f"*{GENRE_NAMES[g].capitalize()}, level 1 − genre-uniform recall@5 by document length (one-type regime):*\n\n"
            md += table(["length quartile", "difference [95% CI]", "n"], lrows) + "\n"
    return md


# --------------------------------------------------------------------------------------------------------------- severity


def severity_md(tables: Path, V: Values) -> str | None:
    genres = [g for g in GENRES if (tables / f"ablation_severity_{g}.csv").is_file()]
    if not genres:
        return None
    md = _source("ablation_severity_<genre>.csv")
    md += ("**Severity weights on** (`rank.use_severity_weights: true`), paired against the w = 1 evaluation. An ablation of the "
           "ranking, not a gate result; a genre whose catalogue has no weight other than 1 is unchanged.\n\n")
    rows = []
    for regime in REGIMES:
        for s, label in (("level1 - reference", "level 1: weighted − unweighted"), ("genre_uniform - reference", "genre-uniform: weighted − unweighted"),
                         ("level1 - genre_uniform", "weighted level 1 − weighted genre-uniform")):
            cells = []
            for g in genres:
                r = _row(_read(tables / f"ablation_severity_{g}.csv"), regime=regime, system=s, metric="recall@5")
                if r is None:
                    cells.append("")
                    continue
                V.put(f"severity/{g}/{regime}/{s}/recall@5", float(r["value"]))
                V.put(f"severity/{g}/{regime}/{s}/recall@5/ci", [float(r["ci_low"]), float(r["ci_high"])])
                cells.append(f"{s3(r['value'])} {ci(r['ci_low'], r['ci_high'], signed=True)}")
            rows.append([REGIME_NAMES[regime] if s == "level1 - reference" else "", label] + cells)
    fa = []
    for s in ("genre_uniform", "level1"):
        cells = []
        for g in genres:
            r = _row(_read(tables / f"ablation_severity_{g}.csv"), regime="untouched_labelled", system=s, metric="false_alarm@5")
            if r is not None:
                V.put(f"severity/{g}/untouched/{s}/false_alarm@5", float(r["value"]))
            cells.append(f"{f3(r['value'])} {ci(r['ci_low'], r['ci_high'])}" if r else "")
        fa.append(["false alarms@5, weighted", SYSTEM_NAMES[s]] + cells)
    md += table(["regime", "recall@5 difference"] + [GENRE_NAMES[g] for g in genres], rows + fa)
    return md


# -------------------------------------------------------------------------------------------------------------- agreement

AGREEMENT_TABLES = (("privacy", "teacher vs judge, 58 types", "teacher_vs_judge_privacy.csv"),
                    ("papers", "teacher vs judge, 47 types (pack v2)", "teacher_vs_judge_v2_papers.csv"),
                    ("privacy", "teacher vs OPP-115 human labels, 35 mapped types", "teacher_vs_opp115_reference_privacy.csv"),
                    ("privacy", "judge vs OPP-115 human labels, 35 mapped types", "teacher_vs_opp115_judge_privacy.csv"))


def agreement_md(tables: Path, V: Values) -> str | None:
    rows = []
    for genre, label, name in AGREEMENT_TABLES:
        p = tables / name
        if not p.is_file():
            continue
        rs = _read(p)
        tp, fp, fn = (sum(int(r[k]) for r in rs) for k in ("tp", "fp", "fn"))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        key = f"agreement/{genre}/{name.removesuffix('.csv')}"
        V.put(key + "/micro_f1", round(f1, 3))
        V.put(key + "/micro_precision", round(prec, 3))
        V.put(key + "/micro_recall", round(rec, 3))
        V.put(key + "/sections", int(rs[0]["sections"]))
        rows.append([GENRE_NAMES[genre], label, int(rs[0]["sections"]), f3(prec), f3(rec), f3(f1)])
    if not rows:
        return None
    md = _source(*(n for *_, n in AGREEMENT_TABLES))
    md += "**Label agreement on the reference sections** (micro precision, recall and F1 pooled over types from the per-type counts):\n\n"
    md += table(["genre", "comparison", "sections", "micro P", "micro R", "micro F1"], rows)
    return md


# ------------------------------------------------------------------------------------------------------------------- main

FRAGMENTS = [("corpus", corpus_md), ("presence", presence_md), ("agreement", agreement_md), ("synth", synth_md),
             ("gate66", gate66_md), ("revision", revision_md), ("leak", leak_md), ("llm_gap", llm_gap_md),
             ("speed", speed_md), ("errors", errors_md), ("severity", severity_md)]


def write_all(tables: Path, out: Path) -> dict[str, Path]:
    tables, out = Path(tables), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    V = Values()
    written: dict[str, Path] = {}
    missing: list[str] = []
    for name, fn in FRAGMENTS:
        md = fn(tables, V)
        if md is None:
            missing.append(name)
            continue
        p = out / f"{name}.md"
        p.write_text(md, encoding="utf-8")
        written[name] = p
    (out / "values.json").write_text(json.dumps(V.d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    written["values"] = out / "values.json"
    (out / "MISSING.md").write_text(("Fragments whose source table is absent: " + ", ".join(missing) if missing else "Every fragment was written.") + "\n",
                                    encoding="utf-8")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--out", default="reports/tables/report_fragments")
    a = ap.parse_args()
    for name, p in write_all(Path(a.tables), Path(a.out)).items():
        print(f"{name:10s} {p}")


if __name__ == "__main__":
    main()
