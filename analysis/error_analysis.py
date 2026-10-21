"""Missed types and recurring false alarms by length, date, category (plan §6.5).

Pure functions over a recorded eval run; the stage `bench.stage: errors` (experiments/bench.py) reads the files and
writes reports/tables/errors_<genre>.csv. One long table, one row per (analysis, regime, system, group, metric):

  missed_type   per ground-truth type (OmitBench-synthetic): recall@5 of that type (share of its occurrences that
                reach the top 5), its count, the share of occurrences where it was a sampled target (the rest are
                collateral), its median P(present) after the edit (the (1 - P) factor of the score) and its
                corpus prevalence (the indexed documents' profiles)
  distractor    per type: share of instances whose top 5 holds it although it is not ground truth. On synthetic
                documents such a type is absent by the presence model, so a distractor is not necessarily a false alarm
  false_alarm   per type, on the untouched judge-labelled dev documents (§0.2 #15, decision record 006): how often it is
                in the top 5, how often it is in fact present (a false alarm), and that share
  by_length     recall@5 and MRR per quantile bin of the original document's length in characters, per system and the
  by_date       paired difference level1 - genre_uniform; dates in quantile bins labelled by year and month;
  by_category   categories with fewer than min_category instances pooled as "other"

regime "all" pools the regimes for the per-type analyses (a type occurs a few times per regime only); the group
analyses stay per regime, because recall is not comparable across regimes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from bench.metrics import bootstrap_mean, paired_bootstrap

DIFF = ("level1", "genre_uniform")


def _row(genre: str | None, analysis: str, regime: str, system: str, group: str, metric: str, value: float,
         n: int, lo: float | None = None, hi: float | None = None) -> dict[str, Any]:
    r = lambda x: None if x is None else round(float(x), 4)  # noqa: E731
    return {"genre": genre, "analysis": analysis, "regime": regime, "system": system, "group": group,
            "metric": metric, "value": r(value), "ci_low": r(lo), "ci_high": r(hi), "n": int(n)}


def _top5(r: dict[str, Any]) -> list[str]:
    return [t for t in str(r.get("top5") or "").split("|") if t]


def _scopes(regimes: Sequence[str]) -> list[tuple[str, set[str]]]:
    return [(g, {g}) for g in regimes] + ([("all", set(regimes))] if len(regimes) > 1 else [])


def type_rows(instances: list[dict[str, Any]], ranked: list[dict[str, Any]], systems: Sequence[str],
              boot: dict[str, Any], regimes: Sequence[str],
              prevalence: dict[str, float] | None = None) -> list[dict[str, Any]]:
    by_id = {i["instance_id"]: i for i in instances}
    out = []
    for scope, regs in _scopes(regimes):
        for s in systems:
            hits: dict[str, list[float]] = defaultdict(list)
            target: dict[str, list[float]] = defaultdict(list)
            p_after: dict[str, list[float]] = defaultdict(list)
            for r in ranked:
                if r["system"] != s or r["regime"] not in regs or r["instance_id"] not in by_id:
                    continue
                i, top = by_id[r["instance_id"]], set(_top5(r))
                for t in i["ground_truth"]:
                    hits[t].append(float(t in top))
                    target[t].append(float(t in i["targets"]))
                    if t in (i.get("p_after") or {}):
                        p_after[t].append(float(i["p_after"][t]))
            for t in sorted(hits):
                v, lo, hi = bootstrap_mean(hits[t], **boot)
                out.append(_row(None, "missed_type", scope, s, t, "recall@5", v, len(hits[t]), lo, hi))
                out.append(_row(None, "missed_type", scope, s, t, "target_share", np.mean(target[t]), len(hits[t])))
                if p_after[t]:
                    out.append(_row(None, "missed_type", scope, s, t, "p_present_after", np.median(p_after[t]),
                                    len(p_after[t])))
                if prevalence and t in prevalence:
                    out.append(_row(None, "missed_type", scope, s, t, "corpus_prevalence", prevalence[t], len(hits[t])))
    return out


def distractor_rows(instances: list[dict[str, Any]], ranked: list[dict[str, Any]], systems: Sequence[str],
                    regimes: Sequence[str]) -> list[dict[str, Any]]:
    by_id = {i["instance_id"]: i for i in instances}
    out = []
    for scope, regs in _scopes(regimes):
        for s in systems:
            sel = [r for r in ranked if r["system"] == s and r["regime"] in regs and r["instance_id"] in by_id]
            c = Counter(t for r in sel for t in _top5(r) if t not in set(by_id[r["instance_id"]]["ground_truth"]))
            for t, k in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
                out.append(_row(None, "distractor", scope, s, t, "top5_share", k / len(sel), len(sel)))
    return out


def false_alarm_rows(untouched: list[dict[str, Any]], systems: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for s in systems:
        sel = [u for u in untouched if u["system"] == s]
        shown, wrong = Counter(), Counter()
        for u in sel:
            present = {t for t in str(u.get("present") or "").split("|") if t}
            for t in _top5(u):
                shown[t] += 1
                wrong[t] += t in present
        for t in sorted(shown, key=lambda t: (-wrong[t], -shown[t], t)):
            out.append(_row(None, "false_alarm", "untouched_labelled", s, t, "false_alarms", wrong[t], len(sel)))
            out.append(_row(None, "false_alarm", "untouched_labelled", s, t, "in_top5", shown[t], len(sel)))
            out.append(_row(None, "false_alarm", "untouched_labelled", s, t, "false_alarm_share", wrong[t] / shown[t],
                            shown[t]))
    return out


def quantile_bins(values: Sequence[Any], q: int, *, year: bool = False) -> list[str]:
    """Each value's quantile bin (q bins, ties kept together), labelled by its range; year=True reads ISO dates and
    labels the bins by year range."""
    x = np.array([float(str(v)[:4]) + (int(str(v)[5:7]) - 1) / 12 if year else float(v) for v in values])
    edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, max(len(edges) - 2, 0))
    def fmt(v: float) -> str:  # dates as year-month: quantile bins of a recent corpus fall inside one year
        y = int(v)
        return f"{y}-{int(round((v - y) * 12)) + 1:02d}" if year else str(int(v))

    labels = []
    for b in range(max(len(edges) - 1, 1)):
        members = x[idx == b]
        lo, hi = (members.min(), members.max()) if len(members) else (edges[b], edges[min(b + 1, len(edges) - 1)])
        labels.append(f"{fmt(lo)}..{fmt(hi)}")
    return [f"q{b + 1}:{labels[b]}" for b in idx]


def category_groups(values: Sequence[Any], min_n: int) -> list[str]:
    c = Counter(str(v) for v in values)
    return [str(v) if c[str(v)] >= min_n else "other" for v in values]


def group_rows(analysis: str, groups: dict[str, str], ranked: list[dict[str, Any]], systems: Sequence[str],
               metrics: Sequence[str], boot: dict[str, Any], regimes: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for regime in regimes:
        for g in sorted({groups[r["instance_id"]] for r in ranked if r["regime"] == regime and r["instance_id"] in groups}):
            sel = {s: {r["instance_id"]: r for r in ranked if r["system"] == s and r["regime"] == regime
                       and groups.get(r["instance_id"]) == g} for s in systems}
            for m in metrics:
                for s in systems:
                    if sel[s]:
                        v, lo, hi = bootstrap_mean([float(r[m]) for r in sel[s].values()], **boot)
                        out.append(_row(None, analysis, regime, s, g, m, v, len(sel[s]), lo, hi))
                if all(s in systems for s in DIFF):
                    ids = sorted(sel[DIFF[0]].keys() & sel[DIFF[1]].keys())
                    if ids:
                        d = paired_bootstrap([float(sel[DIFF[0]][x][m]) for x in ids],
                                             [float(sel[DIFF[1]][x][m]) for x in ids], **boot)
                        out.append(_row(None, analysis, regime, f"{DIFF[0]} - {DIFF[1]}", g, m, d["diff"], d["n"],
                                        d["ci_low"], d["ci_high"]))
    return out


def error_tables(genre: str, instances: list[dict[str, Any]], ranked: list[dict[str, Any]],
                 untouched: list[dict[str, Any]], *, systems: Sequence[str], regimes: Sequence[str],
                 boot: dict[str, Any], length_bins: int = 4, date_bins: int = 4, min_category: int = 20,
                 metrics: Sequence[str] = ("recall@5", "mrr"),
                 prevalence: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Every analysis of the module docstring. instances need instance_id, regime, ground_truth, targets, p_after,
    date, category and length (characters of the original document); prevalence (type -> share of indexed documents
    stating it) adds that to the per-type rows."""
    ranked = [r for r in ranked if r["system"] in systems]
    rows = type_rows(instances, ranked, systems, boot, regimes, prevalence)
    rows += distractor_rows(instances, ranked, systems, regimes)
    rows += false_alarm_rows(untouched, systems)
    for analysis, labels in (
            ("by_length", _per_regime(instances, lambda xs: quantile_bins([i["length"] for i in xs], length_bins))),
            ("by_date", _per_regime(instances, lambda xs: quantile_bins([i["date"] for i in xs], date_bins, year=True))),
            ("by_category", _per_regime(instances, lambda xs: category_groups([i.get("category") or "none" for i in xs],
                                                                              min_category)))):
        rows += group_rows(analysis, labels, ranked, systems, metrics, boot, regimes)
    return [{**r, "genre": genre} for r in rows]


def _per_regime(instances: list[dict[str, Any]], label: Any) -> dict[str, str]:
    """Bins are drawn within each regime (the regimes hold different documents)."""
    out: dict[str, str] = {}
    for regime in dict.fromkeys(i["regime"] for i in instances):
        xs = [i for i in instances if i["regime"] == regime]
        out.update(zip([i["instance_id"] for i in xs], label(xs)))
    return out
