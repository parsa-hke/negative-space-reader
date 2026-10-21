"""analysis/error_analysis.py: missed types, recurring false alarms and errors by length, date, category (plan 6.5)."""

import pytest

from analysis.error_analysis import (category_groups, distractor_rows, error_tables, false_alarm_rows, group_rows,
                                     quantile_bins, type_rows)

BOOT = dict(n_boot=50, ci=0.95, seed=0)


def inst(i, gt, targets=None, regime="single_type", date="2015-01-01", category="a", length=1000, p_after=None):
    targets = targets or gt[:1]
    return {"instance_id": f"{regime}:d{i}", "regime": regime, "ground_truth": gt, "targets": targets,
            "date": date, "category": category, "length": length, "p_after": p_after or {t: 0.3 for t in gt}}


def ranked(i, system, top5, regime="single_type", r5=None, mrr=None):
    return {"instance_id": f"{regime}:d{i}", "regime": regime, "system": system, "top5": "|".join(top5),
            "recall@5": r5, "mrr": mrr}


def test_type_rows_count_how_often_each_ground_truth_type_reaches_the_top5():
    I = [inst(0, ["x"]), inst(1, ["x", "y"], targets=["x"]), inst(2, ["y"])]
    R = [ranked(0, "level1", ["x", "a"]), ranked(1, "level1", ["a", "y"]), ranked(2, "level1", ["b"])]
    rows = {(r["group"], r["metric"]): r for r in type_rows(I, R, ["level1"], BOOT, regimes=["single_type"],
                                                             prevalence={"x": 0.8})
            if r["regime"] == "single_type"}
    assert rows[("x", "recall@5")]["value"] == 0.5 and rows[("x", "recall@5")]["n"] == 2
    assert rows[("y", "recall@5")]["value"] == 0.5
    assert rows[("y", "target_share")]["value"] == 0.5     # y was a target once (d2) and collateral once (d1)
    assert rows[("x", "p_present_after")]["value"] == pytest.approx(0.3)
    assert rows[("x", "corpus_prevalence")]["value"] == 0.8 and ("y", "corpus_prevalence") not in rows


def test_distractors_are_top5_types_outside_the_ground_truth():
    I = [inst(0, ["x"]), inst(1, ["y"])]
    R = [ranked(0, "level1", ["a", "x", "b"]), ranked(1, "level1", ["a", "y"])]
    rows = {r["group"]: r for r in distractor_rows(I, R, ["level1"], regimes=["single_type"])}
    assert rows["a"]["value"] == 1.0 and rows["a"]["n"] == 2   # in the top 5 of both instances
    assert rows["b"]["value"] == 0.5 and "x" not in rows


def test_false_alarm_rows_per_type_on_untouched_documents():
    U = [{"doc_id": "u0", "system": "level1", "top5": "a|b", "present": "a"},
         {"doc_id": "u1", "system": "level1", "top5": "a|c", "present": ""}]
    rows = {(r["group"], r["metric"]): r["value"] for r in false_alarm_rows(U, ["level1"])}
    assert rows[("a", "false_alarms")] == 1 and rows[("a", "in_top5")] == 2 and rows[("a", "false_alarm_share")] == 0.5
    assert rows[("b", "false_alarms")] == 0 and ("c", "false_alarms") in rows


def test_quantile_bins_and_small_categories():
    labels = quantile_bins([1, 2, 3, 4, 5, 6, 7, 8], 4)
    assert len(set(labels)) == 4 and labels[0] == labels[1] and labels[0] != labels[-1]
    assert quantile_bins(["2010-01-01", "2011-06-01", "2019-01-01", "2020-01-01"], 2, year=True)[0] == "q1:2010-01..2011-06"
    assert category_groups(["a", "a", "a", "b"], min_n=2) == ["a", "a", "a", "other"]


def test_group_rows_give_each_system_and_the_paired_difference():
    I = [inst(k, ["x"], length=100 * (k + 1)) for k in range(8)]
    R = ([ranked(k, "level1", [], r5=1.0, mrr=1.0) for k in range(8)]
         + [ranked(k, "genre_uniform", [], r5=0.0 if k < 4 else 1.0, mrr=0.5) for k in range(8)])
    groups = {i["instance_id"]: ("short" if i["length"] <= 400 else "long") for i in I}
    rows = group_rows("by_length", groups, R, ["genre_uniform", "level1"], ["recall@5"], BOOT, regimes=["single_type"])
    by = {(r["group"], r["system"]): r for r in rows}
    assert by[("short", "level1 - genre_uniform")]["value"] == 1.0
    assert by[("long", "level1 - genre_uniform")]["value"] == 0.0 and by[("long", "level1")]["n"] == 4


def test_error_tables_has_every_analysis():
    I = [inst(k, ["x"], date=f"201{k}-01-01", category="a" if k < 6 else "b", length=100 * (k + 1)) for k in range(8)]
    R = [ranked(k, s, ["x"] if k % 2 else ["z"], r5=float(k % 2), mrr=float(k % 2))
         for k in range(8) for s in ("genre_uniform", "level1")]
    U = [{"doc_id": "u0", "system": s, "top5": "x", "present": "x"} for s in ("genre_uniform", "level1")]
    rows = error_tables("g", I, R, U, systems=["genre_uniform", "level1"], regimes=["single_type"], boot=BOOT,
                        length_bins=2, date_bins=2, min_category=2)
    assert {r["analysis"] for r in rows} == {"missed_type", "distractor", "false_alarm", "by_length", "by_date",
                                             "by_category"}
    assert all(r["genre"] == "g" for r in rows)
    assert {r["group"] for r in rows if r["analysis"] == "by_category"} == {"a", "b"}
