"""analysis/figures.py: the bench figures are drawn from the recorded tables alone (plan 6.4)."""

import csv

import pytest

from analysis.figures import draw_all, parse_interval


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_parse_interval_reads_the_diagnosis_cells():
    assert parse_interval("+0.0110 [-0.0106, +0.0326]") == pytest.approx((0.011, -0.0106, 0.0326))
    value, low, high = parse_interval("[0.1739, 0.1787]")
    assert value is None and (low, high) == pytest.approx((0.1739, 0.1787))
    assert parse_interval("") is None and parse_interval(None) is None


def test_draw_all_writes_one_png_per_figure(tmp_path):
    tables, out = tmp_path / "tables", tmp_path / "figures"
    tables.mkdir()
    for g in ("privacy", "papers"):
        synth = []
        for regime in ("single_type", "three_types", "whole_section"):
            for s, v in (("genre_uniform", 0.4), ("level1", 0.5), ("absence_only", 0.05), ("random", 0.2)):
                synth.append({"genre": g, "regime": regime, "system": s, "metric": "recall@5", "value": v,
                              "ci_low": v - 0.03, "ci_high": v + 0.03, "n": 10})
            synth.append({"genre": g, "regime": regime, "system": "level1 - genre_uniform", "metric": "recall@5",
                          "value": 0.1, "ci_low": 0.05, "ci_high": 0.15, "n": 10})
        _write(tables / f"bench_synth_{g}.csv", synth)
        _write(tables / f"bench_revision_{g}.csv",
               [{"genre": g, "regime": r, "system": "level1 - genre_uniform", "metric": "recall@5", "value": 0.0,
                 "ci_low": -0.01, "ci_high": 0.01, "n": 5} for r in ("revision", "revision_all_added")])
        diag = [{"genre": g, "variant": "genre_uniform", "post_hoc": "False", "within_doc_auc": 0.85,
                 "within_doc_auc_ci": "[0.84, 0.86]", "single_type/edited_minus_uniform": "",
                 "single_type/original_minus_uniform": ""}]
        for v, ph in (("default", "False"), ("k50", "False"), ("post_hoc_best", "True")):
            diag.append({"genre": g, "variant": v, "post_hoc": ph, "within_doc_auc": 0.9,
                         "within_doc_auc_ci": "[0.89, 0.91]",
                         "single_type/edited_minus_uniform": "+0.0100 [-0.0100, +0.0300]",
                         "single_type/original_minus_uniform": "+0.0300 [+0.0100, +0.0500]"})
        _write(tables / f"cohort_diagnosis_{g}.csv", diag)
    written = draw_all(tables, out)
    assert sorted(p.name for p in written) == ["bench_recall5.png", "cohort_diagnosis.png", "level1_minus_uniform.png"]
    assert all(p.stat().st_size > 1000 for p in written)
