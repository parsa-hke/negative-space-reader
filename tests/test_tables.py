"""analysis/tables.py: the report's tables are copied from the recorded CSVs (plan 7.4); nothing is typed by hand."""

import csv
import json

import pytest

from analysis.tables import parse_interval, write_all


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _bench_rows(genre, regimes, n=10):
    rows = []
    for regime in regimes:
        for s, v in (("genre_uniform", 0.4), ("level1", 0.5), ("absence_only", 0.05), ("random", 0.2)):
            for metric in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"):
                rows.append({"genre": genre, "regime": regime, "system": s, "metric": metric, "value": v,
                             "ci_low": v - 0.03, "ci_high": v + 0.03, "n": n})
        rows.append({"genre": genre, "regime": regime, "system": "level1 - genre_uniform", "metric": "recall@5",
                     "value": 0.1 if genre == "papers" else 0.01, "ci_low": 0.05 if genre == "papers" else -0.01,
                     "ci_high": 0.15 if genre == "papers" else 0.03, "n": n})
    return rows


@pytest.fixture
def tables(tmp_path):
    t = tmp_path / "tables"
    t.mkdir()
    stats = []
    for g in ("privacy", "papers"):
        for split, docs in (("train", 200), ("dev", 30), ("test", 25), ("all", 255)):
            stats.append({"genre": g, "split": split, "documents": docs, "entities": docs - 1, "sections_per_doc_mean": 8.6,
                          "sections_per_doc_median": 7.0, "sentences_per_section_mean": 8.09, "chars_per_doc_median": 5796.5,
                          "date_min": "1998-07-12", "date_max": "2019-12-31", "sources": "{}",
                          "top_categories": json.dumps({"education": 40, "business": 39})})
    _write(t / "corpus_stats.csv", stats)
    _write(t / "dedup.csv", [{"genre": g, "near_near_duplicate_rate": 0.1856, "cross_split_dropped": 19} for g in ("privacy", "papers")])
    _write(t / "revisions.csv", [{"genre": g, "year": y, "pairs": 5, "median_change": 0.2} for g in ("privacy", "papers") for y in (2018, 2019)])
    _write(t / "presence_gate.csv", [
        {"genre": "privacy", "presence_model": "classifier + prototypes", "macro_f1": 0.619, "teacher_ceiling": 0.773, "share_of_ceiling": 0.801,
         "required_share": 0.8, "f1_gate": "pass", "profiled_documents": 200, "indexable_documents": 200, "profiles_complete": True,
         "sections_per_second": 133.6, "machine": "arm64 Darwin", "device": "mps", "reference": "silver"},
        {"genre": "papers", "presence_model": "classifier + prototypes", "macro_f1": 0.497, "teacher_ceiling": 0.688, "share_of_ceiling": 0.722,
         "required_share": 0.8, "f1_gate": "not met", "profiled_documents": 200, "indexable_documents": 200, "profiles_complete": True,
         "sections_per_second": 65.2, "machine": "arm64 Darwin", "device": "mps", "reference": "silver"}])
    for g in ("privacy", "papers"):
        _write(t / f"presence_ablation_{g}.csv", [
            {"variant": "classifier only", "macro_f1": 0.617, "ece": 0.061, "ms_per_section": 107.4},
            {"variant": "prototypes only", "macro_f1": 0.417, "ece": "", "ms_per_section": 5.0},
            {"variant": "classifier + prototypes", "macro_f1": 0.619, "ece": 0.003, "ms_per_section": 112.5},
            {"variant": "teacher labels (ceiling)", "macro_f1": 0.773, "ece": "", "ms_per_section": ""}])
        _write(t / f"doc_presence_{g}.csv", [
            {"variant": "v1: best section >= section threshold", "cv_log_loss_fit": "", "macro_f1": 0.687, "micro_f1": 0.739, "ece": 0.15,
             "prevalence_abs_error": 0.136, "types_present_per_document": 22.76, "labelled_types_per_document": 26.76},
            {"variant": "document model: logit_max", "cv_log_loss_fit": 0.392, "macro_f1": 0.753, "micro_f1": 0.807, "ece": 0.016,
             "prevalence_abs_error": 0.067, "types_present_per_document": 25.37, "labelled_types_per_document": 26.76},
            {"variant": "document model: four", "cv_log_loss_fit": 0.39, "macro_f1": 0.754, "micro_f1": 0.808, "ece": 0.018,
             "prevalence_abs_error": 0.063, "types_present_per_document": 26.09, "labelled_types_per_document": 26.76}])
        synth = _bench_rows(g, ("single_type", "three_types", "whole_section"))
        for s, fa, ece in (("genre_uniform", 0.205, 0.048), ("level1", 0.223, 0.067), ("random", 0.176, "")):
            synth.append({"genre": g, "regime": "untouched_labelled", "system": s, "metric": "false_alarm@5", "value": fa, "ci_low": fa - 0.03, "ci_high": fa + 0.03, "n": 150})
            if ece != "":
                synth.append({"genre": g, "regime": "untouched_labelled", "system": s, "metric": "ece_expected", "value": ece, "ci_low": ece - 0.01, "ci_high": ece + 0.02, "n": 150})
        _write(t / f"bench_synth_{g}.csv", synth)
        _write(t / f"bench_revision_{g}.csv", _bench_rows(g, ("revision", "revision_all_added"), n=5))
        _write(t / f"revision_bench_{g}.csv", [
            {"genre": g, "kind": "pairs", "set": "revision", "key": k, "value": v} for k, v in (("pairs", 40), ("no_added_type", 13), ("below_prevalence", 9), ("instances", 18))
        ] + [{"genre": g, "kind": "pairs", "set": "revision_all_added", "key": "instances", "value": 27},
             {"genre": g, "kind": "instances_per_year", "set": "revision", "key": 2018, "value": 8},
             {"genre": g, "kind": "instances_per_year", "set": "revision", "key": 2019, "value": 10}])
        diag = [{"genre": g, "variant": "genre_uniform", "post_hoc": "False", "presence_documents": 80, "brier": 0.176, "within_doc_auc": 0.854,
                 "within_doc_auc_ci": "[0.8497, 0.8587]", "single_type/edited_minus_uniform": "", "single_type/original_minus_uniform": "",
                 "single_type/leak_original_minus_edited": ""}]
        for v, ph in (("default", "False"), ("no_title_no_category", "False"), ("post_hoc_best", "True")):
            diag.append({"genre": g, "variant": v, "post_hoc": ph, "presence_documents": 80, "brier": 0.17, "within_doc_auc": 0.88,
                         "within_doc_auc_ci": "[0.8755, 0.8834]", "single_type/edited_minus_uniform": "+0.0110 [-0.0106, +0.0326]",
                         "single_type/original_minus_uniform": "+0.0260 [+0.0049, +0.0472]", "single_type/leak_original_minus_edited": "+0.0150 [+0.0052, +0.0257]"})
        _write(t / f"cohort_diagnosis_{g}.csv", diag)
        llm = []
        for s, r5 in (("anthropic:claude-opus-5-5", 0.242), ("openai:qwen/qwen-2.5-72b-instruct", 0.137), ("level1", 0.376), ("genre_uniform", 0.375),
                      ("anthropic:claude-opus-5-5 - level1", -0.133), ("anthropic:claude-opus-5-5 - genre_uniform", -0.132),
                      ("openai:qwen/qwen-2.5-72b-instruct - level1", -0.239), ("openai:qwen/qwen-2.5-72b-instruct - genre_uniform", -0.238)):
            row = {"genre": g, "system": s, "n": 100}
            for m in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"):
                row[m] = r5
                row[f"{m}_ci"] = f"[{r5 - 0.05:.3f}, {r5 + 0.05:.3f}]"
            llm.append(row)
        _write(t / f"llm_gap_candidate_{g}.csv", llm)
        _write(t / f"analyse_gate_{g}.csv", [{"doc_id": f"{g}:{i}", "sections": 4 + i, "absences": 10, "read": 0, "segment": 0, "embed": 0.25 * (i + 1),
                                              "presence": 0.05, "document_presence": 0, "document_vector": 0, "cohort": 0.009, "expectation": 0,
                                              "rank": 0, "evidence": 0, "explain": 0, "total": 0.25 * (i + 1) + 0.06} for i in range(5)])
        errors = []
        for tp, fa_n, shown in (("browser_cookie_controls", 11, 41), ("provides_third_party_links", 10, 26), ("security_not_guaranteed", 9, 26)):
            errors.append({"genre": g, "analysis": "false_alarm", "regime": "untouched_labelled", "system": "level1", "group": tp, "metric": "false_alarms", "value": fa_n, "ci_low": "", "ci_high": "", "n": 150})
            errors.append({"genre": g, "analysis": "false_alarm", "regime": "untouched_labelled", "system": "level1", "group": tp, "metric": "in_top5", "value": shown, "ci_low": "", "ci_high": "", "n": 150})
        for tp, r5, prev in (("california_disclosure_rights", 0.0, 0.05), ("uses_cookies", 0.8, 0.87)):
            errors.append({"genre": g, "analysis": "missed_type", "regime": "all", "system": "level1", "group": tp, "metric": "recall@5", "value": r5, "ci_low": 0, "ci_high": 1, "n": 20})
            errors.append({"genre": g, "analysis": "missed_type", "regime": "all", "system": "level1", "group": tp, "metric": "corpus_prevalence", "value": prev, "ci_low": "", "ci_high": "", "n": 20})
        for i in range(1, 5):
            errors.append({"genre": g, "analysis": "by_length", "regime": "single_type", "system": "level1 - genre_uniform", "group": f"q{i}:100..200",
                           "metric": "recall@5", "value": 0.04, "ci_low": 0.002, "ci_high": 0.08, "n": 125})
        _write(t / f"errors_{g}.csv", errors)
    sev = []
    for regime in ("single_type", "three_types", "whole_section"):
        for s in ("level1 - reference", "genre_uniform - reference", "level1 - genre_uniform"):
            sev.append({"genre": "privacy", "regime": regime, "system": s, "metric": "recall@5", "value": 0.005, "ci_low": -0.024, "ci_high": 0.032, "n": 500})
    for s in ("genre_uniform", "level1"):
        sev.append({"genre": "privacy", "regime": "untouched_labelled", "system": s, "metric": "false_alarm@5", "value": 0.245, "ci_low": 0.21, "ci_high": 0.28, "n": 150})
    _write(t / "ablation_severity_privacy.csv", sev)
    for name in ("teacher_vs_judge_privacy.csv", "teacher_vs_judge_v2_papers.csv", "teacher_vs_opp115_reference_privacy.csv",
                 "teacher_vs_opp115_judge_privacy.csv"):
        _write(t / name, [{"type_id": "a", "sections": 300, "human_positives": 10, "tp": 8, "fp": 2, "fn": 2},
                          {"type_id": "b", "sections": 300, "human_positives": 5, "tp": 4, "fp": 2, "fn": 1}])
    return t


def test_parse_interval():
    assert parse_interval("+0.0110 [-0.0106, +0.0326]") == pytest.approx((0.011, -0.0106, 0.0326))
    assert parse_interval("[0.17, 0.18]")[1:] == pytest.approx((0.17, 0.18))
    assert parse_interval("") is None


def test_every_fragment_is_written_from_the_tables(tables, tmp_path):
    out = tmp_path / "fragments"
    written = write_all(tables, out)
    names = {"corpus", "presence", "agreement", "synth", "gate66", "revision", "leak", "llm_gap", "speed", "errors", "severity", "values"}
    assert set(written) == names
    assert (out / "MISSING.md").read_text(encoding="utf-8").startswith("Every fragment")
    for name in names - {"values"}:
        text = (out / f"{name}.md").read_text(encoding="utf-8")
        assert text.startswith("<!-- generated by analysis/tables.py from"), name
        assert "| " in text, name
    synth = (out / "synth.md").read_text(encoding="utf-8")
    assert "0.500 [0.470, 0.530]" in synth and "privacy policies" in synth and "research papers" in synth
    assert "0.205 [0.175, 0.235]" in synth  # false alarms copied
    gate = (out / "gate66.md").read_text(encoding="utf-8")
    assert "+0.100 [+0.050, +0.150] ✓" in gate and "+0.010 [-0.010, +0.030] (n = 10)" in gate
    llm = (out / "llm_gap.md").read_text(encoding="utf-8")
    assert "Claude Opus 5.5" in llm and "-0.133 [-0.183, -0.083]" in llm
    corpus = (out / "corpus.md").read_text(encoding="utf-8")
    assert "255 (200 / 30 / 25)" in corpus and "18.6%" in corpus and "1998-07 to 2019-12" in corpus


def test_post_hoc_variant_is_never_shown_and_values_are_keyed(tables, tmp_path):
    out = tmp_path / "fragments"
    write_all(tables, out)
    leak = (out / "leak.md").read_text(encoding="utf-8")
    assert "post hoc best" not in leak and "+0.015 [+0.005, +0.026]" in leak
    values = json.loads((out / "values.json").read_text(encoding="utf-8"))
    assert values["gate66/papers/single_type"] == pytest.approx(0.1) and values["gate66/papers/single_type/above_zero"] is True
    assert values["gate66/privacy/single_type/above_zero"] is False
    assert values["synth/privacy/single_type/level1/recall@5"] == pytest.approx(0.5)
    assert values["presence/papers/gate/result"] == "not met"
    assert values["doc_presence/privacy/v2/macro_f1"] == pytest.approx(0.754)  # the lowest cross-validated loss
    assert values["revision/privacy/instances"] == 18 and values["revision/privacy/years"] == [2018, 2019]
    assert values["speed/privacy/documents"] == 5 and values["errors/privacy/types_never_in_top5"] == 1
    # pooled counts: tp 12, fp 4, fn 3 -> P 0.75, R 0.8, F1 0.774
    assert values["agreement/privacy/teacher_vs_judge_privacy/micro_f1"] == pytest.approx(0.774, abs=0.001)
    assert not any("post_hoc" in k for k in values)


def test_a_missing_table_leaves_its_fragment_out(tables, tmp_path):
    (tables / "llm_gap_candidate_privacy.csv").unlink()
    (tables / "llm_gap_candidate_papers.csv").unlink()
    out = tmp_path / "fragments"
    written = write_all(tables, out)
    assert "llm_gap" not in written
    assert "llm_gap" in (out / "MISSING.md").read_text(encoding="utf-8")
