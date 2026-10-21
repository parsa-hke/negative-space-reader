"""app/content.py and app/about.py (plan 7.1): the About pages' numbers are copied from the recorded tables into
results.json, and the pages read them back; a missing table shows as "not recorded"; no verdict words."""

import csv
import json

import pytest


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def cfg(tmp_path):
    import yaml

    from experiments.config import load_config

    c = load_config("configs/base.yaml", [])
    t = tmp_path / "tables"
    write_csv(t / "corpus_stats.csv", [
        {"genre": "g", "split": "train", "documents": 100, "entities": 90, "sections_per_doc_mean": 8, "sections_per_doc_median": 7,
         "sentences_per_section_mean": 8, "chars_per_doc_median": 5000, "date_min": "2001-01-01", "date_max": "2019-12-31", "sources": "{}", "top_categories": "{}"},
        {"genre": "g", "split": "test", "documents": 10, "entities": 9, "sections_per_doc_mean": 8, "sections_per_doc_median": 7,
         "sentences_per_section_mean": 8, "chars_per_doc_median": 5000, "date_min": "2001-01-01", "date_max": "2019-12-31", "sources": "{}", "top_categories": "{}"}])
    rows = []
    for system, v in (("genre_uniform", 0.40), ("level1", 0.62), ("random", 0.15)):
        for metric in ("recall@1", "recall@5", "recall@10"):
            rows.append({"genre": "g", "regime": "single_type", "system": system, "metric": metric, "value": v, "ci_low": v - 0.03, "ci_high": v + 0.03, "n": 500})
    write_csv(t / "bench_synth_g.csv", rows)
    write_csv(t / "cohort_diagnosis_g.csv", [
        {"genre": "g", "variant": "genre_uniform", "within_doc_auc": 0.85, "single_type/edited_minus_uniform": "", "single_type/original_minus_uniform": "", "single_type/leak_original_minus_edited": ""},
        {"genre": "g", "variant": "default", "within_doc_auc": 0.88, "single_type/edited_minus_uniform": "+0.2200 [+0.1800, +0.2600]",
         "single_type/original_minus_uniform": "+0.2500 [+0.2100, +0.2900]", "single_type/leak_original_minus_edited": "+0.0300 [+0.0100, +0.0500]"}])
    write_csv(t / "llm_gap_candidate_g.csv", [
        {"genre": "g", "system": "anthropic:claude-opus-5-5", "n": 100, "recall@1": 0.05, "recall@1_ci": "[0.02, 0.08]", "recall@5": 0.24, "recall@5_ci": "[0.18, 0.30]",
         "recall@10": 0.4, "recall@10_ci": "[0.3, 0.5]", "mrr@10": 0.3, "mrr@10_ci": "[0.2, 0.4]"},
        {"genre": "g", "system": "level1", "n": 100, "recall@1": 0.1, "recall@1_ci": "[0.07, 0.13]", "recall@5": 0.62, "recall@5_ci": "[0.55, 0.69]",
         "recall@10": 0.8, "recall@10_ci": "[0.7, 0.9]", "mrr@10": 0.5, "mrr@10_ci": "[0.4, 0.6]"},
        {"genre": "g", "system": "anthropic:claude-opus-5-5 - level1", "n": 100, "recall@1": -0.05, "recall@1_ci": "[-0.08, -0.02]", "recall@5": -0.38, "recall@5_ci": "[-0.45, -0.3]",
         "recall@10": -0.4, "recall@10_ci": "[-0.5, -0.3]", "mrr@10": -0.2, "mrr@10_ci": "[-0.3, -0.1]"}])
    write_csv(t / "analyse_gate_g.csv", [{"doc_id": f"d{i}", "sections": 5 + i, "absences": 10, "total": 0.5 + 0.1 * i} for i in range(5)])
    write_csv(t / "presence_gate.csv", [{"genre": "g", "presence_model": "classifier + prototypes", "macro_f1": 0.6, "teacher_ceiling": 0.75,
                                         "share_of_ceiling": 0.8, "required_share": 0.8, "f1_gate": "pass"}])
    packs = tmp_path / "packs" / "g"
    packs.mkdir(parents=True)
    (packs / "schema.yaml").write_text(yaml.safe_dump({"types": [{"id": f"t{i}", "severity_weight": 1.5 if i == 0 else 1.0} for i in range(7)]}))
    c["paths"].update(tables=str(t), packs=str(tmp_path / "packs"))
    c["genres"] = ["g"]
    return c


def test_results_are_copied_from_the_tables(cfg, tmp_path):
    from app.content import build_results, read_results, write_results

    r = build_results(cfg)
    assert r["corpus"]["g"]["indexed"] == 100 and r["corpus"]["g"]["documents"] == 110 and r["schema"]["g"] == {"types": 7, "examples": [f"t{i}" for i in range(6)], "weighted": 1}
    assert r["synthetic"]["g"]["level1"]["recall@5"]["value"] == 0.62 and r["synthetic"]["g"]["random"]["recall@5"]["n"] == 500
    assert r["gate"]["g"]["edited_minus_uniform"] == {"diff": 0.22, "ci": [0.18, 0.26]} and r["gate"]["g"]["uniform_within_doc_auc"] == 0.85
    assert r["llm_gap"]["g"]["anthropic:claude-opus-5-5"]["recall@5"] == {"value": 0.24, "ci": [0.18, 0.30]}
    assert r["speed"]["g"]["max_seconds"] == pytest.approx(0.9) and r["presence"]["g"]["gate"] == "pass"
    assert "revision" in r and r["revision"] == {} and "bench_synth_g.csv" in r["sources"]
    write_results(cfg, tmp_path / "out")
    assert read_results(tmp_path / "out")["schema"]["g"]["types"] == 7 and read_results(tmp_path / "nowhere") is None


def test_about_pages_show_the_numbers_and_no_verdicts(cfg):
    from app.about import SECTIONS, about_html
    from app.content import build_results
    from app.gradio_app import FORBIDDEN

    h = about_html(build_results(cfg))
    for sid, title, _ in SECTIONS:
        assert f"id='{sid}'" in h and title in h
    assert "62%" in h and "40%" in h and "24%" in h and "+0.220" in h and "a real gain" in h
    assert "Table bench_revision not recorded" in h and "100 " in h
    low = h.lower().replace("never says a document is non-compliant or deficient", "")  # the one sentence that names them
    assert not any(w in low for w in FORBIDDEN)


def test_about_pages_degrade_without_results():
    from app.about import about_html

    h = about_html(None)
    assert "not recorded" in h and "<nav class='nsr-toc'>" in h and "tens of thousands of documents" in h
