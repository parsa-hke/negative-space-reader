"""analysis/demo_figure.py: the report's demo figure is the app's own rendering of a recorded analysis."""

from analysis.demo_figure import build_html


def test_build_html_shows_the_ghost_after_its_section_only():
    report = {"title": "A policy", "absences": [
        {"type": "states_retention_period", "count": 140, "n": 200, "ci": [0.63, 0.76], "weight": 1.0, "section_hint": "retention",
         "definition": "States how long data is kept.", "evidence": [{"source": "cohort", "text": "We keep data for 24 months.", "doc_id": "privacy_pl:1"}],
         "ghost": {"how": "position", "heading": None}}]}
    sections = [("Intro", "Hello. " * 5), ("Retention", "We keep things. " * 100), ("Contact", "Write to us.")]
    html = build_html(report, sections, max_chars=200)
    assert "nsr-ghost" in html and "states retention period" in html and "140 of 200 peers" in html
    assert "Write to us." not in html and " …" in html  # only the ghost's own section, shortened
    assert "1 section above" in html and "<style>" in html
