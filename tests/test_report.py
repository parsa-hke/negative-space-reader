"""analysis/report.py: the report is assembled from its source, the fragments and values.json; nothing is typed."""

import json

import pytest

from analysis.report import build


def test_build_fills_values_includes_fragments_and_numbers_citations(tmp_path):
    frag = tmp_path / "frag"
    frag.mkdir()
    (frag / "synth.md").write_text("| a | b |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
    values = {"x/recall": 0.4567, "x/ci": [0.4, 0.5], "x/n": 12345, "x/diff": -0.012}
    refs = {"a": "A. Author, 2020.", "b": "B. Author, 2021."}
    src = ("Recall {{v:x/recall|f3}} {{v:x/ci|ci}} on {{v:x/n}} items, {{v:x/diff|s3}} [@b; @a]. Again [@a].\n\n"
           "<!-- include: synth -->\n*Table 1.*\n\n## References\n\n{{references}}\n")
    out = build(src, values, frag, refs)
    assert "Recall 0.457 [0.400, 0.500] on 12,345 items, -0.012 [1, 2]. Again [2]." in out
    assert "<!-- include: synth.md -->\n\n| a | b |" in out
    assert out.rstrip().endswith("[1] B. Author, 2021.\n[2] A. Author, 2020.")


def test_build_refuses_missing_values_and_references(tmp_path):
    with pytest.raises(KeyError):
        build("{{v:missing}} {{references}}", {}, tmp_path)
    with pytest.raises(KeyError):
        build("[@nobody] {{references}}", {}, tmp_path, {"a": "x"})
    with pytest.raises(FileNotFoundError):
        build("<!-- include: nothere -->\n{{references}}", {}, tmp_path)


def test_committed_report_matches_its_source():
    from pathlib import Path

    from analysis.report import REFS

    frag = Path("reports/tables/report_fragments")
    values = json.loads((frag / "values.json").read_text(encoding="utf-8"))
    built = build(Path("reports/report_source.md").read_text(encoding="utf-8"), values, frag, REFS)
    assert built == Path("reports/report.md").read_text(encoding="utf-8")
