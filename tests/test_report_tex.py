"""analysis/report_tex.py: the LaTeX export of the assembled report, for Overleaf."""

import re
from pathlib import Path

import pytest

from analysis.report_tex import convert, escape, figures_used, inline

MD = """# A title: with *emphasis*

A. Author · A course · A university · October 2026

Demo: https://example.org/demo

## Abstract

Four words, 50% & more.

## 1. First section

Peers beat frequency [1, 2]. Code `a_b` and **bold *nested***; see https://example.org/x. A score ≥ 0.5 × 2 − 1.

| a | b |
|---|---|
| x_1 | 0.5 |

**Panel two** (details):

| c | d | e |
|---|---|---|
| 1 | 2 | 3 |

*Table 1. Two panels.*

![Figure 1](figures/f.png)
*Figure 1. A figure.*

## GenAI acknowledgement

Written with help.

## References

[1] A. Author, "A paper," in *Proc. X*, 2020, pp. 1–2.
[2] B. Author, "Another," arXiv:0000.00000, 2021.
"""


def test_inline_escapes_and_maps_markup_urls_and_citations():
    assert escape("50% & a_b #1 {x}") == r"50\% \& a\_b \#1 \{x\}"
    assert inline("see [1, 2] and `a_b` at https://e.org/p_q.") == r"see \cite{r1,r2} and \texttt{a\_b} at \url{https://e.org/p_q}."
    assert inline("**bold *in***") == r"\textbf{bold \emph{in}}"
    assert inline("≥ × −", cite=False) == r"$\geq$ $\times$ $-$"
    assert inline(r"in *Proc. (\*SEM 2023)*, x") == r"in \emph{Proc. (*SEM 2023)}, x"  # an escaped asterisk is literal
    with pytest.raises(ValueError):
        escape("a ☃ b")


def test_convert_builds_title_abstract_sections_floats_and_bibliography():
    tex = convert(MD)
    assert r"\title{A title: with \emph{emphasis}}" in tex and r"\author{A. Author \\ A course \\ A university}" in tex
    assert r"\date{October 2026}" in tex and r"\url{https://example.org/demo}" in tex
    assert tex.count(r"\begin{abstract}") == tex.count(r"\end{abstract}") == 1
    assert r"\section{First section}" in tex and r"\section*{GenAI acknowledgement}" in tex
    assert tex.count(r"\begin{table}") == 1 and tex.count(r"\begin{tabular}") == 2  # two panels, one float
    assert r"\panel{\textbf{Panel two} (details):}" in tex and r"\caption{Two panels.}" in tex
    assert r"\includegraphics[width=\linewidth]{figures/f.png}" in tex and figures_used(tex) == ["figures/f.png"]
    assert r"\bibitem{r1} A. Author, ``A paper,'' in \emph{Proc. X}, 2020, pp. 1–2." in tex
    assert tex.count(r"\bibitem") == 2 and r"\cite{r1,r2}" in tex
    assert "References" not in tex  # thebibliography prints the heading itself
    assert tex.rstrip().endswith(r"\end{document}") and tex.count("{") == tex.count("}")


def test_a_lead_paragraph_before_a_table_joins_its_float_and_long_columns_wrap():
    from analysis.report_tex import column_spec

    md = MD.replace("| a | b |\n", "**Lead title** (details):\n\n| a | b |\n")
    tex = convert(md)
    assert r"\panel{\textbf{Lead title} (details):}" in tex and tex.index(r"\begin{table}") < tex.index("Lead title")
    assert column_spec([["a", "b"], ["1", "2"]], 2) == "ll"
    assert column_spec([["name", "a very long description " * 4, "n"], ["x", "y", "1"]], 3).count("P{") == 1


def test_convert_refuses_a_table_without_a_caption():
    with pytest.raises(ValueError):
        convert("# T\n\nA · B · C · D\n\n## 1. S\n\n| a |\n|---|\n| 1 |\n\n## References\n\n[1] X.\n")


def test_committed_tex_matches_the_report():
    tex = convert(Path("reports/report.md").read_text(encoding="utf-8"))
    assert tex == Path("reports/report.tex").read_text(encoding="utf-8")
    for env in set(re.findall(r"\\begin\{(\w+)\}", tex)):
        assert tex.count(f"\\begin{{{env}}}") == tex.count(f"\\end{{{env}}}"), env
    assert tex.count(r"\begin{table}") == 9 and tex.count(r"\begin{figure}") == 4 and tex.count(r"\bibitem") == 30
    for rel in figures_used(tex):
        assert (Path("reports") / rel).is_file(), rel
