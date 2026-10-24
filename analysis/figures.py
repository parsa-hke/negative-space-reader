"""Every figure in the report, regenerated from the recorded tables (plan 6.4; CLAUDE.md: plots in reports/figures/).

    python -m analysis.figures                    # reports/tables -> reports/figures
    python -m analysis.figures --tables T --out F

Figures (one PNG each, both genres side by side):
  bench_recall5.png         recall@5 with 95% bootstrap CIs per OmitBench-synthetic regime, baselines (a) genre-uniform
                            and (b) level 1, with the reference rows (absence only, random) in grey
  level1_minus_uniform.png  the paired difference level 1 - genre-uniform on recall@5 (the gate 6.6 quantity) on the
                            synthetic regimes and OmitBench-revision (the plan's filter and every added type)
  cohort_diagnosis.png      decision record 007: within-document AUC of cohort prevalence per cohort variant, and on
                            single_type the recall@5 gain over genre-uniform with the cohort retrieved from the edited
                            vs the original document (the deletion leak). post_hoc_best is drawn hollow and labelled:
                            it was selected on the test bench and is not a gate result.
  llm_gap.png               plan 7.3: recall@1/5/10 of the two chat models (candidate-list setting) and the two corpus
                            systems on the same 100 single_type instances per genre, with 95% bootstrap CIs

Inputs: bench_synth_<genre>.csv, bench_revision_<genre>.csv, cohort_diagnosis_<genre>.csv, llm_gap_candidate_<genre>.csv.
Nothing is computed here beyond reading those tables.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

GENRES = ("privacy", "papers")
REGIMES = ("single_type", "three_types", "whole_section")
# Categorical slots in a fixed order (the two baselines); reference rows recede in grey.
COLORS = {"genre_uniform": "#eb6834", "level1": "#2a78d6", "absence_only": "#9a9891", "random": "#c3c2b7"}
LABELS = {"genre_uniform": "(a) genre-uniform", "level1": "(b) level 1 cohort", "absence_only": "absence only",
          "random": "random"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
_NUM = r"[+-]?\d+(?:\.\d+)?"


def parse_interval(cell: Any) -> tuple[float | None, float, float] | None:
    """'+0.0110 [-0.0106, +0.0326]' -> (0.011, -0.0106, 0.0326); '[0.17, 0.18]' -> (None, 0.17, 0.18); '' -> None."""
    if cell is None or not str(cell).strip():
        return None
    m = re.fullmatch(rf"\s*({_NUM})?\s*\[\s*({_NUM})\s*,\s*({_NUM})\s*\]\s*", str(cell))
    if not m:
        raise ValueError(f"not an interval: {cell!r}")
    return (float(m.group(1)) if m.group(1) else None), float(m.group(2)), float(m.group(3))


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row(rows: list[dict[str, str]], **match: str) -> dict[str, str] | None:
    return next((r for r in rows if all(r.get(k) == v for k, v in match.items())), None)


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _figure(ncols: int, width: float, height: float) -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelcolor": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), sharey=True, squeeze=False)
    fig.patch.set_facecolor("white")
    return fig, axes[0]


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def draw_recall(tables: Path, out: Path) -> Path:
    fig, axes = _figure(len(GENRES), 7.2, 2.6)
    systems = list(COLORS)
    for ax, genre in zip(axes, GENRES):
        rows = [r for r in _read(tables / f"bench_synth_{genre}.csv") if r["metric"] == "recall@5"]
        for j, regime in enumerate(REGIMES):
            for s_i, s in enumerate(systems):
                r = _row(rows, regime=regime, system=s)
                if r is None:
                    continue
                y = j + (s_i - (len(systems) - 1) / 2) * 0.17
                v, lo, hi = float(r["value"]), float(r["ci_low"]), float(r["ci_high"])
                ax.plot([lo, hi], [y, y], color=COLORS[s], lw=2, solid_capstyle="round")
                ax.plot(v, y, "o", ms=5, color=COLORS[s], mec="white", mew=1, label=LABELS[s] if j == 0 else None)
        ax.set_yticks(range(len(REGIMES)), [r.replace("_", " ") for r in REGIMES])
        ax.set_xlim(0, 1)
        ax.set_xlabel("recall@5 (95% bootstrap CI over documents)")
        ax.set_title(genre, loc="left", color=INK)
        _style(ax)
    axes[0].invert_yaxis()  # the panels share the y-axis: invert it once
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, -0.48), ncol=4, frameon=False, fontsize=8)
    return _save(fig, out / "bench_recall5.png")


def draw_differences(tables: Path, out: Path) -> Path:
    fig, axes = _figure(len(GENRES), 7.2, 2.6)
    rows_spec = [("bench_synth", r, r.replace("_", " ")) for r in REGIMES] + [
        ("bench_revision", "revision", "revision (plan filter)"),
        ("bench_revision", "revision_all_added", "revision (every added type)")]
    for ax, genre in zip(axes, GENRES):
        ax.axvline(0, color=MUTED, lw=0.8)
        for j, (table, regime, _) in enumerate(rows_spec):
            path = tables / f"{table}_{genre}.csv"
            r = _row(_read(path), regime=regime, system="level1 - genre_uniform", metric="recall@5") if path.exists() else None
            if r is None:
                continue
            v, lo, hi = float(r["value"]), float(r["ci_low"]), float(r["ci_high"])
            color = COLORS["level1"] if lo > 0 else MUTED  # filled blue only when the CI is above zero
            ax.plot([lo, hi], [j, j], color=color, lw=2, solid_capstyle="round")
            ax.plot(v, j, "o", ms=5, color=color, mec="white", mew=1)
            ax.annotate(f"{v:+.3f}", (hi, j), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7,
                        color=MUTED)
        ax.set_yticks(range(len(rows_spec)), [lbl for *_, lbl in rows_spec])
        ax.set_xlabel("recall@5, level 1 − genre-uniform (paired, 95% CI)")
        ax.set_title(genre, loc="left", color=INK)
        _style(ax)
    axes[0].invert_yaxis()  # shared y-axis
    fig.text(0.01, -0.12, "blue: 95% CI above zero; grey: CI includes or is below zero", fontsize=7, color=MUTED)
    lo = min(a.get_xlim()[0] for a in axes)
    hi = max(a.get_xlim()[1] for a in axes)
    for a in axes:
        a.set_xlim(lo, hi + 0.03)
    return _save(fig, out / "level1_minus_uniform.png")


def draw_diagnosis(tables: Path, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelcolor": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    variants = [r["variant"] for r in _read(tables / f"cohort_diagnosis_{GENRES[0]}.csv")]
    # panels 1-2: within-document AUC per variant, one panel per genre
    for ax, genre in zip(axes[:2], GENRES):
        rows = {r["variant"]: r for r in _read(tables / f"cohort_diagnosis_{genre}.csv")}
        base = float(rows["genre_uniform"]["within_doc_auc"]) if "genre_uniform" in rows else None
        if base is not None:
            ax.axvline(base, color=COLORS["genre_uniform"], lw=1, ls="--")
        for j, v in enumerate(variants):
            r = rows.get(v)
            if r is None:
                continue
            _, lo, hi = parse_interval(r["within_doc_auc_ci"])
            post_hoc = r.get("post_hoc") == "True"
            color = COLORS["genre_uniform"] if v == "genre_uniform" else COLORS["level1"]
            ax.plot([lo, hi], [j, j], color=color, lw=2, solid_capstyle="round")
            ax.plot(float(r["within_doc_auc"]), j, "o", ms=5, color="white" if post_hoc else color, mec=color, mew=1.2)
        ax.set_yticks(range(len(variants)), [v.replace("_", " ") + (" (test-selected)" if v == "post_hoc_best" else "")
                                             for v in variants])
        ax.invert_yaxis()
        ax.set_xlabel("within-document AUC of cohort prevalence")
        ax.set_title(genre, loc="left", color=INK)
        _style(ax)
    axes[1].set_yticklabels([])
    # panel 3: the leak, privacy single_type
    ax = axes[2]
    rows = [r for r in _read(tables / f"cohort_diagnosis_{GENRES[0]}.csv")
            if parse_interval(r.get("single_type/edited_minus_uniform"))]
    ax.axvline(0, color=MUTED, lw=0.8)
    for j, r in enumerate(rows):
        post_hoc = r.get("post_hoc") == "True"
        for key, color, dy, label in (("single_type/edited_minus_uniform", COLORS["level1"], -0.15, "edited document"),
                                      ("single_type/original_minus_uniform", COLORS["absence_only"], 0.15,
                                       "original document")):
            v, lo, hi = parse_interval(r[key])
            ax.plot([lo, hi], [j + dy, j + dy], color=color, lw=2, solid_capstyle="round")
            ax.plot(v, j + dy, "o", ms=5, color="white" if post_hoc else color, mec=color, mew=1.2,
                    label=label if j == 0 else None)
    ax.set_yticks(range(len(rows)), [r["variant"].replace("_", " ") + (" (test-selected)"
                                                                       if r.get("post_hoc") == "True" else "") for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(f"{GENRES[0]} single type: recall@5 − genre-uniform")
    ax.set_title("deletion leak: cohort retrieved from the …", loc="left", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out / "cohort_diagnosis.png")


LLM_SYSTEMS = (("level1", COLORS["level1"], "(b) level 1 cohort"), ("genre_uniform", COLORS["genre_uniform"], "(a) genre-uniform"),
               ("anthropic:claude-opus-5-5", "#7b3fb8", "Claude Opus 5.5, candidate list"),
               ("openai:qwen/qwen-2.5-72b-instruct", "#2f9e77", "Qwen2.5-72B, candidate list"))


def draw_llm_gap(tables: Path, out: Path) -> Path | None:
    paths = [tables / f"llm_gap_candidate_{g}.csv" for g in GENRES]
    if not all(p.exists() for p in paths):
        return None
    metrics = ("recall@1", "recall@5", "recall@10")
    fig, axes = _figure(len(GENRES), 7.2, 2.6)
    for ax, genre, path in zip(axes, GENRES, paths):
        rows = _read(path)
        n = rows[0]["n"] if rows else "?"
        for j, metric in enumerate(metrics):
            for s_i, (system, color, label) in enumerate(LLM_SYSTEMS):
                r = _row(rows, system=system)
                if r is None or not r.get(metric):
                    continue
                iv = parse_interval(r.get(f"{metric}_ci"))
                y = j + (s_i - (len(LLM_SYSTEMS) - 1) / 2) * 0.17
                if iv:
                    ax.plot([iv[1], iv[2]], [y, y], color=color, lw=2, solid_capstyle="round")
                ax.plot(float(r[metric]), y, "o", ms=5, color=color, mec="white", mew=1, label=label if j == 0 else None)
        ax.set_yticks(range(len(metrics)), list(metrics))
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"same {n} single-type instances (95% bootstrap CI)")
        ax.set_title(genre, loc="left", color=INK)
        _style(ax)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, -0.55), ncol=2, frameon=False, fontsize=8)
    return _save(fig, out / "llm_gap.png")


def draw_all(tables: Path, out: Path) -> list[Path]:
    tables, out = Path(tables), Path(out)
    figures = [draw_recall(tables, out), draw_differences(tables, out), draw_diagnosis(tables, out), draw_llm_gap(tables, out)]
    return [p for p in figures if p is not None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--out", default="reports/figures")
    a = ap.parse_args()
    for p in draw_all(Path(a.tables), Path(a.out)):
        print(p)


if __name__ == "__main__":
    main()
