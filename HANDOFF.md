# Session handoff — v0.1-class tagged and public; report final; Parsa submits; next: Phase 8 (MAM)

Written 2026-10-25 at the end of the submission session. Read with: `CLAUDE.md`, `NEGATIVE_SPACE_PLAN.md` (Phase 7 notes,
Phase 8), `CHANGELOG.md` (the three entries of 2026-10-25), `reports/submission_v0.1-class.md`. Replace this file at the
end of the next session.

## Where things stand

- **Submission state:** tag `v0.1-class` = commit `1b5647b` on `main`, pushed; repository **public** at
  https://github.com/parsa-hke/negative-space-reader; licence **Apache 2.0** (`LICENSE`, `NOTICE`); the Space is public
  and always on (Pro). **Parsa submits** the report, the repository link at the tag and the Space link on the portal, then
  ticks 7.7 in the plan (GATE: submitted). No video (7.2 dropped, Parsa's decision).
- **The report:** `reports/report.md` ← `reports/report_source.md` + `reports/tables/report_fragments/` (`python -m
  analysis.tables`, `python -m analysis.report`); `reports/report.tex` + `reports/report_overleaf.zip` (`python -m
  analysis.report_tex --zip ...`); `reports/report.docx` (session docx-js converter, or pandoc). Compiled on Overleaf:
  16 pages, 6 figures, 14 tables (10 in the body, Appendix A), **3,728 words all-in / 3,574 for Sections 1–10 / 3,992 with
  table leads and captions**; checker zero errors. Two rubric reads (marker-style) applied in full; see CHANGELOG.
  Figures: `python -m analysis.figures` (pipeline diagram included) and `python -m analysis.demo_figure` (headless Chrome).
- **Checks that keep it honest:** `tests/test_report.py` (committed md = build), `tests/test_report_tex.py` (committed tex
  = build; 14 tables, 6 figures), `tests/test_tables.py`, `tests/test_figures.py`, `tests/test_demo_figure.py`.
- **Hub:** the two `space_cpu_timing_<genre>_s0` runs and today's smoke / bench_eval_readme runs are local only; push with
  `python -m experiments.hub_sync push --bundles runs` when no job is running.
- Tests: 491 passed (25 s). Tree clean after this commit.

## Next session, in order

1. Checks below. If Parsa has submitted, tick 7.7 with the date (GATE passed).
2. **Phase 8, the masked-assertion model (MAM):** read plan Phase 8 and §0.2 #12; `configs/mam.yaml` and
   `experiments/train_mam.py` are stubs from Phase 1. Smoke first (`--overlay local --overlay smoke`), then the Mac
   overnight or a Colab job per `configs/colab_jobs.yaml`. The report's Section 6 says why: an expectation conditioned
   on what the document states cannot be moved by the absence it is asked to find (the privacy leak).
3. Keep the report's build green: any new table goes through `analysis/tables.py`; `report_source.md` never carries a
   typed number.

## Watch

- Never push a hub bundle while a job writes into it.
- FAISS and PyTorch must not both initialise in one macOS process.
- `post_hoc_best` is not a gate result; severity-weighted numbers are an ablation.
- `analyse()` is not bitwise reproducible on MPS (≈ 1e-9 in P(present)); compare reports with rounded floats.
- `space/` is assembled, never committed; the Space is re-uploaded only when Parsa says so.
- `app/examples/results.json` is generated: rerun `python -m app.examples` when tables or runs change.
- Report prose lives in `reports/report_source.md`; `report.md` and `report.tex` are generated and tested against it.
- The smoke overlay comes last (`--overlay local --overlay smoke`) and writes only under `data/processed_smoke/`.
- Commits are authored by Parsa only; no Claude co-author line. The tag `v0.1-class` is immutable: fixes after it go
  on `main` as new commits, never a moved tag.

## Checks before starting

```bash
.venv/bin/python -m pytest -q        # expected: 491 passed
git status --short                   # expected: clean
.venv/bin/python -m analysis.report --check && .venv/bin/python -m analysis.report_tex --check   # expected: up to date, up to date
git describe --tags                  # expected: v0.1-class (or v0.1-class-N-g... after new commits)
```

## Prompt for the next session

> Read CLAUDE.md, HANDOFF.md, NEGATIVE_SPACE_PLAN.md (Phase 7 notes and Phase 8) and the CHANGELOG entries of
> 2026-10-25. v0.1-class is tagged and public; I have submitted (tick 7.7). We start Phase 8, the masked-assertion
> model. Run the checks in HANDOFF.md and tell me if anything differs. Then: (1) read plan Phase 8 and §0.2 #12 and
> propose the MAM data pipeline and training config as a decision record before writing code; (2) implement
> experiments/train_mam.py with a test and a smoke run through experiments/run.py; (3) report the smoke result in the
> CHANGELOG. Rules as before: every run through experiments/run.py with a config; schemas frozen; no §0.2 change without
> editing the plan first and telling me; the Space is re-uploaded only when I say so; commits authored by me only;
> nothing sent to any API without asking. Update HANDOFF.md at the end and give me the prompt for the next session.
