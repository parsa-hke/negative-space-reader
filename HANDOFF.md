# Session handoff — 7.4 report drafted and polished; 7.5 GenAI section drafted; next 7.2 video, 7.6 README, 7.7 submit

Written 2026-10-25 at the end of the report session. Read with: `CLAUDE.md`, `NEGATIVE_SPACE_PLAN.md` (§0.2 #21, #24;
Phase 7; §11.0–11.1), `CHANGELOG.md` (entry "2026-10-25"), `reports/report.md`. Replace this file at the end of the next
session.

## Where things stand

- **7.4 (drafted, polished, awaiting Parsa's final read):** `reports/report.md`, 3,991 words all-in (title block and
  abstract included; 3,865 for Sections 1–10), excluding tables, captions, references and the GenAI section. Structure:
  abstract; 1 introduction; 2 task; 3 data; 4 architecture; 5 model selection; 6 OmitBench and the privacy turn
  (decision 007); 7 frontier gap, speed, errors; 8 related work; 9 limitations (written first); 10 conclusion; GenAI
  acknowledgement; 30 IEEE references. The AI-tell checker (`~/.claude/skills/my-academic-writing/check_ai_tells.py`)
  reports zero errors. `reports/report.docx` is the submission copy (A4, 16 tables, 4 figures).
- **How the report is built (no number typed by hand):** `python -m analysis.tables` writes
  `reports/tables/report_fragments/*.md` and `values.json` from `reports/tables/*.csv`; `reports/report_source.md`
  holds the prose with `{{v:key|fmt}}`, `<!-- include: name -->` and `[@ref]` placeholders;
  `python -m analysis.report` assembles `reports/report.md` (`--check` fails if it is stale; `tests/test_report.py`
  asserts the committed report equals the build). Edit the prose in `report_source.md`, never in `report.md`.
  Figures: `python -m analysis.figures` (now also `llm_gap.png`).
- **The .docx** was made with a docx-js converter kept in the session scratchpad (pandoc and LibreOffice are not
  installed on this Mac); rebuild = rerun that converter or install pandoc (`pandoc reports/report.md -o
  reports/report.docx --resource-path=reports`). Fonts Calibri, page numbers in the footer, figures at 6.3 in.
- **7.5 (drafted):** the GenAI acknowledgement at the end of the report states the practice from CLAUDE.md; **Parsa
  confirms or edits the claims before submission.**
- **Facts in the report stated qualitatively because they are not in `reports/tables/`:** the Space's cpu-basic
  timings (status line, CHANGELOG 2026-10-21 night 4), the share of Qwen lists in catalogue order (run metrics of
  `llm_gap_candidate_qwen_s0`), the coverage-check rates (plan 3.5). If numbers are wanted, add a stage that writes
  them to a table and a fragment to `analysis/tables.py`.
- **Not cited:** Claude Opus 5.5 and Claude Haiku 5.5 (no Anthropic page surfaced in search; third-party coverage
  only). Add a vendor reference if Parsa has the URL.
- **Hub:** 984 matched, 8 differ / 8 missing on the hub: the two `space_cpu_timing_<genre>_s0` runs made after the last
  push. Push with `python -m experiments.hub_sync push --bundles runs` when convenient (no job running).
- **Uncommitted:** `analysis/tables.py`, `analysis/report.py`, `analysis/figures.py` (llm_gap), `tests/test_tables.py`,
  `tests/test_report.py`, `tests/test_figures.py`, `reports/report.md`, `reports/report_source.md`, `reports/report.docx`,
  `reports/tables/report_fragments/`, `reports/figures/llm_gap.png`, CHANGELOG, HANDOFF. Commit authored by Parsa only.
- Tests: 483 passed.

## Next session, in order

1. Checks below. Commit what Parsa approves (authored by Parsa only; no Claude co-author).
2. Parsa's read of `reports/report.md`: edits go into `reports/report_source.md`, then `python -m analysis.report`,
   the checker, the word count (`tests/test_report.py` keeps report.md in step), and the .docx rebuilt.
3. 7.2: the 3–5 minute walkthrough video on the Space (one policy, one paper; pin examples with `serve.examples`;
   the benchmark reveal makes a good ending; the first drawn privacy example is an adult site).
4. 7.6: README (one-command setup, smoke pack build, bench, runtimes); fresh-clone smoke test.
5. 7.7: tag `v0.1-class`; submit report (.docx), repository link and demo per the outline. GATE: submitted.

## Watch

- Never push a hub bundle while a job writes into it.
- FAISS and PyTorch must not both initialise in one macOS process.
- `post_hoc_best` is not a gate result; severity-weighted numbers are an ablation. The report's leak table excludes the
  post-hoc row by construction (test).
- `analyse()` is not bitwise reproducible on MPS (≈ 1e-9 in P(present)); compare reports with rounded floats.
- `space/` is assembled, never committed; the Space is re-uploaded only when Parsa says so.
- `app/examples/results.json` is generated: rerun `python -m app.examples` when tables or runs change.
- Report prose lives in `reports/report_source.md`; `reports/report.md` is generated and tested against it.
- Commits are authored by Parsa only; no Claude co-author line.

## Checks before starting

```bash
.venv/bin/python -m pytest -q        # expected: 483 passed
git status --short                   # this session's changes, unless committed
.venv/bin/python -m analysis.report --check                              # expected: up to date
.venv/bin/python -m experiments.hub_sync status --bundles runs,packs   # expected: 0 differ after the runs push
```

## Prompt for the next session

> Read CLAUDE.md, HANDOFF.md, NEGATIVE_SPACE_PLAN.md (Phase 7) and the CHANGELOG entry of 2026-10-25. We are on
> Phase 7, tasks 7.2, 7.6 and 7.7; 7.4 and 7.5 are drafted. Run the checks in HANDOFF.md and tell me if anything
> differs. Then: (1) apply my edits to reports/report_source.md, rebuild reports/report.md and reports/report.docx,
> rerun the checker and the word count; (2) plan the 7.2 walkthrough video on the Space (one policy, one paper, the
> benchmark reveal as the ending) and pin the examples; (3) 7.6: README with one-command setup, smoke pack build and
> bench, expected runtimes, then a fresh-clone smoke test; (4) prepare the v0.1-class tag and the submission list for
> 7.7. Rules as before: every run through experiments/run.py with a config; schemas frozen; no §0.2 change without
> editing the plan first and telling me; the Space is re-uploaded only when I say so; commits authored by me only;
> nothing sent to any API without asking. Update HANDOFF.md at the end and give me the prompt for the next session.
