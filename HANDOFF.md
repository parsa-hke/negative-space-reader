# Session handoff — 7.6 done (README, fresh-clone smoke); 7.2 planned; 7.7 prepared; report rebuilt; next: Parsa's read, the video, tag and submit

Written 2026-10-25 at the end of the second session of the day. Read with: `CLAUDE.md`, `NEGATIVE_SPACE_PLAN.md`
(§0.2 #21, #24; Phase 7), `CHANGELOG.md` (both entries of 2026-10-25), `reports/submission_v0.1-class.md`,
`reports/demo_video_plan.md`. Replace this file at the end of the next session.

## Where things stand

- **7.4 report (awaiting Parsa's final read):** `reports/report.md`, 3,989 words all-in (3,835 for Sections 1–10;
  `python -m analysis.report --words`), checker zero errors, 30 IEEE references. Parsa's only edit so far (Demo and Code
  links on two lines) is in `reports/report_source.md`; `reports/report.docx` rebuilt from it. **Edit the prose in
  `report_source.md`, never in `report.md`;** then `python -m analysis.report`, `--check`, `--words`, the checker
  (`python3 ~/.claude/skills/my-academic-writing/check_ai_tells.py --no-color reports/report.md`), and the .docx.
- **The .docx converter** lives in a session scratchpad (`docxbuild/md2docx.js`, `docx` npm package installed there;
  `node md2docx.js reports/report.md out.docx`); this session's copy is under
  `/private/tmp/claude-501/-Users-parsa-Documents-negative-space-reader/6c3a6e0b-ee4c-4c7d-94c0-20a24d5e6d2c/scratchpad/docxbuild/`.
  If it is gone: `cd <scratch>/docxbuild && npm install docx` and copy the script from a previous scratchpad, or install
  pandoc (`pandoc reports/report.md -o reports/report.docx --resource-path=reports`). Word must be closed when the
  file is replaced (`reports/~$report.docx` is Word's lock file, now gitignored).
- **7.5:** the GenAI acknowledgement at the end of the report; Parsa confirms or edits the claims before submission.
  **Not cited:** Claude Opus 5.5 / Haiku 5.5 (no vendor page surfaced); add a reference if Parsa has the URL.
- **7.6 done:** README rewritten; `configs/smoke.yaml` redirects packs and embeddings under `data/processed_smoke/`
  and defines the smoke pack build (`embed`, `index`, `cohort_report`); builder bug fixed (pack folder of a pack built
  from nothing); fresh clone: install 60 s, 484 tests passed + 1 skipped, smoke builds 28.6 s / 43.4 s. The bench smoke
  (`bench_eval_readme_s0`) reproduces `reports/tables/bench_synth_<genre>.csv` byte for byte.
- **7.2 planned, not recorded:** `reports/demo_video_plan.md` (shot list, narration gist, timings; MarIA benchmark
  instance as the reveal ending; the Space must be woken first). Examples are pinned and shipped; nothing to change.
- **7.7 prepared, not done:** `reports/submission_v0.1-class.md` (deliverables, tag message and commands, checklist).
  `pyproject.toml` version 0.1.0. **The GitHub repository is private**: make it public or share it with the marker.
- **Hub:** unchanged, 984 matched, the two `space_cpu_timing_<genre>_s0` runs (8 files) not pushed; this session's
  runs (`pack_*_smoke*`, `bench_eval_readme_s0`) are local only. Push with
  `python -m experiments.hub_sync push --bundles runs` when no job is running, before the tag if wanted.
- **Uncommitted (everything from this session):** `.gitignore`, `NEGATIVE_SPACE_PLAN.md` (7.2, 7.4, 7.6 ticked, 7.7
  notes), `README.md`, `analysis/report.py`, `configs/smoke.yaml`, `packs/builder.py`, `pyproject.toml`,
  `reports/report.docx`, `reports/report.md`, `reports/report_source.md`, `runs/log.csv`, `tests/test_builder.py`,
  `tests/test_report.py`, `tests/test_run.py`, `CHANGELOG.md`, `HANDOFF.md`; new `reports/demo_video_plan.md`,
  `reports/submission_v0.1-class.md`. Commit authored by Parsa only, no Claude co-author line.
- Tests: 485 passed (27 s).

## Next session, in order

1. Checks below. Commit what Parsa approves.
2. Parsa's read of the report → edits in `report_source.md` → rebuild (md, check, words, checker, docx).
3. Record the 7.2 video per the plan; export; upload per the outline; link it in the README's demo section; tick 7.2.
4. Make the repository visible to the marker; optional hub push; final `pytest`; commit; tag `v0.1-class`; push
   (`reports/submission_v0.1-class.md` has the commands).
5. Submit (.docx, repository link at the tag, Space link, video); tick 7.4, 7.5, 7.7; GATE passed. Then Phase 8 (MAM).

## Watch

- Never push a hub bundle while a job writes into it.
- FAISS and PyTorch must not both initialise in one macOS process.
- `post_hoc_best` is not a gate result; severity-weighted numbers are an ablation.
- `analyse()` is not bitwise reproducible on MPS (≈ 1e-9 in P(present)); compare reports with rounded floats.
- `space/` is assembled, never committed; the Space is re-uploaded only when Parsa says so; it sleeps after 48 h idle.
- `app/examples/results.json` is generated: rerun `python -m app.examples` when tables or runs change.
- Report prose lives in `reports/report_source.md`; `reports/report.md` is generated and tested against it.
- The smoke overlay must come last (`--overlay local --overlay smoke`); it writes only under `data/processed_smoke/`.
- The smoke pack build covers the label-free stages only; the presence stages need the real corpus and its labels.
- Commits are authored by Parsa only; no Claude co-author line.

## Checks before starting

```bash
.venv/bin/python -m pytest -q        # expected: 485 passed
git status --short                   # this session's changes, unless committed
.venv/bin/python -m analysis.report --check                              # expected: up to date
.venv/bin/python -m experiments.hub_sync status --bundles runs,packs   # expected: 984 matched, 8 differ (space_cpu_timing) until pushed
```

## Prompt for the next session

> Read CLAUDE.md, HANDOFF.md, NEGATIVE_SPACE_PLAN.md (Phase 7) and both CHANGELOG entries of 2026-10-25. We are on
> Phase 7: 7.6 is done, 7.2 is planned, 7.7 is prepared, 7.4 and 7.5 await my read. Run the checks in HANDOFF.md and
> tell me if anything differs. Then: (1) apply my edits to reports/report_source.md, rebuild reports/report.md and
> reports/report.docx, rerun the checker and `python -m analysis.report --words`; (2) walk me through recording the
> 7.2 video from reports/demo_video_plan.md (wake the Space first), then link the exported video in the README; (3)
> when I confirm, commit everything as me, create and push the tag v0.1-class per reports/submission_v0.1-class.md, and
> tick 7.2, 7.4, 7.5, 7.7 in the plan with the date; (4) update HANDOFF.md for Phase 8. Rules as before: every run
> through experiments/run.py with a config; schemas frozen; no §0.2 change without editing the plan first and telling
> me; the Space is re-uploaded only when I say so; commits authored by me only; nothing sent to any API without asking.
