# Session handoff — 7.1 live on a ZeroGPU Space and ticked; 7.3 done; next 7.2 video and 7.4 report

Written 2026-10-21 at the end of the first Phase 7 session. Read with: `CLAUDE.md`, `NEGATIVE_SPACE_PLAN.md` (§0.2 #14,
#17, #21, #24; Phase 7), `CHANGELOG.md` (entry "2026-10-21 (afternoon)"), decision record 007. Replace this file at
the end of the next session.

## Where things stand

- **7.1 (ticked):** live at https://huggingface.co/spaces/parsa-hke/negative-space-reader (public, **cpu-basic**; ZeroGPU
  was tried and dropped the same evening: it charges each visitor's own GPU quota, so anonymous visitors hit "runs limit").
  Rebuild and re-upload after any change: `python -m app.space_bundle --clean` (CPU variant is the default) then
  `HfApi().upload_folder(folder_path="space", repo_id="parsa-hke/negative-space-reader", repo_type="space")`.
  Packaging facts in the CHANGELOG (night, 3). Timings on cpu-basic: policy 9.6 s, paper 35.8 s (first calls); a paid
  "CPU upgrade" (8 vCPU) would cut that 3-4× if the video needs it. The Space sleeps after 48 h idle (wakes in ~1 min).
- **7.1 design (Parsa's palette, 2026-10-21 evening):** cream #FFF0C5 page; teal #359FA0 = absent; light teal #8AD6D1 =
  evidence and text-carrying fills; orange #FF8C52 = action, severity, reveals; derived darker teal/orange for small text
  (contrast); containers near-white with solid teal borders and one soft shadow (`--nsr-card`, `--nsr-line`,
  `--nsr-shadow`). CSS variables keep their old names (`--nsr-lime` is the teal). Severity as a badge only when the weight is
  not 1; 7/5 layout with Cohort and Evidence tabs; comparison panel full width below. No `backdrop-filter`/`transform`
  on ancestors of Gradio dropdowns (it detaches their options list).
- **7.3 (ticked):** recall@5 on 100 single_type instances per genre, privacy / papers: Opus 5.5 0.242 / 0.235
  (`llm_gap_candidate_opus_s0`, ≈ $4.2), Qwen2.5-72B 0.137 / 0.181 (`llm_gap_candidate_qwen_s0`, ≈ $0.40), level 1
  0.376 / 0.655, genre-uniform 0.375 / 0.465; both chat models below both corpus systems, paired CIs excluding zero.
  Table `reports/tables/llm_gap_candidate_<genre>.csv`. Caveats and the Qwen catalogue-order finding in the CHANGELOG.
  API keys are exported per command: `set -a; . ./.env; set +a` (OpenRouter: `export OPENROUTER_API_KEY="$OPEN_ROUTER_KEY"`).
- **Gate 5.9 rerun on mains power:** privacy max 1.37 s, papers max 3.28 s; `packs/*/pack.json` and
  `reports/tables/analyse_gate_<genre>.csv` updated.
- **Hub:** runs and packs pushed at the end of the session (986 matched, 0 differ, 0 missing).
- **Committed:** `fc6e4a9` (this session up to the Space upload). **Uncommitted:** the post-upload fixes in
  `app/space_bundle.py`, `tests/test_space_bundle.py`, CHANGELOG, HANDOFF, plan (Parsa reviews; authored by Parsa only).
- Tests: 476 passed.

## Optional 7.3 follow-ups (ask Parsa; each a few dollars)

1. Opus at effort high on the same instances (at medium it used ≈ 220 output tokens per answer, thinking included).
2. Both chat models on the 150 untouched dev documents per genre: false alarms@5, as reported for the system.

## Next session, in order

1. Checks below. Commit what Parsa approves (authored by Parsa only).
2. Any 7.3 follow-up Parsa approves (the table keeps other models' rows; a rerun of a model replaces its own).
3. 7.2: the walkthrough video on the Space (one policy, one paper; the benchmark reveal makes a good ending).
4. 7.2 video, then 7.4 report via `analysis/tables.py`.

## Watch

- Never push a hub bundle while a job writes into it.
- FAISS and PyTorch must not both initialise in one macOS process.
- `post_hoc_best` is not a gate result. Severity-weighted numbers are an ablation, not a gate result. The app ranks
  with severity weights on (the config default); its numbers are not bench results.
- `analyse()` is not bitwise reproducible on MPS (≈ 1e-9 in P(present)); compare reports with rounded floats.
- The pane's stop button kills `serve` without SIGTERM: no metrics. Use Ctrl-C / SIGTERM.
- `space/` is assembled, never committed; rebuild it after any code or pack change before uploading.
- `app/examples/results.json` and `app/examples/*.json` are generated: rerun `python -m app.examples` when tables or runs change.
- Gradio 6: a synthetic Textbox `input` event does not reach the backend (press a hidden button instead); `gr.Tabs(selected=)`
  as a return value is ignored (`gr.update(selected=)`); theme fonts are `gr.themes.Font` objects; verify the app with the
  browser pane visible (a hidden pane throttles timers and drops dropdown focus).
- The first drawn privacy example is an adult site: pin examples before recording the video.
- Commits are authored by Parsa only; no Claude co-author line.

## Checks before starting

```bash
.venv/bin/python -m pytest -q        # expected: 476 passed
git status --short                   # this session's changes, unless committed
.venv/bin/python -m experiments.hub_sync status --bundles runs,packs   # expected: 0 differ, 0 missing
```
