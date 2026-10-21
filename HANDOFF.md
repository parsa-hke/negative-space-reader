# Session handoff — 7.1 redesigned and verified (approval pending); 7.3 done

Written 2026-10-21 at the end of the first Phase 7 session. Read with: `CLAUDE.md`, `NEGATIVE_SPACE_PLAN.md` (§0.2 #14,
#17, #21, #24; Phase 7), `CHANGELOG.md` (entry "2026-10-21 (afternoon)"), decision record 007. Replace this file at
the end of the next session.

## Where things stand

- **7.1 (not ticked, design awaiting Parsa's approval):** `app/gradio_app.py` redesigned per Parsa's brief (bold dark
  design, animations, lens cursor and particle trail: `app/static/nsr.css`, `nsr.js`), pinned examples with the cached
  7.3 chat-model answers (`app/examples.py` → `app/examples/*.json`, config `serve.examples`, `serve.bench_examples`,
  `serve.chat_runs`, `serve.examples_dir`), the "compare with a chat model" switch (cached answers only;
  `serve.allow_live_chat: false`), ghost click → Evidence, guided demo (five-step strip, narrated status with the numbers,
  progress narration, challenge/reveal for benchmark examples) and the **About the project** tab (`app/about.py`;
  numbers from `app/examples/results.json`, written by `python -m app.examples` via `app/content.py`; rerun it after any
  table changes). Verified in the browser (CHANGELOG, evening entries 1-4). Launch: `python -m experiments.run --config configs/serve.yaml`; stop with Ctrl-C.
  **Design approved (2026-10-21 evening).** **Space bundle ready, not uploaded:** `python -m app.space_bundle --clean`
  → `space/` (305 MB, 149 MB LFS; corpus-free). Two variants: `--variant zerogpu` (default; Parsa's Pro account: H200
  slice per analysis via `spaces.GPU`, untestable locally) and `--variant cpu` (verified by running `space/app.py`
  locally: 4.6 s per paper). Proposal in CHANGELOG (night entries); waiting for Parsa's go-ahead on the Space name and
  visibility; hardware `zero-a10g`/H200 ZeroGPU at creation (`create_repo(..., space_sdk="gradio", space_hardware=...)`).
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
- **Uncommitted:** everything from this session (Parsa reviews first; commits authored by Parsa only).
- Tests: 476 passed.

## Optional 7.3 follow-ups (ask Parsa; each a few dollars)

1. Opus at effort high on the same instances (at medium it used ≈ 220 output tokens per answer, thinking included).
2. Both chat models on the 150 untouched dev documents per genre: false alarms@5, as reported for the system.

## Next session, in order

1. Checks below. Commit what Parsa approves (authored by Parsa only).
2. Any 7.3 follow-up Parsa approves (the table keeps other models' rows; a rerun of a model replaces its own).
3. Upload the Space once Parsa says where (`space/` is the complete repo content; LFS for `*.npy`): ZeroGPU first;
   watch the build log for the `spaces` patch and the first analysis; if it fails, rebuild `--variant cpu` and upload
   that. Time a policy and a paper on the Space, record both, then tick 7.1.
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
