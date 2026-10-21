# Changelog

Date, what was done, what was found (negative findings included). Newest first.

## 2026-10-21 (night, 3) — 7.1 closed: the demo is live on a ZeroGPU Space

**Committed first** (`fc6e4a9`, Parsa's authorship, Parsa's request), then `parsa-hke/negative-space-reader` created
(public, Gradio, hardware `zero-a10g`, which is the ZeroGPU pool) and `space/` uploaded with `huggingface_hub`
(`upload_folder`, 305 MB, 17 s). **Four builds to green**, each a packaging fact worth keeping:
1. `short_description` in the Space header is capped at 60 characters.
2. ZeroGPU images build on **Python 3.10** whatever `python_version` says: `numpy==2.4.6`, `pandas==3.0.6`,
   `pyarrow==26.0.0`, `scikit-learn==1.9.1` have no 3.10 wheels. The zerogpu bundle now pins the latest releases that
   support 3.10 (2.2.6, 2.3.3, 25.0.1, 1.7.2; checked against PyPI `requires_python`); our models are arrays and
   JSON, not sklearn pickles, so the older scikit-learn is safe.
3. The builder installs `gradio[oauth,mcp]`, whose `mcp` extra caps pydantic at 2.12.5: pydantic is a range now.
4. The Hub stores `.npz` and `.parquet` as LFS, but a Space checkout smudges only the patterns in the repo's own
   `.gitattributes`; with `*.npy` alone the npz and parquet files arrived as pointer text and `np.load` fell through
   to its pickle path ("This file contains pickled (object) data"). `.gitattributes` now lists npy, npz, parquet.
**Running:** both packs loaded in 13.4 s on `cuda` (the `spaces` patch accepted the models' `.to("cuda")` at
startup: "ZeroGPU tensors packing 1.35 G"). **Timings on the Space** (status line, same documents as the CPU
measurements): Etilize policy **1.95 s** on the first call (GPU attach included); FANS paper **0.52 s** first,
**0.28 s** warm (Mac CPU: 1.8 s and 5.6 s; the papers gate on MPS: 3.3 s). Results identical to the local app
(privacy: 200 peers, 172 entities, similarity 0.769, 24 types stated; papers: 0.741, 24 stated). URL:
https://huggingface.co/spaces/parsa-hke/negative-space-reader. Plan 7.1 ticked. Not yet done: the Space is on the
free ZeroGPU quota rules (visitors get a GPU-time allowance per day; the owner's Pro quota applies when logged in),
and it sleeps after 48 h idle. The post-commit fixes (bundle pins, LFS patterns, tests, these notes) are uncommitted.
Tests: 476.

## 2026-10-21 (night, 2) — 7.1: ZeroGPU variant of the Space bundle (Parsa has a Hugging Face Pro account)

Pro accounts can create **ZeroGPU** Spaces: shared H200 slices attached only while a decorated function runs, at no
charge beyond the subscription. Embedding is 90% of an analysis, so a paper should take about a second there instead
of 15-30 s on a free CPU Space. **Code:** `serve.zerogpu: true` wraps the analyse handler in `spaces.GPU(duration=
serve.zerogpu_seconds)` (the only handler that touches the models; re-ranking and the evidence panel stay on the
CPU); `configs/space_zerogpu.yaml` (after `space.yaml`: `run.device cuda`, fp16, batch 8); `python -m app.space_bundle
--variant zerogpu` (now the default) writes an `app.py` that imports `spaces` before torch, a `requirements.txt` with
PyPI torch (CUDA build) and `spaces`, and both overlays; `--variant cpu` keeps the CPU bundle. **Not verified:**
ZeroGPU exists only on the Space; locally the wrapper is tested with a fake `spaces` module and the bundle's `app.py`
parses. Known risks to watch on the first build: the `spaces` patch must see the models' `.to("cuda")` at startup
(it supports that), and the wrapped handler runs in a worker, so the per-analysis `record` list is not filled there
(metrics only; the Space logs no run). Fallback if ZeroGPU misbehaves: the CPU bundle, unchanged. A template bug
found by the tests: `app.py` holds f-strings, so the bundle fills it by token replacement, not `str.format`.
Tests: 476.

## 2026-10-21 (night) — 7.1: the Space bundle (assembled and run locally; nothing uploaded)

**Approved by Parsa:** package for a Hugging Face CPU Space; propose before uploading. **Measured first** (Mac M5):
what the app reads of each pack is `schema.yaml`, `pack.json`, `cascade.json`, `doc_presence.json`, `classifier/`,
`prototypes/`, `profiles/`, `evidence/` and `index/{index.json, payloads.jsonl, vectors.npy, best_sections.npz}`:
privacy 181 MB, papers 124 MB (section_probs, profiles_v1, drafts and label sets are training artefacts, left out).
The only files over Hugging Face's 10 MB git limit are the two `vectors.npy` (91 + 58 MB → Git LFS). Models download
from the Hub at start (Qwen3-Embedding-0.6B ≈ 1.2 GB, answerai-colbert-small-v1 ≈ 0.13 GB). **Memory** with both
packs and fp32 models on CPU: 4.5 GB after loading, 6.1 GB peak during analyses (a free Space has 16 GB). **CPU
timing** (run `space_cpu_timing_<genre>_s0`, `configs/analyse.yaml`, `run.device=cpu`): Etilize policy 1.8 s, FANS
paper 5.6 s on 10 cores; with 2 threads 1.6 s / 6.2 s (embedding is 90% of it). A free Space's 2 x86 vCPUs are slower
than an M5: expect roughly 4-8 s for a policy and 15-30 s for a paper; to be measured on the Space itself.

**Code:** `packs/spec.py section_texts_or_none`: a pack with a stored evidence set loads without the corpus (the
Space ships no `data/processed`); without a store the corpus is still required and the error says so.
`app/gradio_app.py build_from_config` is shared by run.py's `serve` (non-blocking, logs a run) and the Space's
`app.py` (blocking). `configs/space.yaml` overlay: CPU, fp32, 0.0.0.0, both packs, nothing that calls an API.
**`app/space_bundle.py`** (`python -m app.space_bundle --clean`, tests `tests/test_space_bundle.py`) assembles
`space/` (gitignored): the Python of app / engine / packs / data / experiments / models / bench / analysis, app's
static assets and example files, configs, prompts, the pack subsets, and the Space's `app.py`, `README.md` (Space
header: gradio 6.30.0, Python 3.11), `requirements.txt` (pins from this environment, PyTorch from the CPU wheel
index) and `.gitattributes` (LFS for `*.npy`); `BUNDLE.json` records sizes. **Result:** 307 MB, 149 MB of it LFS; no
missing artefact. **Run from inside `space/` with the project's interpreter:** loaded in 11.9 s on CPU, 6 examples;
MarIA with the comparison on gave the same output as the dev app (mean similarity 0.781, 5 reveal badges, quotes from
the stored evidence set, no prototype fallback) in 4.6 s. A first bundle copied `data/processed/splits` and
`label_sets.json`; the copier now takes Python only from code directories (test added). Tests: 475.

## 2026-10-21 (evening, 4) — 7.1: containers lifted off the background (Parsa: more contrast, everywhere)

Every container, not only the About steps: surfaces are now near-white (`--nsr-card` rgba(255,252,242,0.92), inputs
white) instead of a 3-5% teal tint on the cream; borders are solid teal at 60% alpha (`--nsr-line`, 1.5 px on cards,
2 px dashed on ghosts) with a softer 30% line (`--nsr-line-soft`) for inner dividers; cards, steps, guide cells, tiles,
columns, folds, callouts, banners and the results table carry one soft teal shadow (`--nsr-shadow`); ghosts and
quotes sit on a light-teal tint (20-22%); bar and ring tracks are darker; the aurora is a little calmer (0.38). Checked
in the pane on the demo (guide, input card) and About (steps). Tests 472.

## 2026-10-21 (evening, 3) — 7.1: Parsa's palette (cream, teal, light teal, orange); the detached dropdown list fixed

**Parsa:** the options list opened far from its field; use the Color Hunt palette #8AD6D1 / #359FA0 / #FFF0C5 / #FF8C52
for the whole page. **Dropdown:** the card's `backdrop-filter` made it the containing block for Gradio's
fixed-position options list, so the list's viewport coordinates were applied inside the card; removed (cards are
now plain translucent cream), the list attaches to its field. **Palette** (`app/static/nsr.css`, variables kept
under their old names): page #FFF0C5, inputs and cards #FFF8E6; teal #359FA0 for what is absent (ghost outlines,
bars, ring meter, big numbers); light teal #8AD6D1 for evidence and for fills that carry text (rank badges, the
selected tab); orange #FF8C52 for the Analyse button, severity and benchmark-reveal badges (dark text on them).
None of the four reads as small text on cream (teal 3.2:1, orange 2.2:1), so small text uses derived shades: ink
#12393a, secondary #2f6364, teal text #1d7374, orange text #b24a12 (all ≥ 4.5:1 on cream). Title gradient dark
teal → teal → light teal → orange; aurora in light teal and orange at higher opacity; grain multiplied at 5%; the
cursor ring is teal and turns orange over clickable things (no difference blend on a light page); trail particles in
the palette with normal compositing. Gradio reloads once into `?__theme=light` (was dark); theme hues teal / orange /
stone. Verified in the pane on Owens (ghosts, reveal badges, Evidence ring 87%, comparison) and the About results
table. Tests unchanged (472).

## 2026-10-21 (evening, 2) — 7.1: snappier cursor, guided demo, the About pages (project explained for any reader)

**Parsa's changes:** (1) the cursor felt laggy; (2) make the demo more intuitive and better narrated; (3) the site is
demo + information: problem, method, results, constraints, understandable by anyone.

1. **Cursor:** the ring now closes 60% of the gap per frame (was 20%) and its hover transitions are 120 ms (were 200):
   a hint of weight, no visible lag. The dot was always immediate.
2. **Guided demo:** a five-step strip under the tabs (pick · analyse · read the ghosts · check the evidence · change the
   peers or compare); a colour legend; an empty state in the reading pane showing what a ghost paragraph looks like
   before anything is analysed; progress narration during analysis (`gr.Progress`: reading and embedding → detecting
   statements in N sections → finding the 200 closest peers → placing the ghosts), which needed `prepare()` to take
   the already-embedded upload; the status line is now a narration with the numbers ("Read 6 sections and found this
   document's 200 closest peers among 22,300 privacy policies of the same category (education), mean similarity 0.76,
   in 0.78 s. It states 20 of the catalogue's 58 kinds of statement; below are the 10 kinds …"). **Benchmark examples
   are a challenge:** with the comparison off, the narration says how many kinds of statement were removed and asks
   whether the ghosts caught them; turning the switch on reveals them, badges the ghosts of removed kinds ("removed by
   the benchmark · found") and shows the three-column comparison. The switch now redraws document, narration and panel.
3. **About the project** (top-level tab beside Demo; `app/about.py`): the problem (absence has no text to search for;
   the research question), how it works in five steps with "more" folds (what a ghost says and does not; the score),
   how well it works (three tests with tables and plain-language readings, the privacy negative finding and the leak
   explained, speed and detector quality), where it falls short (seven limits), what it is not (the §0.2 #21 stance),
   under the hood, glossary, credits. **Every number is copied from the recorded tables** by `app/content.py`
   (`python -m app.examples` now also writes `app/examples/results.json` from corpus_stats, bench_synth, bench_revision,
   cohort_diagnosis, llm_gap_candidate, analyse_gate and presence_gate); a missing table shows "not recorded", never a
   typed number. Tests check the copy, the sections, and that no verdict word appears (the one sentence that names
   "non-compliant" and "deficient" says the system never uses them).

Verified in the pane: guide strip and empty state; Owens Community College (challenge line, then reveal: ours rank 1,
Opus 3, Qwen none); Difficulty-Aware MT Evaluation (papers, 1.20 s; reveal badges on 3 ghosts, comparison ours rank 1
with 3 of 4, Opus rank 1 with 3 of 4, Qwen none); About page: 8 sections, 3 tables, 0 "not recorded". Tests: 472
(new: `tests/test_app_about.py`). **Open:** Parsa's approval, then the Space.

## 2026-10-21 (evening) — 7.1: the demo redesigned (dark, animated, lens cursor), pinned examples, chat-model comparison

**Parsa's brief:** a bold, modern, animated design with a custom cursor and mouse trail; deploy to a Space after approval.
**Design** (`app/static/nsr.css`, `nsr.js`, loaded by `launch(css=, head=)`; the page reloads itself once with
`?__theme=dark`): near-black void, aurora drift and film grain behind glass cards; Syne / DM Sans / JetBrains Mono
(Google Fonts, system fallbacks offline); three accents with fixed meanings: lime = absent (ghosts, bars, the ring
meter), cyan = evidence (quotes, similarity), magenta = action and severity. Motion: gradient sweep on the title,
ghosts slide in staggered by rank and breathe, prevalence bars fill, the Evidence ring animates (`@property --p`),
percentages count up; a lens cursor (ring + dot, grows over anything clickable) with an additive particle trail whose
density follows pointer speed. Pointer devices only; all motion off under `prefers-reduced-motion`. The ui-ux-pro-max
skill's palette (slate + blue) was too tame for the brief; its dark-OLED and research-typography guidance was kept.

**Examples** (`app/examples.py`, `python -m app.examples` → `app/examples/<genre>.json`, 59 KB, tracked): per genre
one corpus document (Etilize `privacy_pl:36309`, FANS `unarxive:2111.00400`) and two OmitBench-synthetic instances
with the lists the 7.3 runs recorded (`serve.bench_examples`, `serve.chat_runs`): privacy RSsportscars
`single_type:privacy_pl:539904` and Owens Community College `single_type:privacy_pl:67783`; papers MarIA
`single_type:unarxive:2107.07253` and Difficulty-Aware MT Evaluation `single_type:unarxive:2107.14402`. Chosen by hand
for readable titles among instances of 5-14 sections and 3-14k characters, not by outcome (Opus beats level 1 on
RSsportscars, rank 1 vs 3). The app reads the files when present, so a Space needs neither the corpus nor the runs.

**Chat-model comparison** (plan 7.1's toggle): a switch; for a benchmark example, three columns (this system, Claude
Opus 5.5, Qwen2.5-72B) with the deleted types marked where a system found them and each column's first hit rank;
for any other document, a note that cached answers exist only for benchmark examples and that a live call is off
(`serve.allow_live_chat: false`; not implemented in the lean tier). Ghost paragraphs are clickable: the click fills a
hidden textbox and presses a hidden button (`#nsr-pick-go`), which selects the absence and opens the Evidence tab.
**Gradio 6 findings:** a synthetic `input` event on a Textbox does not reach the backend, a synthetic button click
does; `gr.Tabs(selected=)` as a return value is ignored, `gr.update(selected=)` works; theme fonts must be
`gr.themes.Font` objects (strings crash `launch`, first run failed and its folder was removed; `runs/log.csv` keeps
the row); inactive Tab children are not rendered until the tab is shown. A hidden browser pane throttles the page's
timers and drops dropdown focus: verify with the pane visible.

**Verified in the pane** (Mac M5, MPS): hero, cursor and trail; RSsportscars (privacy, 10 sections, 0.55 s, 10 ghosts
after their sections, comparison: ours rank 3 / Opus 1 / Qwen none of 3); MarIA with pack auto (papers chosen, 2.15 s,
comparison: ours rank 1, 5 of 5 / Opus 2, 4 of 5 / Qwen 6, 1 of 5); Etilize (corpus, 0.91 s, the cached-only note);
ghost click → Evidence tab with the ring at 51% and five quotes; re-rank; 375 px wide: no horizontal scroll. Tests: 469
(new: `test_app_examples.py`, comparison panel, ghost click, category reset). **Open:** Parsa's approval of the
design, then the Space (`serve.examples_dir` makes the data self-contained; pack sizes and CPU timing to check).

## 2026-10-21 (afternoon, 3) — 7.3 closed: the open teacher (Qwen2.5-72B) is below Opus and both corpus systems

**Approved by Parsa.** Route `configs/openrouter.yaml` (DeepInfra fp8, 32k context, pinned; key mapped from
`OPEN_ROUTER_KEY`), `baselines.max_tokens=1024` (16,000 would not fit the context). Smoke
`llm_gap_candidate_qwen_smoke_s0`; run `llm_gap_candidate_qwen_s0` (4.7 min): 1.10 M input and 18 k output tokens,
≈ $0.40. **One papers prompt (≈ 34 k tokens) exceeds the route's context**: not truncated, scored as an empty
ranking and counted (`papers/unanswered` 1). The stage now falls back to answering prompts one at a time when an item
fails (answers already received come from the cache); test added. The table keeps the Opus rows (fix of the previous
entry).

| Recall@5 (same 100 instances) | Privacy | Papers |
|---|---|---|
| Qwen2.5-72B (open teacher) | 0.137 [0.090, 0.187] | 0.181 [0.126, 0.245] |
| Claude Opus 5.5 | 0.242 | 0.235 |
| Level 1 | 0.376 | 0.655 |
| Genre-uniform | 0.375 | 0.465 |
| Qwen − level 1 | −0.239 [−0.311, −0.160] | −0.473 [−0.569, −0.378] |
| Qwen − genre-uniform | −0.238 [−0.304, −0.166] | −0.283 [−0.387, −0.182] |

**Negative finding: Qwen does not rank.** 60 of 100 privacy and 69 of 99 papers lists are in catalogue order (Opus:
0 of 100 in both), and only 87 / 80 lists are distinct: it filters the catalogue rather than ordering absences by how
conspicuous they are, so recall@1 is near floor (0.042 / 0.022). Both chat models are below both corpus systems in
both genres; the open teacher is below Opus (not tested paired). Plan 7.3 ticked. Tests: 466 passed.
Hub: runs and packs pushed (no job running); status 986 matched, 0 differ, 0 missing.

## 2026-10-21 (afternoon, 2) — 7.3: Claude Opus 5.5 on the candidate list, far below both corpus systems

**Approved by Parsa** (one pass, 100 single_type instances per genre, effort medium). The key is exported from `.env`
for the command (`set -a; . ./.env`): run.py does not read `.env` (the first smoke attempt failed on that, logged).
Smoke `llm_gap_candidate_smoke_s0_20261021-115944-987469` (2 instances per genre, Messages API, ≈ $0.17). Run
`llm_gap_candidate_opus_s0` (Message Batches, 12 min, 196 requests sent + 4 reused from the smoke cache): 1.88 M
input and 44 k output tokens, ≈ $4.2 at batch list prices (not checked against the Console invoice). The dry run's
estimate ($8) assumed 3.5 characters per token (measured ≈ 2.9) and 2,500 output tokens per request (measured ≈ 220:
at effort medium the model thinks little). Every answer valid: 10 catalogue ids, none outside it.

`reports/tables/llm_gap_candidate_<genre>.csv`, same 100 instances, MRR cut at 10 for every system, paired bootstrap:

| | Privacy recall@5 | Papers recall@5 |
|---|---|---|
| Claude Opus 5.5 (candidate list) | 0.242 [0.185, 0.301] | 0.235 [0.172, 0.306] |
| Level 1 (cohort prevalence) | 0.376 | 0.655 |
| Genre-uniform prevalence | 0.375 | 0.465 |
| Opus − level 1 | −0.133 [−0.198, −0.064] | −0.420 [−0.518, −0.325] |
| Opus − genre-uniform | −0.132 [−0.199, −0.061] | −0.230 [−0.328, −0.129] |

Recall@1 0.051 / 0.070 (level 1 0.098 / 0.189); MRR@10 0.339 / 0.275 (level 1 0.451 / 0.545). The gap is outside the
CI against both systems in both genres. **Caveats for the report:** the instances were built and verified with the
system's own presence model (the deleted type is "absent" by that model's judgement; the model judges presence by
reading, and may count paraphrases left in the text as stated); one pass at effort medium (no run-to-run spread,
little thinking); the open teacher not yet run.

**Fix before the open teacher:** the stage overwrote `llm_gap_candidate_<genre>.csv`; it now keeps the other models'
rows and replaces only its own (test added). Tests: 465 passed.

## 2026-10-21 (afternoon) — Phase 7.1 Gradio app built and verified; 7.3 prompt and job proposed; gate 5.9 on mains power

**Checks (HANDOFF):** pytest 441 passed, as expected. Two differences from HANDOFF, both resolved before this session:
nothing was uncommitted (the round-trip and 6.4/6.5 work is in commits `10ab7a1` and `79fb84b`, authored by Parsa), and
`hub_sync status --bundles runs,packs` gave 948 matched, 0 differ, 0 missing (runs and packs had been pushed).

**7.1 engine split** (tests first, `tests/test_prepare_score.py`): `analyse()` is now `score(prepare(..))`.
`embed_upload` (read, segment, embed) → `prepare` (presence, document presence, document vector: the slow part, once
per upload) → `score` (cohort with k / category filter / date window overriding the config, expectation, rank,
evidence, explanation). Checked on the real packs: on one test document per genre, prepare + score equals analyse to
6 decimals; **analyse is not bitwise reproducible with itself on MPS** (P(present) differs by ≈ 1e-9 run to run: the
ColBERT prototype scores; section embeddings are identical), so the check rounds floats.

**Ghost placement** (`engine/explain.py` `ghost_position` / `place_ghosts`, `tests/test_ghosts.py`): from the type's
section_hint only, no genre vocabulary: the heading that shares the most hint words (first five letters, each word
weighted by log(1 + headings / headings containing it), so "privacy" in every privacy heading counts little), else
the hinted part (near the start → after the first section; first half → after its last section; second half, near
the end or no hint → after the last section). On the Etilize policy 9 of 10 ghosts are placed by position, most after
the last section: the hints mostly name a half of the document, not a heading.

**Auto-detect** (`engine/onthefly.py` `detect_pack`, `tests/test_onthefly.py`): the upload is embedded once per
embedder and reused by `prepare`; the pack whose index holds the nearest 10 documents (mean similarity) wins, the
margin is shown. `onthefly.pack_similarity_threshold` stays null (Phase 10), so a document of another genre is still
assigned to the nearer pack. FANS paper: papers 0.827, privacy 0.522.

**App** (`app/gradio_app.py`, task `serve`, `configs/serve.yaml`; `tests/test_gradio_app.py` builds the Blocks app on
the small pack and calls its handlers): input row (paste, upload .txt/.md/.html, pack or auto, three test-split
examples per genre via `document_markdown`), the document with dashed ghost paragraphs ("N% of peers like this one
state X; this document does not", count of n, Wilson interval, severity badge only when the weight is not 1, where
it was placed and one peer quote), Cohort tab (k, category filter, category, date window, re-rank without embedding
again, similarity, fallback flag, nearest peers), Evidence tab (selected absence, quotes with document ids, prototype
quotes marked as such, template explanation). An uploaded file is deleted once read and Gradio's cache is cleared
every two minutes; `serve.store_uploads: true` is refused. Teacher rationale only behind `serve.allow_teacher`
(default false), then off by default and flagged. No verdict words (tested). Not done: the chat-model toggle (waits for
7.3), the CPU Space.

**Verified in the browser pane** (run `serve_s0_20261021-114046-082824`, Mac M5 / MPS, loaded in 10.9 s): Etilize
policy (`privacy_pl:36309`): 7 sections, 24 types stated, 10 ghosts, 0.74 s; re-rank at k 50 changes the cohort without
re-embedding; FANS paper (`unarxive:2111.00400`) with auto: papers chosen, 8 sections, 2.20 s. **Bug found and fixed
there:** the category of a privacy example carried over to the paper (category filter fell back); now a category not
in the chosen pack resets to the document's own or none (test added). **Negative finding (for the report's error
analysis):** the policy's top ghost, "opt out of communications" (70% of peers), is a false alarm: the policy gives an
unsubscribe address; the document model gives P(present) 0.40. The first drawn privacy example is an adult site:
pin examples with `serve.examples` before recording the 7.2 video. `preview_stop` kills the server without SIGTERM, so
the first app run wrote neither metrics nor a log row (its folder, holding only config.yaml, was removed); SIGTERM
and Ctrl-C are handled (check run `serve_s0`, logged ok). Stop the app with Ctrl-C to record its timings.

**7.3 proposed, not run** (tests first, `tests/test_baselines.py`): prompt `prompts/llm_gap_candidate.md` v1
(`v1:3c5f75db208f`): the catalogue (id: definition) before the document (a cached prefix per genre), the model
returns at most 10 absent type ids, most conspicuous first, structured output restricted to the catalogue ids; the
prompt never mentions edits. Stage `baselines.stage: candidate`: seeded sample (`sample_seed` 2026) of
`llm_gap_instances` from `bench_synth_v1`, scored with `bench/metrics.py`, paired with level 1 and genre-uniform from
`bench_eval_v2` (MRR cut at 10 for every system). Overlay `configs/llm_gap_anthropic.yaml` (Claude Opus 5.5, Message
Batches, effort medium, responses cached). Dry run `llm_gap_candidate_dry_s0` (no API call): 100 single_type
instances per genre, 1.84 M + 3.75 M prompt characters ≈ 0.53 M + 1.07 M input tokens; ≈ $3.55 + $4.64 at an assumed
2,500 output tokens per request (thinking cannot be turned off on Opus 5.5).

**Gate 5.9 on mains power** (`build.force=[analyse_gate]`, runs `gate59_mains_<genre>_s0`): privacy max 1.37 s, median
0.35 s (battery: 2.32 / 0.75); papers max 3.28 s, median 3.11 s (battery: 8.15 / 7.1). `pack.json` and
`analyse_gate_<genre>.csv` updated. Runs bundle pushed (no job running); `packs/*/pack.json` differ from the hub.
Tests: 464 passed.

## 2026-10-21 — Phase 6.4 and 6.5 closed: severity ablation, figures, error analysis; synth checkpoints

**Checks (HANDOFF):** pytest 429 passed (423 + the round-trip session's 6); `git status`: the round-trip session's
changes still uncommitted (as HANDOFF said; not committed in this session); `hub_sync status --bundles runs,packs`:
890 matched, 2 differ (`packs/*/pack.json`, which hold that session's uncommitted battery-power gate timings), 0
missing.

**6.4 severity weights** (eval option `bench.eval.compare_to`: paired differences against a recorded run's
per-instance results; `bench.eval.table` names the table; tests first). Run `bench_ablation_severity_v2` →
`reports/tables/ablation_severity_<genre>.csv`. Papers has no weight ≠ 1 (47 types at 1.0): identical, as it must be.
Privacy (16 of 58 types at 1.5), weighted − w = 1: level 1 recall@5 single_type +0.005 [−0.024, 0.032], MRR +0.040
[0.014, 0.066], three_types recall@5 +0.031 [0.017, 0.045]. **Negative finding:** the weights raise false alarms@5
on the untouched labelled documents (genre-uniform 0.205 → 0.245, level 1 0.223 → 0.243), and level 1 still does not
beat genre-uniform on single_type (+0.017 [−0.001, 0.035]); the weighted ranking is not a gate result either way.
`bench_ablation_severity_v1` (logged) compared freshly computed metrics with the reference CSV's 3-decimal values
(±0.0005 noise); v2 rounds both sides alike and replaced its tables.

**6.4 figures** (`python -m analysis.figures`, `tests/test_figures.py`; reads only the recorded tables):
`reports/figures/bench_recall5.png` (recall@5 per regime and system), `level1_minus_uniform.png` (the gate quantity on
the synthetic regimes and both revision variants; blue where the CI is above zero), `cohort_diagnosis.png`
(within-document AUC per cohort variant; the deletion leak, edited vs original vector; post_hoc_best drawn hollow and
labelled test-selected). **Deferred (Parsa):** cascade vs classifier-only presence end to end: only the edited
sections would need embeddings (privacy 3,358 sections / 3.0 M characters, papers 11,195 / 14.8 M, Mac-sized), but
the instances were verified with the cascade, so a classifier-only ranking meets edits chosen to fool the cascade;
4.6 compared the two at section level.

**6.5 error analysis** (`analysis/error_analysis.py`, tests first; stage `bench.stage: errors`). The eval stage now
also writes `<run>/<genre>/untouched.csv` (the 150 labelled dev documents' top 5 and judge-present types); rerun as
`bench_eval_v2`: every recorded table and metric identical to `bench_eval_v1`. Run `bench_errors_v3` →
`reports/tables/errors_<genre>.csv` (v1: date bins labelled by year only, two papers bins read "2022-2022"; v2: added
corpus prevalence; v3: the correlation below over types with ≥ 15 occurrences). Findings (level 1, all regimes):
- **Missed types are the rare ones.** Per-type recall@5 against corpus prevalence, Spearman 0.90 (privacy, 57 types)
  and 0.56 (papers, 40 types). Privacy: 13 of 58 types never reach the top 5 (median prevalence 0.10 vs 0.51), e.g.
  `california_disclosure_rights`, `deletes_children_data_if_found`, `complies_with_laws_and_programs` (82
  occurrences). Papers: `discusses_societal_impact_or_misuse` 0.05, `acknowledges_contributors` 0.08,
  `generates_synthetic_data` 0.12 (prevalence 0.13-0.23). Papers types stated by every paper (`proposes_method`,
  `motivates_problem`) occur 1-7 times: long papers state them everywhere, so they are hardly ever deletable.
- **Recurring false alarms** (top 5 of untouched documents, in fact present): privacy `provides_third_party_links`
  10 of 26 shown, `security_not_guaranteed` 9 of 26, `browser_cookie_controls` 11 of 41; papers
  `assumption_and_distribution` 15 of 21, `refers_to_appendix` 16 of 32, `code_and_data_availability` 15 of 32.
  On synthetic documents the most frequent non-ground-truth type in privacy's top 5 is
  `reserves_right_to_change_policy` (34% of instances).
- **By length** (single_type, level 1 − genre-uniform): papers gains most in the third length quartile (+0.187
  [0.120, 0.261]) and least in the shortest (+0.071 [0.002, 0.133]); privacy's only bin with a CI above zero is the
  shortest quartile (+0.042 [0.002, 0.086]). **By date:** no trend in either genre. **By category:** papers cs.CL
  +0.101, cs.LG +0.112; privacy within ±0.09 in every site category, none with a CI above zero.

**Synth stage, for the next long run** (tests first): every `progress_every_s` a log line with a timestamp,
documents tried of total, instances per regime, docs/min and an ETA; every `checkpoint_every_s` an atomic
`<run>/<genre>/checkpoint.pkl` (deleted once the outputs are written); `bench.synth.resume_from` continues an
interrupted run into a new run folder (settings must match). Checked on the Mac: a smoke run interrupted with SIGINT
after 4 documents (the last complete checkpoint had 3) and resumed gave byte-identical `instances.jsonl` and
`edited.jsonl`; `arrays.npz` differs by at most 6e-8 in the cascade probabilities (ColBERT on MPS is not bitwise
reproducible across batches; document vectors identical). Tests: 441 passed.

## 2026-10-20 / 21 — Phase 6: OmitBench-synthetic and -revision built and run; gate 6.6 passes for papers, fails for privacy

**Checks (HANDOFF):** pytest 397 passed; git status clean; `hub_sync status --bundles runs,packs`: 712 matched, 0
differ, 0 missing on the hub (the Phase 5 push had been done). As expected.

**6.1 design (Parsa's choices, 2026-10-20).** Test documents had no stored section or sentence probabilities, so a
new stage `bench.stage: probs` scores every test section and sentence with the pack's presence model and writes them
to the run folder, never into a pack (`check_outside_packs`; `_presence_model` refuses the test split unless asked,
`wanted_documents`). Colab job `bench_probs_v1` (A100, 23 min): privacy 3,097 documents / 185,276 sentences, papers
1,746 / 500,330. A sentence carries t when its probability passes the cascade threshold (the threshold `analyse()`
quotes evidence at); the edited document is verified with the document-level presence model (`doc_presence`, the type
must drop below 0.5). Types are drawn uniformly from the document's present types; ground truth is every type that
goes from present to absent (targets plus collateral); whole_section deletes every section carrying the sampled type;
one document per entity per regime, in one seeded order (`bench.seed` 2026). The Mac measured the full presence model
at 95 sentences/s (all test sentences ≈ 2 h), hence Colab.

**Negative finding: a Markdown round trip does not reproduce the corpus segmentation.** `segment(document_markdown(..))`
gave back the stored sections for 214 of 300 privacy and 34 of 300 papers test documents (headingless policies come
back as one section; inner pseudo-headings split paper sections). Edits therefore work on the stored sections. Fixed
in a separate session (its own entry).

**Negative finding: in long papers a type cannot be deleted sentence by sentence.** Smoke (20 test documents per
genre): privacy deletions verify easily, papers do not: `states_own_limitations` is stated in 13 of 23 sections and 35
sentences of one paper, and with every section capped at P = 0.1 the document model still gives 0.71 (its per-type
intercept carries the type's prevalence). Escalation added (tests first): the next most likely carriers (5 per round,
P ≥ 0.02, up to 8 rounds), then whole sections (up to 5); at most 50% of the characters removed; up to 3 draws per
document. A first schedule (halving the threshold) swept up too many weak sentences and was replaced.

**6.1 run** (`bench_synth_v1`, Colab, 1 h 50 min, papers three_types nearly all of it): privacy 500 / 500 / 500
(single / three types / whole section), papers 500 / **468** / 500: three_types tried every test paper (4,067 draws
rejected for removing too much; Parsa: keep the rules and report the shortfall). Per-type acceptance in
`reports/tables/bench_synth_acceptance_<genre>.csv`. Verified on the Mac: arrays aligned with the instances, every
ground-truth type absent after its edit, one instance per entity per regime.

**6.2** `bench/metrics.py` (tests first): recall@k, MRR, nDCG@10, false alarms, ECE, bootstrap CIs over documents,
paired bootstrap, bootstrap over groups. **6.3** (`bench_eval_v1`, w = 1, `reports/tables/bench_synth_<genre>.csv`):
recall@5 single_type, genre-uniform vs level 1: privacy 0.385 vs 0.396, papers 0.514 vs 0.622 (random 0.159 / 0.278).
False alarms@5 on the 150 judge-labelled dev documents (out-of-fold document presence): privacy 0.205 vs 0.223, papers
0.305 vs 0.341. ECE of P(expected) vs the judge: privacy 0.048 vs 0.067, papers 0.079 vs 0.091 (genre prevalence is
better calibrated). **Benchmark artefact:** deleted targets end just under the presence threshold (median P(present)
after the edit 0.32 / 0.45), so (1 − P) halves their score for both systems; absence-only recall@5 is 0.03-0.06.

**6.3b OmitBench-revision** (`bench_revision_probs_v1` on the Mac, 1 h 46 min: 54,774 + 13,631 revision sections,
34,501 newly embedded; `bench_revision_v1`): privacy 1,678 instances of 4,029 pairs (1999-2019), papers 137 of 343
(2019-2022); genre-uniform uses the corpus prevalence of the same date window; bootstrap over entities. Recall@5 level
1 − uniform: privacy −0.010 [−0.023, 0.002], papers +0.107 [0.046, 0.166]. The plan's filter selects pairs with the
level-1 quantity (cohort prevalence ≥ 0.5), which favours level 1; a system-neutral variant (every added type):
privacy −0.009 [−0.017, −0.002], papers +0.041 [−0.003, 0.086]. 34% of privacy's time-windowed cohorts fell back
from the category filter.

**Gate 6.6** (paired recall@5 difference on single_type, CI above zero): **papers pass** (+0.108 [0.074, 0.141]),
**privacy fail** (+0.011 [−0.010, 0.033]). **Diagnosis** (stage `diagnose`, run `bench_diagnose_v1`,
`reports/tables/cohort_diagnosis_<genre>.csv`, decision record 007): privacy cohorts carry little information about
practices (Brier vs genre prevalence −3.4% for privacy, −20% for papers); the category filter and the title part of
the document vector weaken them (without both, within-document AUC 0.880 → 0.915); and the deletion leaks into the
cohort: retrieved from the original document's vector, the better cohort would give +0.069 [0.036, 0.102] on privacy
single_type, from the edited one +0.013 [−0.020, 0.048]. A variant found by searching settings on the test bench
reaches +0.081 and is reported as post hoc only. **Decision (Parsa, 2026-10-21):** the privacy failure is reported as a
negative finding and Phase 7 proceeds; §0.2 #9 and #10 unchanged; plan task 6.6 edited first, 6.1-6.4 annotated.
Exploratory versions of the diagnosis ran as scratch scripts first; every number above comes from the recorded run.

**Also:** `hub_sync` worked as designed through two Colab jobs. Process lesson: the synth stage writes its outputs only
at the end and logs no timestamps; a long Colab run cannot be interrupted without losing it (next session: incremental
parts, timestamps, documents tried, ETA). Tests: 423 passed (429 with the round-trip session's new test file).

## 2026-10-20 — A corpus document rebuilt as Markdown now segments back to its stored sections

**Finding (Phase 6 session).** `analyse()` segments the Markdown it is given, while the indexed peers' profiles were
built on the corpus's stored sections (`data/processed/<genre>/sections.jsonl`). A corpus document rebuilt by
`engine/pipeline.document_markdown` and re-segmented by `segment` (gate 5.9, the `analyse` task, the Phase 7 demo
examples) did not come back with its stored sections, so the query was segmented differently from its peers. Causes:
(1) headingless policies (stored as `fallback_section_chars` chunks) came back as one section, because the `# Title`
line counts as a heading and the fallback chunking never ran (`privacy_pl:1002201`: 6 sections → 1 of 7,854
characters); (2) 145 of 300 paper titles span two lines, so the second line of `# Title` became text of the Abstract
(`unarxive:1801.00632`, 1,336 → 1,366 characters); (3) lines inside paper sections that look like headings
(`Parameterparameters`, `Definition 2 (Sectional curvature [cite])`) started new sections (23 → 26 sections). Plain
Markdown cannot carry the stored segmentation in any case: on the same 300 papers, 238 sections contain blank lines,
54 are under `min_section_chars` (re-segmenting merges them), and `clean_inline` is not idempotent (double spaces,
edge pipes).

**Fix (test first, `tests/test_segment_roundtrip.py`, 6 tests: a headingless chunked policy built with the
loader's own segmentation, a headed policy with heading-like lines, pipes and `#`/`\` line starts, a paper with a
two-line title, inner pseudo-headings, a blank line and a short section, documents without a title, a hand-edited
export, and unmarked text).** `data/segment.py` gains `write_sections_markdown` / `read_sections_markdown`: Markdown
that starts with the comment `<!-- negspace: sections -->`, the title as `# `, each section opened by `## heading`
(or `<!-- section -->` when it has none), its text verbatim; a text line starting with `#`, `\` or `<!--` gets one
backslash, which the reader removes. Rendered, it is ordinary Markdown. `document_markdown` writes it; `segment()`
reads it (sections as written, only oversized ones split) and sends any text without the marker to the heuristic
segmenter, unchanged, so uploads and the corpus loaders behave exactly as before. `guess_title` skips HTML-comment
lines. No genre code.

**Match rate (section texts identical, in order; first 300 test documents per genre, documents.jsonl order):**

| genre | before | after |
|---|---|---|
| privacy | 224 / 300 (74.7%) | 300 / 300 |
| papers | 34 / 300 (11.3%) | 300 / 300 |
| privacy, whole test split | 2,358 / 3,097 (76.1%) | 3,097 / 3,097 |
| papers, whole test split | 186 / 1,746 (10.7%) | 1,746 / 1,746 |

Headings match too, except for 12 of 1,746 papers whose stored heading contains a line break (written on one line).
Before, headings matched for 6 / 300 privacy and 0 / 300 papers (the title became the first section's heading).
The finding quoted privacy 214 / 300; this count, with the old `document_markdown` reproduced exactly, is 224 / 300
(possibly a different "first 300"); the papers count agrees. `bench/synth.py`'s docstring is updated: OmitBench
still edits stored sections directly, which is now equivalent.

**Gate 5.9 rerun** (`build.force=[analyse_gate]`, runs `gate_roundtrip_privacy_s0`, `gate_roundtrip_papers_s0`):
**pass**, privacy max 2.32 s (17 sections), median 0.75 s; papers max 8.16 s (31 sections), median 7.10 s; limit
10 s. Segmentation drops from 7–16 ms to under 1 ms per paper. Papers are slower than on 2026-10-20 (3.31 s max),
but not because of this change: the Mac was on battery (42%), and an A/B on the same machine state embedding the old
and the new section lists of two gate papers took the same time (4.7–6.4 s each). `reports/tables/analyse_gate_<genre>.csv`
and the pack manifests now hold the battery-power timings; rerun on mains power for the report's numbers.

## 2026-10-20 — Phase 5 closed: gate 5.9 passed, teacher-mode check, a lost label file restored

**Checks (HANDOFF):** pytest 394 passed (HANDOFF said 393); the "uncommitted" Phase 5 work was already committed and
pushed as `c7c1781`, so `git status` was clean. `hub_sync pull --bundles runs,packs`: evidence 45 privacy / 29 papers
parts, as expected.

**Gate 5.9 passed** (pack builder, `build.adopt=true`): `sentence_evidence` adopted, every other stage skipped,
`analyse_gate` run on 5 held-out dev documents per genre on the M5 (MPS), template explanations, after one warm-up
call. Privacy (`pack_privacy_adopt_s0`): max **1.98 s** (17 sections), median 0.69 s. Papers
(`pack_papers_restored_s0`): max **3.31 s** (32 sections), median 3.07 s, embedding ≈ 85% of the time; the
query-time fixes of 2026-10-19 (batch 4, stored sentences) hold, no further speed-up needed. Task 5.1's Mac timing
from the same runs: 15.0 / 10.2 sections/s at query time. A report by eye (`privacy_pl:172230`, a 3.2k-character
policy): the top absences (automatic usage data 94% of peers, browser cookie controls 76%, third-party links 78%, ...)
are genuinely absent (no mention of cookies, IP addresses or browsers); quotes fit their types, two weak picks (an
affiliates quote about "Requested Service Providers", a personalisation quote mostly about marketing). Cosmetic:
`count` is a float in the Report (188.0). Side note: base.yaml's three seeds made the first privacy build log two
no-op runs (`pack_privacy_adopt_s1`, `_s2`); later builds pass `run.seeds=[0]`.

**Negative finding: `hub_sync pull` overwrote a finished file with a partial one.** The first papers build failed
(`pack_papers_adopt_s0`: `doc_presence` no longer skipped, then "need at least one array"): the papers judge labels
`runs/judge_docs_haiku_s0/teacher_labels_papers.jsonl` were **empty** on the Mac. Timeline from file times and hub
commit dates: the runs bundle was pushed at 00:31 while the papers batch was still running (empty file on the hub);
the run finished at 00:40; both manifests were recorded at 00:46 with the complete file; a runs pull at 00:49 replaced
it with the hub's empty copy (`snapshot_download` replaces any local file that differs from the hub's). Profiles v2
and decision record 006 were built before 00:49 and are valid; every pack artefact still matched its manifest (all
stages checked, both genres); nothing else was lost. **Fix (test first, `tests/test_hub_sync.py`):** pull keeps a
local file that differs from the hub's copy and was modified after the hub's last commit of it, and reports it
(`kept_local`, push it); an older local copy is still replaced, so new Colab artefacts arrive. Checked read-only
against the hub: today it would keep only the two updated manifests.

**Restore from the teacher cache** (no API spend): run `judge_docs_papers_from_cache_s0`, the same stage with an
unusable API key so that a cache miss fails instead of paying: 5,668 of 5,668 calls from the cache, 0 failures,
13,703 labels (original 13,698). The difference: sections whose text occurs more than once share one cache entry, so
every copy gets one answer, where the batch answered each copy. Measured on privacy, whose original file is intact
(`judge_docs_privacy_from_cache_s0`): all 15 differing rows are such duplicates; 3 of 1,393 sections have different
labels (5,419 vs 5,420). Refit on the restored papers labels (`doc_presence_papers_cache_check_s0`, outputs restored
afterwards): the same model is chosen (all five features, C 1.0); eval-half macro F1 0.789 → 0.788, ECE 0.019 →
0.020. The restored file replaced the empty one (note `restored_from_cache.json` in the run folder) and the papers
pack was rebuilt (`doc_presence` and the gate rerun, the rest skipped): **papers profiles v2 now reproduce from the
cache**; 846 of 663,358 bits flip (0.13%), largest prevalence shift 1.2 points (`names_software_and_libraries`),
29.70 → 29.68 types per paper. Decision record 006 has an addendum.

**5.8 teacher mode, one real call** (new task `analyse`, `experiments/analyse.py`, `configs/analyse.yaml`, test
first: analyses named corpus documents, one Report JSON each, and in teacher mode lists the words of the teacher's
text found in neither the quotes it was given nor the type card, `engine/explain.py` `unsupported_words`). Run
`explain_teacher_privacy_s0`: `privacy_pl:172230`, top 5, Qwen2.5-72B via OpenRouter (DeepInfra fp8, pinned),
prompt `explain` v1:3ea55796474c, 2,157 + 462 tokens ≈ **$0.001**, 8.7 s with the calls. By eye: no invented fact,
number, law or period; rationales restate the quotes in general terms; three of five suggested wordings copy peer
sentences verbatim (faithful, but a product would show them as peer text, not as the tool's wording); one rationale
("advise users to review the privacy policies of those sites") comes from the type's definition, which the prompt
includes, not from its quotes; one wording ends with a bracketed instruction instead of a placeholder. The check flags
3 function words (`including`, `options`, `certain`). The check is lenient (first five letters) and the definition
counts as a source, so it catches invented specifics, not paraphrase drift.

**Plan:** 5.1-5.9 ticked with their records; **§0.2 #14 edited** to record the evidence retrieval change of
2026-10-19 (each peer's best sentence under the full presence model, stored by `sentence_evidence`, instead of the
prototype index); the decision itself (up to five peer sentences with document ids, template explanation) is
unchanged. `engine/pipeline.py` `document_markdown` is shared by the gate and the new task. Tests: 397 passed.

## 2026-10-19 — Phase 5 session 1: checks, Colab path fixed

**Checks (HANDOFF):** pytest 329 passed; privacy schema v1, 58 types; papers schema v2, 47 types. **Hub vs Mac:**
`hub_sync status` (new, read-only; test first) over data, embeddings, cache, packs, runs: 511 of 511 bundle files
match, none differ or are missing; run log 91 rows on both sides with the same keys. The hub's 514 files are those
511 plus `.gitattributes`, the merged run log and a stray `runs/.DS_Store` (the `runs/*/*` pattern also matches
files directly in `runs/`); `select` now skips Finder/Windows junk files (test). The stray file stays on the hub
(harmless; hub_sync never deletes).

**Colab smoke test, first attempt (negative findings, all fixed test first, `tests/test_colab_runner.py`):**
- **The clone cell printed the GitHub token** in the error of a failed `git clone` (the token was part of the clone
  URL, and a failed command prints its arguments). Parsa revoked it and made a new one. Cell 2 now sends the token in
  an HTTP header for github.com only (set in the session's `~/.gitconfig`), never in a URL or `.git/config`, and shows
  git's own error with the token masked. Test: no cell interpolates a secret into a URL.
- The clone failed because GitHub rejected the token (401 on `/user`); a new fine-grained token fixed it.
- **Colab now runs Python 3.13**; `pip install -e .` refused (pyproject pins the Mac's 3.11), and
  `requirements-colab.txt` replaced Colab's numpy, pandas and pyarrow under the running session (numpy `_slice`
  ImportError; google-colab needs pandas 2.2.3). Fixed: Colab keeps its own scientific stack (only the model
  libraries are pinned), and the package is not installed (the code runs from the clone). Plan §0.6 (Colab row)
  edited to say so; no §0.2 decision changed. The Mac-vs-Colab smoke comparison checks this changes no result.
- **Runner from GitHub:** the notebook opens directly from GitHub (link in its first cell) and runs the job named
  under `next:` in the new `configs/colab_jobs.yaml` (pull bundles, run line, push bundles), so a new GPU job is a
  commit on the Mac, not a notebook edit. Test: every job's config exists, uses `--overlay colab`, names its run after
  the job, and names real bundles.
- **Colab smoke passed** (run `profiles_smoke_colab_s0`, A100, Python 3.13, Colab's own numpy/pandas/torch, opened
  from GitHub, Run all): presence profiles of the first 200 train documents per genre, compared document by document
  with the Mac's profiles (`packs/<genre>/profiles`, MPS): same documents and section counts; largest probability
  difference 1.0e-6 (privacy) and 2.4e-3 (papers, one value; mean 2.9e-7); **0 of 21,000 presence bits differ**.
  254 / 248 sections/s (Mac 134 / 65). The Colab environment changes no result.
- **Bug found by the smoke's `-dirty` commit flag:** the profiles stage wrote `reports/tables/profile_prevalence_*.csv`
  even when the profiles went elsewhere, so a limited run overwrote the corpus table with 200-document numbers
  (the Mac's copy is intact: committed in `b7eae69`). Fixed (test): such a run writes its table next to its profiles.

**Document-level presence (HANDOFF first job), started.** Method (proposed, Parsa's choices 2026-10-19): label
**whole documents** and fit a document-level calibrator on them; no genre code. Findings that shaped it:
- No corpus document has every section labelled: papers have one labelled section per paper, and OPP-115's policies
  are cut into 33 short segments each (3,792 segments of 115 policies, median 369 characters) against 8.6 sections
  (median 585 characters) for a corpus policy, so a document-level rule fitted on OPP-115 would not transfer.
  Privacy needs the same method as papers.
- A noisy-OR over section probabilities is no fix on its own: it is never below the best section, so it inflates
  presence further; any document threshold needs document-level labels to be fitted.
- **Labels from the judge, Claude Haiku 5.5, through the Anthropic API** (Parsa: use the Anthropic credit instead of
  OpenRouter). Consistent with the presence model, whose section probabilities were calibrated against the judge's
  labels (fit half of the silver reference). One labelling pass covers both the calibration and a document-level
  silver reference.

Built (tests first): `AnthropicTeacher` in `models/teacher.py` (official `anthropic` SDK 1.15.0, pinned; structured
outputs with closed schemas; no sampling parameters, which Claude Haiku 5.5 rejects; a prompt prefix shared by a batch
is cached; `mode: batch` sends Message Batches at half price, chunked, and an interrupted batch is resumed from the
cache directory instead of paid twice; refusals and cut-off answers are failed items, retried with a doubled budget);
`configs/judge_anthropic.yaml`; stage `select_documents` (whole dev documents stratified by year and category, only
from sources the indexed corpus has, never the induction sample, other label sets or test): run `select_documents_s0`,
**150 policies (1,393 sections) and 150 papers (2,834 sections)**, added as the `documents` set of
`packs/<genre>/label_sets.json` (existing sets unchanged); stage `section_probs` (every section's probabilities, train
+ dev, `packs/<genre>/section_probs/`, synced through the hub), sharing the presence model loader with `profiles`.
Mac smoke `section_probs_smoke_s0` (150 documents per genre): the best section per document equals the stored profile
(max difference 7e-8). Colab job `section_probs_v1` queued in `configs/colab_jobs.yaml`.

**`section_probs_v1` on Colab** (A100, commit `2097ebf`, 30 min): 564,675 sections (privacy 422/s, papers 269/s),
train + dev. Against the Mac's v1 profiles over the whole train split: 19 of about 2 million document-type values
differ by more than 0.001 (fp16 near-ties in ColBERT on CUDA), 7 of 1.3 million privacy bits flip; negligible.

**Route check, judge through the Anthropic API** (run `judge_route_check_s0`, 20 reference sections per genre, plain
calls, $0.05): the same model (Claude Haiku 5.5) agrees with its own earlier OpenRouter labels at micro F1 **0.936**
(privacy) and **0.850** (papers); teacher vs judge is 0.775 / 0.692. The gap is the judge's run-to-run variation
(sampling defaults differ between routes; Claude Haiku 5.5 rejects temperature), mostly on papers' close neighbours;
papers got 13% more labels through the direct route. **Documents run** `judge_docs_haiku_s0` (Message Batches):
privacy 1,393/1,393 sections, 5,420 labels, 536 from the second look, 0 failures.

**Document-level presence, privacy (run `doc_presence_privacy_try_s0`, eval half 75 policies):** the judge finds
**26.8 of 58 types per policy**; the v1 rule marks 22.8. **Negative finding, corrected diagnosis:** v1 is not
inflated overall; it is miscalibrated at document level in both directions: it overstates 8 types by more than 5
points (`limits_internal_access` 71% vs 21% labelled, `deletes_children_data_if_found` 45% vs 11%) and understates 33
(`sends_marketing_communications` 27% vs 71%, `shares_with_service_providers` 20% vs 59%). The document model
(shared slopes, per-type intercept; features chosen by cross-validated log loss: best, second-best, sum of section
probabilities, number of sections): macro F1 0.687 -> **0.754**, micro 0.739 -> 0.808, mean prevalence error 0.136
-> **0.063**, ECE 0.150 -> **0.018**. The three feature sets tried score within 0.001 of each other.

**Phase 5 tasks 5.2-5.6 built (tests first):**
- 5.2 `engine/index.py`: `Index` with FAISS and Qdrant backends, payload filters (genre, category, date window,
  splits, excluded entities and ids), shared save format; both backends return the same top-10 on exact search
  (1,000 vectors, four filter settings). **Negative finding:** on macOS, faiss-cpu 1.15.1 and torch 2.13.0 bundle
  different builds of libomp and a process that loads both aborts (either import order). Not worked around with
  KMP_DUPLICATE_LIB_OK; instead FAISS is loaded only for unfiltered approximate search (exact and filtered search,
  which cohorts use, is a NumPy pass), and in a macOS process with PyTorch loaded an approximate request runs exactly.
- Document vector (§0.2 #9) in `engine/embed.py` with the existing key `models.embedder.title_abstract_weight` (0.5;
  a duplicate key added by mistake was removed; the vectors are identical).
- 5.3 `engine/cohort.py` (never the query's own entity; category filter falls back when the pool is under k);
  builder stages `index`, `evidence` (best section per document and type) and `cohort_report` in `packs/builder.py`,
  run by `experiments/build_pack.py` (runs `index_v2_<genre>_s0`, `evidence_v1_<genre>_s0`). Indexes: privacy 22,300
  documents, papers 14,114 (1.5-1.9 s on the Mac). Cohorts of 500 dev documents (k 200): mean peer similarity 0.783 /
  0.745, nearest 0.848 / 0.858, own entity in a cohort 0 / 0, category fallback 5.6% / 0%. **Finding:** privacy cohorts
  hold several yearly snapshots of the same other site (88.4% distinct entities among the 200 peers), so such sites
  count more than once in prevalence; one snapshot per entity is a possible ablation (not changed).
- 5.4 `engine/expectation.py`: level-1 prevalence over cohort profiles (bits; mean probability as an option) with
  Wilson intervals. 5.5 `engine/rank.py` (score per §0.2 #1, ties by prevalence) and `engine/evidence.py`.
- 5.6 `engine/pipeline.py` `analyse()` -> `Report` (JSON), `packs/spec.py` `GenrePack.load`, `engine/explain.py`
  template. Smoke on a held-out privacy policy (12 sections): 4.4 s on the M5 (embedding 2.7 s), model loading 10.5 s.
- **Evidence (negative finding):** quoting the sentence of a peer's best section that best matches the type's
  prototypes (ColBERT MaxSim) gave wrong quotes (an asset-sale sentence for `states_retention_period`, an SSL sentence
  for `limits_internal_access`): MaxSim is a weak sentence selector (prototypes alone reached macro F1 0.42 in 4.6),
  and a type-specificity variant agreed with the full model on only about half of the picks. The full presence model
  on sentences picks the right ones, but at about 9 ms per sentence on the Mac it took evidence to 7.6 s (total
  10.6 s). Fix: stage `sentence_evidence` precomputes, for every document and type of the indexed corpus, the best
  sentence and its probability (5.3M sentences; Colab job `sentence_evidence_v1`); `analyse()` looks them up, keeps
  those at or above the type's threshold, and drops duplicate boilerplate. Mac smoke (30 documents per genre):
  correct quotes, e.g. retention periods, access limits, code availability URLs.
- The live dashboard shows any run log written in the last 24 hours (state and last lines, including Message Batch
  progress) and counts the documents set.

**Judge documents run finished** (`judge_docs_haiku_s0`): papers 2,834/2,834 sections, 13,698 labels (4.8 per
section, 14% from the second look), 0 failures; spans 99.8% verbatim. Whole run 8,454 batch calls, 29.7M input +
4.4M output tokens, about **$2.60** of the Anthropic credit.

**Document-level presence, papers** (eval half 75 papers): the judge finds 27.8 of 47 types per paper; macro F1 0.734
-> **0.789**, micro 0.789 -> 0.850, prevalence error 0.152 -> **0.094**, ECE 0.218 -> **0.019** (feature set chosen:
all five, with the noisy-OR). The headline "inflation" was mostly real presence: the judge finds
`states_own_limitations` in **82.7%** of papers (v1 profiles 96.6%) and `states_limitations_of_existing_approaches` in
97.3%. v1 overstated 16 types (`gives_illustrative_example` 95% vs 55%) and understated 20 (`task_description` 28% vs
96%, `code_and_data_availability` 16% vs 41%). Residual: v2 overstates a few near-universal types
(`claims_broader_applicability` 100% vs 67%) and understates some rare ones (`generates_synthetic_data` 15% vs 36%).
Decision record `reports/decisions/006-document-level-presence.md`; plan notes on 4.7 and a new task 5.0 (no §0.2
change: #4 already aggregates section presence to document level).

**Profiles v2** (pack builder, stage `doc_presence` forced; v1 kept in `packs/<genre>/profiles_v1`): privacy 24.2
types per policy (v1 20.2), papers 29.7 (v1 26.8). Corpus prevalence moves toward the labels:
`limits_internal_access` 69% -> 21%, `sends_marketing_communications` 19% -> 69%, `code_and_data_availability` 14% ->
39%. **Finding:** a bit at 0.5 saturates near-universal types (`states_own_limitations` bit 98.6%, mean probability
0.82); level 1 keeps the plan's bit mean, the mean probability is the Phase 6 ablation.

**5.7 pack builder** (`packs/builder.py` `build`, `pack_stages`; `experiments/build_pack.py`): stages embed, heads,
prototypes, cascade, section_probs, sentence_evidence, doc_presence, index, evidence, cohort_report, analyse_gate; a
stage is skipped when its input fingerprints (content; repository-relative paths), its settings and the model ids
match `packs/<genre>/pack.json`; `build.adopt` records existing artefacts; a stale GPU stage on a machine without
CUDA stops and names its Colab job. Bug found by running it: adopting took precedence over forcing (fixed, test).
Both packs' manifests recorded (every stage adopted, then `doc_presence` rebuilt).

**5.8** `explain.mode: teacher` (`prompts/explain.md` v1; only the peer quotes go to the teacher, never the document;
top-N absences with cohort quotes; same Report fields as the template). Not yet run against a real model.

**Query-time embedding** (`models.embedder.query_batch_size: 4`): with batch 32 a document's sections were padded to
the longest one; on the M5 (MPS, fp16) a 29-section paper embeds in 2.6 s at batch 4 vs 8.6 s at 32, vectors
unchanged (min cosine to the stored ones 0.99978). Papers smoke before the fix: 15.0 s for 26 sections (embedding
8.0 s, query-time evidence 6.7 s).

## 2026-10-18 (night) — Compute moves to Colab Pro; private GitHub and Hugging Face stores

**Parsa's resources:** Colab Pro (A100), $100 of Anthropic API credit, Hugging Face Pro. **Plan edited first:** §0.6
(Colab Pro A100 is the primary GPU from Phase 5 on, through one runner notebook; Hugging Face Pro for the private data
store and a ZeroGPU Space as the demo host option) and 7.3 (frontier model: **Claude Opus 5.5** via the Anthropic API,
Message Batches at 50% off, ≈ $10–20, official `anthropic` SDK; Parsa's choice). No §0.2 decision changed.

- **GitHub:** private repo https://github.com/parsa-hke/negative-space-reader created with Parsa's OK; `main` pushed
  (commits authored by Parsa, no Claude co-author). Commit `d624f13` holds Phase 4 up to the ablations.
- **Hugging Face:** private dataset repo `parsa-hke/negative-space-reader-data` (Parsa's OK), the store shared by the
  Mac and Colab. `experiments/hub_sync.py` (`push` / `pull --bundles ...`, tests `tests/test_hub_sync.py`): bundles in
  `configs/base.yaml` (`hub.bundles`: data 2.4 GB, embeddings 1.7 GB, the teacher cache as one archive, packs, runs);
  push uploads only files whose content differs (git blob sha1 / LFS sha256), in bounded commits, never deletes; the
  run log is merged (`logs/runs_log.csv` on the hub), not overwritten. Credentials come from the logged-in HF client;
  none is stored in the repo or printed.
- **Colab:** `notebooks/colab_runner.ipynb` (secrets HF_TOKEN, GH_TOKEN read-only, optional API keys; clone, install
  `requirements-colab.txt` keeping Colab's CUDA torch, pull bundles, run with `--overlay colab`, push outputs);
  `configs/colab.yaml` (CUDA, fp16, larger batches). Tests and smoke stay on the Mac.
- `.gitignore`: pack artefacts built by the presence stages (`packs/*/profiles/`, `packs/*/cascade*.json`) are synced
  through the hub, never committed.
- **First upload failed (negative):** after 16 files (2.06 GB) an S3 part upload hit `RequestTimeout` (the socket
  idled during a slow upload, ~1.5 MB/s) and the whole push stopped. Fixed (test first): commits of at most 512 MB,
  each retried up to 4 times with back-off; the cache archive is staged as a file so the hub's chunked upload is used
  (an in-memory buffer fell back to plain HTTP). The rerun skips everything already on the hub.

## 2026-10-18 — Phase 4.3–4.6: linear heads, prototypes, cascade, ablations

All presence numbers below are on the **evaluation half of the silver reference set** (judge Claude Haiku 5.5;
decision record 004), with calibration, thresholds and the combination layer fitted on the **fit half** (the 300
reference sections split by document, `engine/classifier.reference_split`, seed 1234: privacy 15/16 policies, papers
150/150 papers). "Ceiling" = the teacher's own labels scored the same way (a student trained on teacher labels can
hardly beat it). Papers use pack version 2 (47 types).

**4.3 linear heads** (`engine/classifier.py`, `distil.stage: heads`, runs `heads_linear_plan_s0`,
`heads_linear_teacher_s0`; tables `reports/tables/presence_heads_<genre>_<labels>.csv`). One logistic head per type
on the stored Qwen3-Embedding-0.6B section vectors (C 1, balanced), temperature per type and an F1-tuned threshold on
the fit half. Trains in seconds. Logistic regression is deterministic, so the three-seed rule gives identical runs
(one seed recorded).

| Heads | Privacy macro F1 | Papers macro F1 | ECE before → after temperature |
|---|---|---|---|
| plan labels (privacy: OPP-115 for 35 rule types, teacher for 23) | 0.576 | 0.486 | 0.124 → 0.089 / 0.189 → 0.146 |
| teacher labels for every type (privacy) | **0.613** | – | 0.120 → 0.077 |
| ceiling (teacher's own labels) | 0.773 | 0.688 | |

Privacy against OPP-115's human labels (26 rule types with ≥ 5 positives in the eval half): plan-trained 0.568,
teacher-trained 0.534. Each training source wins on its own kind of reference.

**4.4 prototypes** (`packs/prototypes.py`, `engine/late_interaction.py`, run `prototypes_v1_s0`; `reports/tables/
prototypes_<genre>.csv`): late-interaction model answerdotai/answerai-colbert-small-v1 (Apache-2.0, commit `934fa8bb`,
colbert-ai format loaded with transformers; downloaded with Parsa's OK, 134.6 MB; query augmentation left out on
purpose). Candidates = the teacher's verbatim evidence spans on training sections; k-means, centroid-nearest:
privacy 1,031 prototypes (58 types), papers 1,139 (47); 10–30 per type except `governing_law_and_disputes` and
`ethics_and_declarations` (9 candidates each).

**4.5 cascade** (`engine/presence.py`, `distil.stage: cascade`, run `cascade_v1_s0`; `packs/<genre>/cascade.json`;
figure `reports/figures/presence_band_<genre>.png`). Reranker BAAI/bge-reranker-v2-m3 (Apache-2.0, commit `953dc6f6`,
2.29 GB, Parsa's OK): 13 pairs/s on MPS for section-length text. Band from the training sections' prototype scores
(5% quantile of positives, 95% of negatives): **privacy [0.934, 0.951), papers [0.921, 0.946)**. ColBERT MaxSim
scores sit high and positives and negatives overlap heavily: 38% (privacy) and 67% (papers) of eval section-type
pairs fall in the band.

**4.6 ablations** (`reports/tables/presence_ablation_<genre>.csv`; latency on the M5, MPS; the classifier's latency
is the section embedding, which dominates for long paper sections):

| Variant | Privacy macro F1 | ECE | ms/section | Papers macro F1 | ECE | ms/section |
|---|---|---|---|---|---|---|
| classifier only | 0.576 | 0.089 | 141 | 0.486 | 0.146 | 911 |
| prototypes only | 0.417 | – | 5 | 0.328 | – | 16 |
| prototypes + reranker | 0.442 | – | 1,187 | 0.309 | – | 3,455 |
| classifier + prototypes | **0.612** | **0.004** | 146 | **0.505** | **0.016** | 926 |
| full cascade | 0.614 | 0.005 | 1,328 | 0.501 | 0.015 | 4,366 |
| teacher labels (ceiling) | 0.773 | | | 0.688 | | |

- **Negative: the reranker does not pay.** Full cascade vs classifier + prototypes: +0.002 (privacy), −0.004
  (papers), at 9× / 4.7× the time. Prototypes alone are weak; their value is as a second signal next to the
  classifier, where they lift macro F1 by 0.036 (privacy) and 0.019 (papers) and fix calibration (ECE 0.089 → 0.004).
- **Gate 4.8 (macro F1 ≥ 0.80 against the reference) is not met by any variant**, and cannot be against this
  reference: the teacher's own labels reach 0.773 / 0.688. Heads trained on teacher labels recover 79% (privacy) and
  73% (papers) of that ceiling. Decision for Parsa.

**Decisions (Parsa, 2026-10-18; decision record `reports/decisions/005-presence-model-and-gate.md`; plan edited
first: §0.2 #7, §0.2 #8, 4.8):**
1. **Gate 4.8 is relative:** the presence model's macro F1 against the reference ≥ 0.80 × the teacher's own macro F1
   against it (eval half, both genres). With hand labels the absolute 0.80 applies again.
2. **Presence heads train on teacher labels for every type** (`presence.train_labels: teacher`); OPP-115 stays a
   reported human check.
3. **The lean presence model is classifier + prototypes** (`presence.cascade.reranker: false`); the reranker stays
   implemented and ablated.
4. **Papers below the gate: recorded as not met, continue** (after the diagnosis below); SetFit is the recorded
   upgrade.

**Final 4.3 / 4.5 runs** with those settings: `heads_linear_v2_s0` (C chosen on the fit half from 0.1–10: **3.0** in
both genres; fit-half macro F1 privacy 0.666–0.710, papers 0.526–0.552 across the grid), `cascade_v2_s0`
(`packs/<genre>/cascade.json` = classifier + prototypes). Band thresholds recorded in `configs/pack_<genre>.yaml`.

| Eval half vs reference | Privacy macro F1 | ECE | ms/section | Papers macro F1 | ECE | ms/section |
|---|---|---|---|---|---|---|
| classifier only (teacher labels, C 3) | 0.617 | 0.061 | 107 | 0.508 | 0.124 | 485 |
| prototypes only | 0.417 | – | 5 | 0.328 | – | 13 |
| **classifier + prototypes (the model)** | **0.619** | **0.003** | 113 | **0.497** | **0.013** | 498 |
| teacher labels (ceiling) | 0.773 | | | 0.688 | | |
| **share of the ceiling (gate ≥ 0.80)** | **0.801: pass** | | | **0.722: not met** | | |

- **Diagnosis (papers):** the heads reach only 0.687 macro F1 on their own training sections (privacy 0.729) and
  0.477 in 5-fold cross-validation against the teacher (privacy 0.605): a frozen whole-section embedding cannot
  separate 47 fine-grained statement types in long sections (median 1,074 characters, 90th percentile 3,474; about
  5 types per section). The plan's remedy is SetFit (fine-tuning the embedder on the labels), kept as the upgrade.
- For papers, prototypes improve calibration (ECE 0.124 → 0.013) but not F1 (0.508 → 0.497); the pipeline uses the
  combined model for its calibrated probabilities.

**4.7 profiles** (`packs/profiles.py`, `distil.stage: profiles`, run `profiles_v1_s0`, smoke `profiles_smoke_s0` first
into a separate folder; `packs/<genre>/profiles/part-*.parquet`, 500 documents per part, resumable;
`reports/tables/profile_prevalence_<genre>.csv`): classifier + prototypes over every section of every train-split
document on the M5 (MPS). Privacy 22,300 documents, 192,596 sections, 134 sections/s (24 min); papers 14,114
documents, 305,088 sections, 65 sections/s (78 min). A document's `prob_<type>` is its best section's probability,
`bit_<type>` whether that reaches the type's threshold.
- **Negative finding (important for Phase 5): document-level presence is inflated.** Privacy 20.2 of 58 types per
  document, papers 26.8 of 47; `uses_cookies` in 78% of policies, `states_own_limitations` in 96.6% and
  `states_limitations_of_existing_approaches` in 99.2% of papers, while `code_and_data_availability` is at 13.5%. A
  document counts a type when any of its ~10–20 sections passes a threshold tuned for section-level F1 (which leans to
  recall), so small per-section false-positive rates compound. A type that looks present everywhere can never be
  ranked as a conspicuous omission, so document-level calibration (for example thresholds fitted on documents whose
  every section is labelled, such as the OPP-115 policies, or a noisy-OR with calibrated section probabilities) comes
  before ranking in Phase 5.

**4.8 gate** (`distil.stage: gate`, run `gate_v1_s0`, `reports/tables/presence_gate.csv`): privacy 0.619 / 0.773 =
**0.801 of the ceiling: pass**; papers 0.497 / 0.688 = **0.722: not met** (Parsa's decision: recorded, continue;
SetFit on Colab is the upgrade); profiles cover the whole indexable corpus (22,300 / 22,300 and 14,114 / 14,114).
**Phase 4 closed** with papers below its gate.

## 2026-10-17 — Silver presence reference (decision record 004); paper re-embed finished on Colab

**Decision (Parsa): no hand-labelled sanity set for now.** Labelling 300 + 300 sections against 50–58 types (about
32,000 judgements) does not fit the schedule. The 4.2 reference set is now **silver**: the same 300 fixed sections per
genre labelled by a second judge of another family, **Claude Haiku 5.5** via OpenRouter (pinned to Anthropic,
structured outputs), with the teacher's own prompts; for privacy, OPP-115's human labels on the 35 mapped types are
reported alongside. Hand labels on the page stay as the upgrade (`distil.score.reference`). **Plan edited first:**
§0.2 #15 (false alarms judged by the reference set), §0.2 #25 (reference set is a swappable component), tasks 4.2,
4.3, 4.5, 4.6, 4.8 ("human set" → "reference set"; the gate states which reference), Appendix B. Decision record
`reports/decisions/004-silver-presence-reference.md`. Options weighed: hand labels (20+ h), adjudicating only
teacher–judge disagreements (1–2 h; the cheapest later upgrade), verify mode (anchoring), OPP-115 alone (loose,
35 of 58 privacy types, no papers).
- Judge overlay `configs/judge.yaml` (`sets: [human]`). Test call `judge_haiku_try_s0` (2 paper sections): works;
  agrees with the teacher on core types, differs on the watch-list neighbours (the three comparison types,
  `proposes_method`/`method_proposal`). **Cost estimate corrected:** Claude counts ≈ 13k input tokens per section
  (≈ 3.4k for Qwen: tokenizer, schema in the structured-output request, retries), so 600 sections ≈ $1.50, not the
  $0.40 first quoted.
- **Judge run** `judge_haiku_s0`: 600/600 sections, 0 first-pass failures, but **the second look failed on 4 privacy
  and 29 paper sections**: 252 answers were cut off mid-JSON (Claude quotes long spans; the 400-token budget, doubled
  once to 800 on retry, ran out) or empty. Rerun with `distil.teacher_labels.check_max_tokens=1500` (judge only; the
  teacher's second look had no such failures) as `judge_haiku_s0_20261017-004935-303611`: **0 failures**; privacy
  765 labels (2.55 per section, 8% from the second look), papers 1,435 (4.78, 15%); spans 99.8–100% verbatim. Cost
  of both judge runs ≈ $1.05 (6.39M input + 0.82M output tokens at $0.10 / $0.50 per 1M).
- **Judge vs OPP-115** on the 300 privacy reference sections, 35 mapped types (run `score_judge_vs_opp115_s0`, table
  `reports/tables/teacher_vs_opp115_judge_privacy.csv`): micro P/R/F1 0.657/0.577/**0.614**, macro F1 0.630 over 29
  types with ≥ 5 positives (first pass alone 0.676/0.541/0.601, macro 0.610). The same pattern as the teacher's
  calibration (0.551 on other sections): the OPP-115 mapping, not only the models, limits agreement.
- `distil.score` now leaves out sections the judge failed on (a failed section has no labels and must not count as
  all-negative; test added).
- **Watch (from decision 004):** the plan fits calibration (4.3) and the combination layer (4.5) on the same reference
  set it evaluates on (4.6, 4.8); decide a document-level fit/evaluation split before 4.3.

**Paper re-embed: done on Colab** (see the 2026-10-15 entry for details): A100, 140 s; import checked (cosine
≥ 0.99999 against the Mac's encoder on 50 texts); store aligned with the rebuilt corpus (380,736 rows).

**72B teacher labelling** resumed (`label_72b_s0_20261017-001458-852549`, concurrency 24, 40–136 calls/min): privacy
done (teacher set 1,999/2,000 sections, 5,379 labels, 20% from the second look; reference set 300/300, 842 labels,
19%), then the papers phase **crashed on one HTTP 429** (OpenRouter: the shared DeepInfra pool for the 72B was
overloaded, "engine_overloaded"). **Fix:** `OpenAITeacher` now waits out rate limits and transient server errors (up
to 8 retries, 5 s doubling to a 120 s cap, jittered) instead of failing the run (test added). Relaunched; privacy
replays from the cache.

**Privacy scores** (teacher labels from the run above; tables `reports/tables/teacher_vs_<name>_privacy.csv`):

| Comparison (sections) | micro P | micro R | micro F1 | macro F1 (types ≥ 5 pos.) | first pass F1 |
|---|---|---|---|---|---|
| teacher vs judge, all 58 types (300 reference) | 0.740 | 0.814 | **0.775** | 0.777 (46) | 0.736 |
| teacher vs OPP-115, 35 types (300 reference) | 0.606 | 0.586 | 0.596 | 0.605 (29) | 0.569 |
| judge vs OPP-115, 35 types (300 reference) | 0.657 | 0.577 | 0.614 | 0.630 (29) | 0.601 |
| teacher vs OPP-115, 35 types (all 2,299 labelled) | 0.578 | 0.575 | 0.576 | 0.613 (35) | 0.539 |

- The second look raises teacher–judge agreement (micro F1 0.736 → 0.775: recall 0.70 → 0.81, precision 0.78 →
  0.74) and agreement with OPP-115 (0.569 → 0.596).
- The two models agree with each other (0.78) much more than either agrees with the OPP-115 mapping (≈ 0.60), which
  supports the 2026-10-15 reading that the mapping, not the models, is the main source of disagreement there.
- **Plan rule (4.2) triggered:** all 12 privacy types with ≥ 20 reference positives are below teacher F1 0.85 against
  the judge (0.66–0.84; lowest `shares_with_service_providers` 0.66, `data_use_for_services` 0.67,
  `sends_marketing_communications` 0.73, `shares_with_affiliates_and_partners` 0.74). Against a silver reference this
  is model–model agreement, so a low value means the two models read the definition differently, not which one is
  wrong. Revisiting definitions is a schema change (version-2 pack): for Parsa, with the papers results.

**72B labelling finished (2026-10-18)** after a stop at Parsa's request and a resume: run
`label_72b_s0_20261018-195259-515194` (all 4,600 sections; privacy replayed from the cache). Papers: teacher set
1,989/2,000 sections (11 failed first passes, 6 failed second looks), 10,412 labels (5.2 per section, 22% from the
second look), spans 98.4% verbatim; reference set 300/300, 1,492 labels (19% second look). Whole run: 9,804 calls
(6,777 from the cache), 4.15M + 0.78M tokens in the last segment. The provider rate-limited the run 37 times in the last segment (HTTP 429); the new back-off waited them out (worst case retry 7 of 8, waits up to ~100 s) and no call failed.

**Papers scores, teacher vs judge** (300 reference sections, all 50 types; `reports/tables/teacher_vs_judge_papers.csv`,
run `score_judge_papers_s0`): micro P/R/F1 0.658/0.684/**0.671**, macro F1 0.666 over 47 types (first pass alone
0.631, macro 0.630). Lower than privacy (0.775). **All 30 types with ≥ 20 reference positives are below 0.85**;
lowest `method_proposal` 0.11, `performance_comparison_across_other_works` 0.33, `states_own_limitations` 0.42,
`defines_notation` 0.50, `performance_comparison` 0.58, `method_comparison` 0.61, `task_description` 0.61.
- **The disagreement is between neighbouring types** (where the judge said A and the teacher did not, the teacher's
  other labels in that section): papers `performance_comparison_across_other_works` → `performance_comparison` (10),
  `describes_method_details` → `defines_notation` (9), `performance_comparison` → `reports_experimental_results` (7),
  `task_description` → `motivates_problem` / `proposes_method` (6 each), `states_limitations_of_existing_approaches`
  → `states_own_limitations` (6); privacy `shares_with_affiliates_and_partners` → `shares_with_service_providers` (9).
  These are the 3.4 watch-list pairs.
- **Merging the clear duplicates helps little** (simulated on the labels, no schema change): `method_proposal` into
  `proposes_method` and the three comparison types into one: micro F1 0.671 → 0.692, macro 0.666 → 0.693 (merged
  comparison type 0.77). The rest needs sharper definitions, not merges.
- **Consequence for gate 4.8:** a classifier trained on teacher labels can hardly agree with the judge more than the
  teacher does, so against the silver reference papers macro F1 is capped near 0.67 with schema v1; the 0.80 gate is
  out of reach for papers unless the types become more separable or the gate's reference changes. For Parsa.

**Papers pack version 2 (2026-10-18).** Parsa asked for "the best and fastest approach with no cost" that "keeps the
novelty"; chosen by Claude: **merge only the two evidence-backed near-duplicates, no relabelling.**
`method_proposal` → `proposes_method`; `performance_comparison`, `performance_comparison_across_other_works`,
`method_comparison` → **`compares_with_other_methods`** (new card: definition, two positive examples taken from the
merged types, a near-miss taken from a neighbouring type's claims). 50 → **47 types** (inside the 40–60 bound);
v1 kept byte-for-byte as `packs/papers/history/schema.v1.yaml` (sha1 `34253189e850`); `schema.yaml` records
`derived_from`, `changes` and `merged_from` per merged type; seed mapping rewritten (NeurIPS 1 → `proposes_method`).
Written by `analysis/make_papers_schema_v2.py` with `packs/schema_versions.py` (tests: `test_schema_versions.py`).
Labels stay as produced under v1 and are **projected** when read (`distil.stage: score` does it from the schema's
`merged_from`; training will too). Privacy stays version 1.
- Kept apart on purpose: own vs existing-approach limitations (a missing limitations statement is a core omission for
  this project), notation vs method details, task vs motivation: they need sharper definitions (a relabelling run),
  not merges.
- **Rescored under v2** (run `score_judge_papers_v2_s0`, table `teacher_vs_judge_v2_papers.csv`): teacher vs judge
  micro P/R/F1 0.673/0.711/**0.692**, macro F1 **0.693** (44 types with ≥ 5 positives; first pass alone 0.656/0.650);
  `proposes_method` 0.71, `compares_with_other_methods` 0.77. The plan's 0.85 rule is still triggered for most common
  types; the gate (4.8) is to be decided with the classifier's numbers, measured next to this teacher–judge ceiling.
- The labelling page (upgrade path) still shows the v1 types; its answers would be projected the same way.
  `analysis/label_page.py` now lists the v2 families for a rebuild.

## 2026-10-15 (evening) — Phase 4 session 1: oversized paper sections fixed; 4.1 privacy labels; label sets; 4.2 page

**Checks at start:** 248 tests pass; both schemas `version: 1`, `status: frozen`, 58 and 50 types, as in HANDOFF.md.
(This session ran in a git worktree; the gitignored data, venv, caches and run folders are symlinks to the main
checkout, so everything rebuilt here lives in the main checkout's `data/`.)

**Phase 2 bug fixed: long paper sections are now split** (test first: `tests/test_data_segment.py`,
`tests/test_data_loaders.py`). The split step of `normalise_sections` is now its own function,
`data/segment.split_oversized` (no genre code; the policy path is unchanged: privacy has 0 sections over 8,000), used
by `data/loaders/unarxive.py` and `data/loaders/arxiv_versions.py` with `segment.max_section_chars`. A section over
the limit is cut at paragraph breaks into chunks of about 4,000 characters keeping its heading; a paragraph over the
limit at sentence boundaries; a "sentence" over the limit (tables, listings) at spaces. Rebuilt with `configs/data.yaml`
(smoke first): run `data_papers_split_s0_20261015-223426-488981` (the first rebuild, `data_papers_split_s0`, was
replaced because of the next finding). Old files kept as `data/processed/_backup_papers_unsplit_2026-10-15/`.

| Papers corpus | Before | After |
|---|---|---|
| documents | 17,644 | 17,644 (identical `documents.jsonl`) |
| sealed test hash | `12bc037d…` | `12bc037d…` (unchanged) |
| sections | 373,478 | 380,736 |
| sections > 8,000 chars (> 24,000) | 2,462 (127), in 1,788 papers; longest 245,576 | 0; longest 7,999 |
| sentences | 4,981,128 | 4,981,476 |
| revision documents: sections, > 8,000 | 13,015, 243 (143 documents) | 13,631, 0 |
| revision pairs | 343 | 343 |

Checked section by section: the 2,462 long sections became 9,720 chunks, the other 371,016 sections are byte-identical,
and no document's text changed (whitespace aside).

- **Found (bug, fixed):** the e-print converter (pylatexenc) rendered `\maketitle` as "[NO \title GIVEN] [NO \author
  GIVEN] <today's date>", so the build date leaked into 8 of the 686 revision documents and rebuilds were not
  reproducible. `\maketitle` and `\today` now render as nothing (test added). One revision pair's change ratio moved
  0.164 → 0.156; the same 343 pairs.
- **Found:** the e-print revision documents had the same long-section gap (243 sections, 143 documents); fixed by the
  same function.
- **Claims re-linked** (stage `induce.stage: relink`, `tests/test_relink_claims.py`; run `relink_claims_s0` →
  `runs/relink_claims_s0/claims_papers.jsonl`). Claim ids are unchanged; `section_idx` now points into the rebuilt
  sections, with `section_idx_unsplit` and `relink` (how it was matched) added. Of 44,579 claims (304 papers, 27 of
  them with a split section): 44,076 in sections that were not split (4,010 claims changed index in total, because
  sections after a split shift); of the 503 in split sections, 480 matched their verbatim span in one chunk, 4 a span
  running over a chunk boundary, 19 by shared words. All 19 are claims whose span was already marked not verbatim at
  extraction. The Phase 3 record (`induce.cluster.claims`) still points to the original file; Phase 4+ uses the
  re-linked one.
- The eye-test selection (`prompts/eval/induce_extract_eye_test.json`): 3 of the 20 paper references pointed to
  shifted indices (texts unchanged, matched by their stored sha1) and were updated: 2204.13650 57→65, 2206.02468
  19→21, 2206.07681 16→23.
- **Re-embedding only the changed sections:** `embed.reuse: true` (`engine.embed.embed_and_store(..., reuse=True)`,
  test added) keeps every row whose text sha1 is in the store and encodes the rest. Store backed up as
  `data/embeddings/Qwen__Qwen3-Embedding-0.6B/papers/_sections_unsplit_2026-10-15/`. Run `embed_papers_split_s0` (9,728 new chunks, MPS fp32) **stopped at Parsa's request at batch 147 of 304 (≈ 70 min); the store only saves at the end, so that work is lost** and the store on disk is the old, intact one. Rerun needed (≈ 1.5–2 h on MPS, longest texts first; faster with nothing else on the GPU).
  **Done on Colab instead (2026-10-17, Parsa's choice):** new `embed.mode: export` → bundle of the texts the store
  lacks (run `embed_export_papers_s0`: 9,608 unique texts, 30.6M chars, 9.4 MB zip; manifest pins the model snapshot
  `97b0c614…` and settings) → `notebooks/colab_embed.ipynb` with the self-contained `experiments/colab_embed.py`
  (encodes exactly like `engine/embed.Embedder`; a test checks this on a small model) → `embed.mode: import`
  (run `embed_import_papers_s0`). Parsa ran it on an A100 (fp16, torch 2.11, same sentence-transformers 6.1.0 and
  transformers 5.17.0): 140 s, 68.7 texts/s. Import: 9,716 rows imported, 371,020 reused; 50 imported texts
  re-encoded on the Mac: cosine min 0.99999, mean 1.0. Store checked: 380,736 rows aligned with the rebuilt sections
  by id and sha1, all finite, unit norm. Tests: export → Colab script → import round trip, refusal when a text has
  no vector, Colab settings = Embedder.

**Task 4.1, privacy labels from OPP-115** (`packs/presence_labels.py`, `distil.stage: labels`,
`tests/test_presence_labels.py`; run `labels_opp115_s0` → `runs/labels_opp115_s0/labels_privacy.jsonl`, table
`reports/tables/presence_labels_privacy.csv`). Each `label_source` entry is parsed (`Category` or `Category :: A = v &
B = w`, all conditions on one annotation) and every name is checked against the annotation file (an unknown name is
refused instead of giving a silent zero): all 35 mapped types passed. Every section of the 115 OPP-115 policies is
labelled (3,792 sections; one has no annotation and is negative for every type). Every mapped type has 32–648
positives (none under 20). The 23 `teacher` types: `uses_web_beacons`, `tracking_data_not_linked_to_identity`,
`defines_personal_information`, `transfers_data_internationally`, `disclaims_control_of_third_party_cookies`,
`does_not_store_payment_cards`, `security_not_guaranteed`, `advises_users_on_security`,
`warns_disabling_cookies_limits_site`, `states_minimum_age`, `deletes_children_data_if_found`,
`continued_use_means_acceptance`, `states_last_updated_date`, `scope_excludes_third_party_sites`,
`third_party_sites_and_their_policies`, `consent_by_use`, `complies_with_laws_and_programs`,
`governing_law_and_disputes`, `policy_coverage`, `disclose_data_rights`, `content_and_usage_restrictions`,
`company_identification_and_service_definition`, `provides_third_party_links`. (The run went through the config's
default three seeds, `labels_opp115_s0..s2`, identical; deterministic stages now pass `run.seeds=[0]`.)

- **Found (schema, for Parsa):** `data_collection_methods` and `data_collection_contests_surveys` have the identical
  `label_source` (`First Party Collection/Use :: Collection Mode = Explicit & Does/Does Not = Does`), so their OPP-115
  labels are identical (493 positives each): the classifier cannot tell them apart. A fix changes the schema (a
  version-2 pack). **Parsa's decision: keep v1 for now and decide after 4.2/4.3** (per-type F1 and the hand labels
  show whether to merge them or teacher-label one in a version-2 pack).
- **Found:** `states_policy_scope_and_commitment` takes OPP-115's broad `Other :: Introductory/Generic` (648 positives,
  17% of sections), probably looser than its definition; 4.2 cannot measure it (OPP-mapped types are not hand-labelled).
- **Found (plan wording):** §0.2 #7 says OPP-115 gives "~23k labelled segments". The consolidated data has 3,792
  segments (115 policies) carrying 19,720 annotations at threshold 0.75; ~23k is OPP-115's count of practice
  annotations. **Corrected with Parsa's OK** (wording only, no decision changed) in §0.2 #7 and in decision record
  001's tier table.

**Label sets (4.1, 4.2)**: `distil.stage: select` → `packs/<genre>/label_sets.json` (tracked; ids + text sha1; refuses
to overwrite; refuses the test split). Run `label_sets_s0_20261015-225337-783546` (the first, `label_sets_s0`, gave
privacy only 285 human sections and was redone with a per-document rule).
- Papers: human set 300 dev sections, one per paper, excluding the 500 papers sampled for schema induction; teacher set
  2,000 train sections, one per paper; both stratified by relative position (5 bins) × length (< 600, 600–2,000,
  ≥ 2,000 chars), in proportion to the pool.
- Privacy: human set 300 sections, 10 from each of 31 whole OPP-115 policies drawn at random; teacher set 2,000
  sections of the other 84 policies. No policy is in both, so heads trained on the teacher set are evaluated on
  unseen policies.

**Task 4.1, teacher labels: method and prompts** (`packs/presence_labels.teacher_labels`, `distil.stage:
teacher_labels`; prompts `label_presence` v1 `36470c3f3686`, `label_presence_check` v1 `e3dd44279652`). With the 3.5
lesson (about 1 statement in 30 missed with 50+ definitions in one prompt) built in:
1. First pass, whole schema in the prompt: the section's statements, each mapped to a type or "none", with a
   verbatim span (the 3.5 coverage prompt plus spans).
2. Second look, one call per section: the types most similar (pack embedder, type cards = definition + examples) to
   each first-pass statement ("none" ones first, 2 each) and to the section itself (4), not found by the first pass,
   at most 8; the teacher says which the section states, with spans.
Every label records `found_by` (first | second_look) and whether its span is verbatim; the teacher labels every type
(privacy: the 35 OPP-mapped types too, a free check of the teacher against human annotations). `agreement()`
computes per-type P/R/F1 against the human set for the final labels and for the first pass alone, which measures
the second look's gain and cost in 4.2.
- **Local 7B check** (run `label_7b_iter1_s0`, 6 teacher-set sections per genre, embedder on CPU): privacy ran
  (6/6, 12 labels, 6 from the second look). Read against OPP-115: the format works end to end; of the 6 second-look
  additions about 4 are right (`data_collection_methods`, `warns_disabling_cookies_limits_site`,
  `third_party_sites_and_their_policies`, `states_minimum_age` defensible) and 2 wrong (`states_last_updated_date` for
  "terms are subject to change", `does_not_sell_or_share` in a children's section); the 7B's first pass also filed
  "subject to change" under `states_policy_scope_and_commitment`. **Negative:** the papers half failed with a Metal
  out-of-memory error (longer prompts while the MPS re-embedding job held memory); not retried. Prompts stay at v1 for
  the 72B check: the 7B is too weak to judge the content (3.5).
- **72B trial** (Parsa's go-ahead: 5-section trial, then the full run unless the trial shows a problem). Run
  `label_72b_try_s0`: 5 sections of each set per genre (20), 42 calls, 67,890 + 12,709 tokens ≈ $0.03; 0 failures;
  every span verbatim; 3.0–3.6 labels per privacy section, 5.4–5.8 per paper section. **Second look adds 22–40% of the
  labels** (far above the ~1 in 30 first-pass misses of 3.5). Read all 25 additions: privacy about 5 of 10 right (5 of
  7 right against OPP-115's own labels; all 3 on teacher types wrong: `deletes_children_data_if_found` for "does not
  knowingly collect", `tracking_data_not_linked_to_identity` for repeat-visit cookies, `provides_third_party_links`
  with no link); papers about 10 of 15 right. Most wrong additions reuse a passage the first pass already gave a
  better-fitting type. **Finding: the second look recovers real misses at roughly 60% precision.** Measured cost per
  section ≈ $0.0015 (3,395 input + 635 output tokens), so the full run is ≈ $6.8.
- **Calibration (Parsa's choice, ≈ $0.15):** 100 seeded random privacy teacher-set sections, both check versions,
  scored against OPP-115 on the 35 mapped types (new `distil.stage: score`, reusable for 4.2; option
  `distil.teacher_labels.sample`). Runs `label_calib_v1_s0`, `label_calib_v2_s0` (first pass from the cache),
  `score_calib_v1_s0`, `score_calib_v2_s0`; tables `reports/tables/teacher_vs_opp115_calib_v{1,2}_privacy.csv`.

  | vs OPP-115 (100 sections) | micro P | micro R | micro F1 | macro F1 (17 types ≥ 5 pos.) | labels from 2nd look |
  |---|---|---|---|---|---|
  | first pass only | 0.591 | 0.473 | 0.525 | 0.470 | – |
  | + second look v1 | 0.468 | 0.581 | 0.518 | 0.514 | 118 of 331 |
  | + second look v2 | 0.565 | 0.538 | **0.551** | **0.525** | 43 of 256 |

  `label_presence_check` v2 (`b5617b165eb1`; v1 kept in `prompts/history/`) shows the types already found and counts a
  candidate only when the section states it explicitly. Reading 22 random v2 additions: about 16–17 right (~75%).
  **v2 adopted for the full run.** Run `label_72b_s0` (concurrency 8, ~43 calls/min) was restarted at concurrency 24
  (~55–75 calls/min, no errors) as `label_72b_s0_20261015-233752-628701`, then **stopped at Parsa's request** after
  ≈ 1,000 calls; every answer is in the response cache, so the rerun replays them free. Remaining ≈ 7,500 calls
  (≈ 2 h, ≈ $6).
- **Finding (important, negative): the 72B agrees poorly with OPP-115-derived labels overall (micro F1 ≈ 0.53–0.55)**,
  but the disagreement sits in loose mappings: crisp ones agree (security measures 8/10, business transfer 3/3,
  does-not-sell 4/5, cookies, encryption), loose ones do not (`states_policy_scope_and_commitment` 3/18 from OPP's
  "Introductory/Generic"; `data_use_for_services` from "Basic service/feature"; `data_collection_contests_surveys`
  1/10, the identical-mapping pair). Reading the disagreements, the teacher was mostly right and the OPP→schema
  mapping too narrow or too broad (log files with IP addresses are automatic usage collection; "companies that
  perform marketing services on our behalf" are service providers; a mailing address is not scope and commitment).
  So OPP-115 labels mapped through `label_source` are noisier against the schema's definitions than the plan assumed
  ("derive from OPP-115 where the mapping is direct").

**Task 4.2 page**: private artifact https://claude.ai/artifact/Sy6Z8atpJ45WzJWHNkcKJi (source
`reports/human_labels/page/`, data from `analysis/label_page.py`, current data hash `946582249e6e`, page version 2).
One section per screen; **all types in both genres** (papers 50, privacy 58). Version 1 asked privacy for the 23
teacher types only (plan 4.2: "derive from OPP-115 where the mapping is direct"); after the calibration finding above,
**Parsa chose to hand-label all 58 privacy types**, so the human set measures the OPP-115-derived labels as well as
the teacher, type by type (no answers had been saved under version 1). Blind to the teacher. A "closest by wording" shortlist (8 types by
embedding) is shown as a reading aid. Guards from the 3.4 lessons: no save without "I went through every type";
nothing marked requires "states none of these types"; "unsure" kept apart. Answers in collections
`labels_papers` / `labels_privacy` (one document per section). Database checked with one probe write, read, delete.

**Tests:** 271 pass (new: `test_relink_claims.py`, `test_presence_labels.py`, cases in `test_data_segment.py`,
`test_data_loaders.py`, `test_embed.py`; `test_run.py`'s stub example moved from `distil` to `train_mam`). Nothing
committed.

## 2026-10-15 — Phase 3 closed: human pass, coverage check, both schemas frozen (GATE 3.6 passed)

**Task 3.4 (human pass), Parsa.** Done on a guided review page (private artifact
https://claude.ai/artifact/BXtRWoATairWDT6uvZiweU; source `reports/schema_review/page/`, data from
`analysis/schema_review_page.py`; one group per screen with plain-language readings, Claude's recommendation and
Parsa's decision, stored in the page's database). Raw answers: `reports/schema_review/decisions/page_answers_2026-10-15.json`
(154 groups, 125 item records, 44 checklist rows). Of 154 groups: 127 recommendations accepted, 17 kept as their own
item, 10 split. Read back by Claude, four problems were reported and Parsa resolved them in chat
(`decisions/resolutions_2026-10-15.yaml`):
- 9 splits had been saved with placeholder halves (`…_a`/`…_b`, identical definitions): **kept as one item each**.
  (The page let a split be saved without naming the halves, and let new items be marked checked with empty fields:
  page design faults of mine.)
- privacy had 68 items (> 60): **merged** the three contact items, cookie non-identification into
  `tracking_data_not_linked_to_identity`, log files into `collects_usage_data_automatically`, access restriction and
  confidentiality into `limits_internal_access`.
- items kept under the group's own name **inherit the severity weight and OPP-115 label source** of the proposed
  item they replace (5 privacy items back at 1.5).
- papers c051 (its item removed on the page, so orphaned): **folded into** `describes_training_procedure`.
Applied by `analysis/apply_review_decisions.py` → `reports/schema_review/final_<genre>.yaml` → workbooks
(`review_final_v3_s0`) → `packs/<genre>/schema_candidate.yaml` (`candidate_v3_s0`): 0 gate problems.

**Task 3.5 (coverage check).** Held-out sections = dev documents outside the extraction sample; teacher = the 72B.

| Run | Prompt | Privacy sections with an unmapped statement | Papers |
|---|---|---|---|
| `coverage_7b_iter1_s0` (10+10, 7B) | v1 | 40% | 50% (the 7B misses existing types; not used further) |
| `coverage_72b_iter1_s0` (10+10) | v1 | 10% | 0% |
| `coverage_72b_s0` (100+100, seed 0) | v1 | **18%** (fail) | 3% |
| `coverage_72b_v2_s1` (fresh, after widening 11 definitions) | v1 | **17%** (fail) | 1% |
| `coverage_72b_v3_s2` (fresh, + 5 widenings) | v2 (`background_or_filler` label) | **14%** (fail); 25% counting filler | 4%; **31%** counting filler |
| **`coverage_72b_v4_s3`** (fresh, recorded) | **v3 + second look** | first pass 9%, **after second look 2%** | first pass 2%, **0%** |

**Found**
- **Negative:** each round of widening fixed the misses of its own sample and a fresh sample showed a new tail:
  definition-only fixes do not converge on privacy's long tail. Evaluating on a fresh sample after each change
  (seeds 1, 2, 3) kept the reported number honest.
- **Negative (withdrawn the same day):** a `background_or_filler` label for textbook explanations and promotional
  filler (prompt v2, plan wording briefly changed and then rewritten) absorbed real introduction and related-work
  statements in papers (31% of sections). Background is a genre move in papers; the label was not genre-neutral.
- Most remaining privacy "none" answers were **classifier misses, not schema gaps**: about 10 of 14 sections in run
  v3 held statements that a definition already covered, some verbatim ("does not create any contractual rights",
  "fees for access requests"). With 58 definitions in one prompt, the 72B misses about one statement in 30.
- **Method adopted (plan 3.5/3.6 wording edited first):** every "none" gets a second look: the teacher sees only the
  6 types most similar to the statement (embedding of type cards: definition + examples) and decides again
  (prompt `induce_coverage_verify` v1 `b8acf18cc1e5`; first pass `induce_coverage` v3 `ac266c25ad29`). In the recorded
  run, 14 of 19 privacy and 2 of 2 papers second looks found a type. Reading all 16: about 3 are stretches ("does not
  use spyware" → `does_not_sell_or_share`); counting them as failures, privacy is at **3%**, still under the bar.
  Remaining privacy gaps: a "you must have permission to share other people's data" section and a
  terms-and-conditions block (refund, assignment), both one-offs.
- 15 definitions widened in total (privacy 14, papers 1), each approved by Parsa; listed in the resolutions file.
- The coverage stage now keeps a copy of the checked candidate and a content fingerprint (types, seed mapping,
  pruned groups); `freeze` refuses a candidate whose content changed since its coverage run (a rebuild that only
  changes provenance is accepted). Cost of all coverage runs ≈ $0.70.

**Task 3.6 GATE passed.** `packs/privacy/schema.yaml` and `packs/papers/schema.yaml`, **version 1**, frozen
2026-10-15, signed off by Parsa Hosseini, coverage run `coverage_72b_v4_s3`, run `freeze_v1_s0`. Changes from now on
are new pack versions. (Minor: the reason stored for papers' two pruned groups reads "reviewed on the page; see
decisions/…"; the actual reason, item removed by the reviewer, is in those files.)

*Privacy, 58 types* (1.5 = GDPR Art. 13/14 or CCPA): `states_information_required_to_register` (1.5),
`collects_usage_data_automatically` (1.5), `uses_cookies`, `uses_web_beacons`, `tracking_data_not_linked_to_identity`,
`defines_personal_information`, `uses_data_to_improve_services`, `uses_data_to_personalize`,
`sends_marketing_communications`, `does_not_sell_or_share` (1.5), `shares_with_service_providers` (1.5),
`shares_with_affiliates_and_partners` (1.5), `shares_aggregated_data`, `discloses_when_legally_required`,
`transfers_data_in_business_transfer`, `transfers_data_internationally` (1.5), `third_party_advertising_and_cookies`,
`disclaims_control_of_third_party_cookies`, `states_security_measures`, `encrypts_data_in_transit`,
`stores_data_securely`, `limits_internal_access`, `does_not_store_payment_cards`, `security_not_guaranteed`,
`advises_users_on_security`, `opt_out_of_communications` (1.5), `opt_out_of_tracking_or_ads`,
`browser_cookie_controls`, `warns_disabling_cookies_limits_site`, `information_provision_is_optional`,
`deletion_on_request` (1.5), `states_retention_period` (1.5), `no_knowing_collection_from_children`,
`states_minimum_age`, `deletes_children_data_if_found`, `reserves_right_to_change_policy`,
`states_how_changes_are_notified`, `continued_use_means_acceptance`, `states_last_updated_date`,
`states_policy_scope_and_commitment`, `scope_excludes_third_party_sites`, `third_party_sites_and_their_policies`,
`consent_by_use`, `complies_with_laws_and_programs`, `governing_law_and_disputes`, `provides_privacy_contact` (1.5),
`do_not_track_response`, `california_disclosure_rights`, `data_collection_methods` (1.5), `data_use_for_services`
(1.5), `policy_coverage`, `user_data_control` (1.5), `discloses_public_areas`, `disclose_data_rights` (1.5),
`content_and_usage_restrictions`, `data_collection_contests_surveys` (1.5),
`company_identification_and_service_definition` (1.5), `provides_third_party_links`.

*Papers, 50 types:* `motivates_problem`, `reviews_prior_work`, `proposes_method`, `states_method_advantages`,
`claims_broader_applicability`, `defines_notation`, `states_theoretical_result`, `proves_or_derives`,
`describes_model_architecture`, `describes_training_objective`, `describes_training_procedure`,
`describes_algorithm_steps`, `describes_method_details`, `fine_tunes_pretrained_model`, `describes_data_augmentation`,
`gives_illustrative_example`, `describes_datasets`, `reports_dataset_statistics`, `describes_data_split`,
`describes_preprocessing`, `generates_synthetic_data`, `describes_annotation_process`, `describes_evaluation_metrics`,
`specifies_hyperparameters`, `uses_baselines`, `specifies_compute_resources`, `names_software_and_libraries`,
`reports_computational_cost`, `reports_variance_or_significance`, `reports_experimental_results`,
`reports_ablation_or_improvement`, `visualizes_results`, `reports_observations`, `refers_to_appendix`,
`states_own_limitations`, `states_limitations_of_existing_approaches`, `suggests_future_work`,
`code_and_data_availability`, `discloses_funding`, `acknowledges_contributors`, `ethics_and_declarations`,
`discusses_societal_impact_or_misuse`, `assumption_and_distribution`, `mdp_component_definition`, `task_description`,
`method_proposal`, `performance_comparison_across_other_works`, `performance_comparison`, `method_comparison`,
`introduces_regularization`. Excluded checklist items: NeurIPS 16 and ACL E1 (LLM / AI-assistant declarations
post-date the 2018–2022 corpus). Pruned groups: c040 (difference from prior work), c064 (new dataset or benchmark).

**Watch in Phase 4 (risks, not changes):** some kept-apart items are close neighbours and may be hard for the
presence detector to separate: papers `performance_comparison`, `performance_comparison_across_other_works`,
`method_comparison` and `reports_experimental_results`; `proposes_method` and `method_proposal`;
`describes_training_procedure` and `introduces_regularization` / `describes_training_objective`; privacy
`policy_coverage` and `states_policy_scope_and_commitment`, `provides_third_party_links` and
`third_party_sites_and_their_policies`, `data_collection_methods` and `data_collection_contests_surveys`. Per-type F1
in 4.3 will show whether any pair needs merging in a version 2 of a pack.

**Tests:** 248 pass (new: `tests/test_schema_review.py`, `tests/test_induce_schema_gate.py`,
`tests/test_induce_cluster.py`). Nothing committed.

## 2026-10-13 — Phase 3 session: tasks 3.1 and 3.2 closed, decisions for 3.3–3.5

**Checks at start:** 232 tests pass; claims 21,046 (privacy) and 44,579 (papers), as in HANDOFF.md.

**Decisions (Parsa, 2026-10-13)**
- **3.2:** the 303 complete papers are accepted as the genre-B extraction sample (one teacher, the 32B, for every
  claim). The partial paper `unarxive:2110.04947` (35 of 47 sections, 212 claims) is **dropped** at load time; the
  claims file is not edited. Genre-B claims used from here on: 44,367 from 303 documents.
- **Teacher for 3.3 naming and the recorded 3.5 run:** Qwen2.5-72B-Instruct via OpenRouter (`--overlay openrouter`);
  prompts are iterated on the local 7B first.
- **3.5 method:** as planned: the teacher classifies 100 random held-out sections per genre against the schema.

**Task 3.1 ticked** after a full by-eye review of `runs/eye_v2_32b_s0/eye_test.html` (all 40 sections, 213 claims)
against plan Appendix D. Every criterion passes except "related work in a few claims", which is weak. Found:
- 5 of 213 claims misstate their section (no invented content): one Related Work section (A2) credits the cited
  works' findings to "the authors" (4 claims) and calls a cited paper's setup "their approach"; one privacy claim
  inverts who grants a content licence; one paper claim misdefines a metric; one privacy claim says "communications
  from third parties" for "on behalf of third parties".
- **Residuals carried into 3.3/3.4 as watch items:** (a) prior-work results phrased like the paper's own results
  ("The authors report that X outperforms Y"), so a "report results" cluster may mix both; (b) off-genre boilerplate
  extracted faithfully (an FDA disclaimer and a copyright notice in a policy footer), expected to form small
  prunable clusters; (c) proofs split into one claim per step (12 claims in one section).
- The papers `first_person_claim_rate` of 0.007 is a single false positive ("we necessarily have" inside a quoted
  mathematical statement); the true rate is 0.
- Example-label reuse in the full extraction: `policy_change` 48 and `external_context` 23 claims (privacy),
  `external_context` 130 and `evaluation_evidence` 132 (papers), 0.5% of claims; a sample of 25 is almost all related
  content (real policy changes, relating to prior work, results), a few unrelated (an appendix pointer). Clustering
  uses claim text, so this matters only if those labels surface as cluster names.

**Task 3.2 ticked** with the decision above.

**Task 3.3 implemented** (`experiments/induce.py` stages `cluster` and `name`, `packs/schema_induction.py` step 2,
`tests/test_induce_cluster.py`, fake embedder and fake teacher, ~2 s):
- `cluster`: claims (minus excluded documents) → Qwen3-Embedding-0.6B under an instruction prompt → minus each
  document's mean claim vector → UMAP-10 → HDBSCAN swept over min cluster size × min samples; the eligible setting
  (60–120 clusters, no cluster above 10% of claims) with the least noise is kept, and every setting is recorded.
  Claim vectors are stored per instruction version under `data/embeddings/<model>/<genre>/claims__…/` and reused.
  Summaries order members by centrality and record document count, extraction labels, headings, position.
- `name`: `prompts/induce_name.md` (no few-shot examples) gives each cluster a name, definition, 3 representative
  claims and a mixed-cluster flag; the draft flags names that copy an extraction-prompt example label (read from
  the prompt file itself) and duplicate names (merge candidates). A `limit`ed run never writes the pack.
- New prompts: `induce_cluster_embed` v1 `2dd14db565c5` (embedding instruction), `induce_name` v1 `ec100a0631e8` →
  v2 `4f63706fe4d1` (after 10 privacy clusters on the 7B: names must tell neighbours apart; definitions start with
  a verb; `prompts/history/README.md`).
- New pinned dependencies: `umap-learn` 0.5.12 (+ `pynndescent`, `numba`, `llvmlite`), `openpyxl` 3.1.5.
- Smoke (`--overlay smoke`, bge-small, 50 documents) passed for both stages before the full runs.

**Found (clustering; decision record `reports/decisions/003-claim-representation-for-clustering.md`)**
- **Negative: PCA-32 + HDBSCAN** (`cluster_pca32_s0`, 24 settings per genre) left 68–80% (privacy) and 92–95%
  (papers) of claims as noise; papers never reached 60 clusters.
- UMAP-10 (`cluster_umap10_s0`) cut noise to 17% / 39%; privacy clusters read as genre types.
- **Negative: paper clusters were research topics** (crop phenology from remote sensing: 763 claims from 4 papers;
  federated learning; EEG), although the claims are phrased as moves. The plan's Phase 3 risk, at the clustering
  step rather than the extraction step.
- Probe on the stored vectors (UMAP-10, min cluster size 120, papers): claim-weighted document spread 0.19 (plain)
  → 0.30 (document centring) → 0.42 (instruction) → **0.58 (both)**; clusters drawn from < 5% of documents 23 → 1.
  Centring changed nothing measurable on privacy (noise 17.7% vs 17.8%). Both are generic (no genre code), so both
  genres use them.
- Recorded run `cluster_v1_s0`: privacy chose min cluster size 80, min samples 10: **74 clusters, 15.5% noise**.
- Embedding speed on MPS varies 3× with memory pressure (2–14 s per batch of 128); running the 7B under MLX at the
  same time slows both.

- Recorded run `cluster_v1_s0`, papers: min cluster size 100, min samples 5: **80 clusters, 44.4% noise**, sizes
  100–2,669 (median 177), 2 clusters drawn from < 5% of papers (c031: 5 papers, remote sensing; c068: 13,
  summarisation). Papers noise stays high: method sections are topic-specific even after centring.
- **Naming (task 3.3 ticked)**: the 72B via OpenRouter (provider pinned, DeepInfra fp8; first live use of this route,
  one test call first), `name_72b_privacy_s0` (74 calls) and `name_72b_papers_s0` (80 calls), 0 failures,
  ≈ 133k input + 11k output tokens (≈ $0.05). Note: Parsa's `.env` names the key `OPEN_ROUTER_KEY`; commands map it
  to `OPENROUTER_API_KEY` (the config's name) without editing the file.
- Draft quality: privacy clusters are genre types throughout (3 flagged mixed, one duplicate-name pair: two contact
  clusters). Papers: ~50 reviewer-facing moves (hyperparameters in 229 of 303 papers, datasets, splits, metrics, code
  availability, funding, hardware, seeds, ablations, limitations, future work) and ~16 topic-flavoured method
  clusters. c000 ("limitations", 2,669 claims, cohesion 0.39) mixes own and prior-work limitations; c057 (named
  "disclaimer of views") is really ethics statements, consent, competing interests and annotator pay.
- **Example-label leakage**: privacy names flagged `policy_change_right`, `notifies_of_policy_changes`,
  `policy_change_notification`, a false alarm of the substring rule ("Policy Change" is an OPP-115 category); no
  papers name copies an example label. Members carrying an example label: at most 13% of a cluster (privacy c015, the
  policy-change cluster) and 9% (papers c011, relate to prior work), i.e. on semantically matching clusters.
- **Statements too rare for a cluster** (found by keyword search of all claims, most of them HDBSCAN noise): Do Not
  Track (35 claims, 29 policies), California disclosure rights (89 claims, 45 policies), significance tests and
  standard deviations (147 claims, 60 papers), societal impact or misuse (17 claims, 13 papers). Proposed as types
  sourced from noise claims; LLM-use declarations (4 papers) proposed as excluded seeds (they post-date the corpus).
- **Proposals for the human pass** (`reports/schema_review/proposals_<genre>.yaml`, Claude's, not decisions):
  privacy 74 clusters → 54 types (1 pruned: terms-of-use rules; 15 types at severity 1.5 with the GDPR/CCPA
  article noted), papers 80 clusters → 46 types (16 topical method clusters folded into one generic method type).
  Every cluster assigned exactly once; every OPP-115 category and NeurIPS/ACL item mapped or excluded with a reason.
  Missing examples, near-misses and section hints are auto-filled from the clusters and marked "auto-filled, check".
- Review workbooks `reports/schema_review/schema_review_<genre>.xlsx` (run `review_v2_s0`). Formula cells were
  checked by reference, not recalculated (LibreOffice is not installed); Excel/Numbers compute them on open.

- **Bug (mine), fixed 2026-10-15:** 14 of the 34 titles in `packs/papers/seeds.yaml` were cut at their first comma
  (unquoted values inside YAML flow mappings; the rest became stray keys). Ids and mappings were unaffected; titles
  are now quoted and the review page's data rebuilt. Found by Parsa on the review page.
- Guided review page for the human pass (`reports/schema_review/page/`, built by `analysis/schema_review_page.py`,
  published as a private artifact with a database for decisions); papers proposals now split c000 into own vs
  existing-approach limitations (47 proposed types).

**Tasks 3.4–3.6 tooling** (`packs/schema_review.py`, stages `review`, `candidate`, `coverage`, `freeze`;
`tests/test_schema_review.py`, `tests/test_induce_schema_gate.py`): the review workbook (yellow columns are the
reviewer's, nothing pre-accepted, formulas in a Summary sheet), read-back to `schema_candidate.yaml` with the gate's
field checks, the coverage check on held-out dev sections (dev documents outside the extraction sample; output
constrained to the schema's ids or "none"; prompt `induce_coverage` v1), and the freeze (needs
`induce.freeze.signed_off_by`, a passing coverage run on the same candidate, and never overwrites `schema.yaml`).
Seeds recorded as data: `packs/privacy/seeds.yaml` (OPP-115's 10 categories), `packs/papers/seeds.yaml` (NeurIPS
2026 checklist, 16 items; ACL ARR checklist of Oct 2024, 18 items; read 2026-10-13, URLs in `data/README.md`).

## 2026-10-13 — Handoff for the Phase 3 session

- `HANDOFF.md` written: state of tasks 3.1–3.6, artefact locations and formats (claims, sections, embeddings, cache),
  tools available at home (7B via MLX, 72B via OpenRouter untested live, no UMAP installed), three open decisions for
  the user (accept 303 papers for 3.2; teacher for 3.3 naming and 3.5; 3.5 method), known issues (oversized paper
  sections, example-label leakage to watch for, nothing committed yet). README points to it.
- Plan: 3.1 and 3.2 status notes updated (not ticked: 3.1 awaits the full eye review, 3.2 awaits the user's
  acceptance of 303 papers); §0.6 Colab entry rewritten (free tier, optional, two possible uses; no Pro units planned).

## 2026-10-12 (night) — Lab session results, merged and verified

**Lab machine (`--device-info`):** DESKTOP-AIPC-03, Windows Server 2022 (10.0.20348), 4 × NVIDIA RTX 5000 Ada Generation
(32 GiB each, sm_8.9), 112 CPU cores, 1023 GiB RAM, torch 2.13.0+cu126, CUDA 12.6, driver 581.42. Raw report and the
four llama-server logs in `runs/lab_session_2026-10-12/`.

**Runs** (all logged in `runs/log.csv`; teacher `llama.cpp-b11280/Qwen/Qwen2.5-32B-Instruct-GGUF/q4_k_m`, prompt
`induce_extract v2:a756e28a41a1`):

| Run | Status | Result |
|---|---|---|
| `eye_v2_32b_s0` (task 3.1) | ok, 2.2 min | spans verbatim 95.3% / 97.8%, 0 failed, 0 empty; quick visual skim by Parsa passed |
| `extract_32b_privacy_s0` (task 3.2) | ok, 107 min | 500/500 documents, 4,606 sections, **21,046 claims**, 97.8% grounded, 40 empty, 0 failed |
| `extract_32b_papers_s0` (task 3.2) | interrupted on purpose at the session deadline, 179 min | **303 complete papers** + 1 partial, 6,912 / 10,782 sections, **44,579 claims**, 97.7% grounded, 5 failed |
| `embed_0.6b_s0` (task 5.1) | ok, 28.8 min | Qwen3-Embedding-0.6B, all 248,181 + 373,478 sections and 27,772 + 17,644 titles, 1024-d, 586 / 345 sections/s |

**Verification at home** (staging folder first, nothing overwritten): zip CRC check passed (11,486 files); every claim
line parses and points to a processed section of a sampled document; all embeddings finite, unit-norm, and aligned
with the local corpus by id **and** text sha1; the lab's 11,451 cache entries are all from the 32B and clash with none
of the 202 local ones; the lab `log.csv` is the local log plus exactly the 4 new rows (old local log kept as
`runs/log.csv.pre_lab_backup`). Merged files are byte-identical to the zip. 232 tests pass.

**Found**
- Throughput of the 32B on 4 × RTX 5000 Ada with llama.cpp: about 390 claims/min with both genres running (GPUs at
  93–97%), ~52 paper sections/min alone. One process with batches of 256 leaves slots idle at batch ends;
  `concurrency=64, batch_size=512` would have filled them (not used).
- **Bug from Phase 2 (mine):** the unarXive loader never split long sections; 2,462 paper sections (0.7%) exceed
  8,000 characters and 127 exceed 24,000 (one had 25,247 tokens and was rejected by the server; handled as a failed
  item). To fix before Phase 4: split them like policy sections, re-link claims to the new sections by their
  verbatim spans, re-embed only the changed sections (the store is keyed by text sha1, so the rest is reused).
- In `extract(skip_failures=True)`, a batch that fails is retried item by item; the 1,045 cache hits in the papers
  run are those replays (4 failing batches), not duplicate work by the server.
- Not done in the lab (by choice): 197 of 500 papers, the 100-paper presence-label pool, the 8B embedding ablation.

## 2026-10-12 — Lab session, Windows: llama.cpp teacher, corpus embedding job (decision record 002)

**Found in the lab:** Windows, 4 × RTX 5000 Ada 32 GB, driver 581.42 (CUDA 13.0), no git, no admin. vLLM cannot run;
a 72B split across consumer GPUs would take 10+ hours for task 3.2.

**Done**
- Plan §0.2 #6 edited first: the recorded schema-induction pass uses the named fallback, Qwen2.5-32B-Instruct Q4_K_M,
  served by llama.cpp b11280 (Windows CUDA 12.4), one server per GPU; prompt confirmed on that model; the 72B via
  OpenRouter stays the fallback. Decision record `reports/decisions/002-lab-teacher-on-windows.md`.
- `lab_session.py` (Windows driver: check, serve, status, eye, extract, pool, stop, embed, pack), `configs/lab.yaml`,
  `configs/embed.yaml`, `requirements-lab.txt` (torch 2.13.0 from the cu126 index; cu128 has no 2.13.0 wheel).
- `OpenAITeacher`: several servers round-robin; `served_as` labels the build in the route and cache key (so a
  resumed run hits the cache whichever port answers); a rejected request is a failed item, not a crash; retries
  double the token budget once only.
- `engine/embed.py` (Embedder over one or several devices, EmbeddingStore keyed by text sha1) and task `embed`
  (`experiments/embed.py`): every section and title of both genres.
- Extraction processes documents in a fixed pseudo-random order and prints an ETA; the eye test now survives and
  shows failed sections.
- Explicit UTF-8 on every file read/write that lacked it (Windows defaults to cp1252).

**Verified on the laptop before the session** with the same llama.cpp build (macOS) and Qwen2.5-0.5B as a stand-in:
llama-server accepts our `response_format` and **enforces** the schema (asked to "say hello", it returned the
required fields); four servers started and stopped by `lab_session.py`; eye test, extraction (two genres, failures
skipped and counted), embedding with a two-worker pool, and pack all ran. 232 tests pass.

**Found**
- The rehearsal exposed two crash paths that the 32B could also hit in a long run: one looping output aborted the
  whole eye test, and retry budgets could exceed a slot's context (fixed above).
- The rehearsal itself wrote 0.5B answers into the teacher cache under the lab's 32B label. They were purged (135
  entries), the rehearsal runs deleted, and the two mislabelled `eye_v2_32b` rows removed from `runs/log.csv`;
  later rehearsals use a distinct `served_as` label. The lab bundle never carries the cache.
- Embedding on the laptop CPU ran at ~1.5 sections/s for the 0.6B model: the full corpus (≈ 620k sections) is a
  GPU job, as planned.

## 2026-10-12 — Lab preparation: task 3.2 implemented, session script, transfer bundle

**Done**
- Task 3.2 code: `induce.stage: extract` (`experiments/induce.py`). Stratified (year × category) sample of 500 dev
  documents per genre, every section ≥ 40 chars: 4,606 privacy and 10,782 paper sections (15,388 teacher calls,
  ≈ 5.4M output tokens). Claims stream to `claims_<genre>.jsonl` per batch with source section, prompt version and
  teacher build; `extract(skip_failures=True)` retries a failing batch item by item instead of aborting a long run;
  reruns replay finished batches from the response cache. vLLM batches of 256 in `configs/gpu.yaml`.
- `lab_session.sh`: setup → download → vllm-check → eye → extract → pack. `setup` reads `--device-info` and picks
  the teacher build from the VRAM rule; `pack` collects everything to take home.
- `lab_bundle/negspace_lab.tar.gz` (240 MB, gitignored; sha256 in `.sha256`): code, processed documents and sections,
  split manifests, the 7B reference eye test, `runs/log.csv`. Verified by unpacking into an empty folder: 218 tests
  pass, both eye-test selections match their text hashes, both 500-document samples resolve.
- Tests: extract stage end to end and failure skipping (`tests/test_induce.py`).

**Found**
- **Correction to my own plan text (§0.2 #6, §0.6, `gpu.yaml`):** the 72B in bf16 is 145 GB of weights; vLLM's 90%
  memory budget on 2×80 GB is 144 GB, so "bf16 on ≥ 2×80 GB" could not even load. Corrected to ≥ 4×80 GB; AWQ
  otherwise, tensor-parallel over every GPU available.
- **Fresh-clone bug:** `tests/test_skeleton.py` required `data/raw/` and `data/processed/`, which are gitignored and
  so absent from any clean checkout. Found by testing the unpacked bundle; the test now requires tracked folders only.
- Real-model smoke of the extract stage (1.5B, smoke corpus): privacy half completed (74 sections streamed); the
  papers half was interrupted on purpose (the 1.5B is slow on long sections) and logged as `interrupted`.

## 2026-10-12 — Task 3.1: the extraction prompt (local iteration done, 72B confirmation pending)

**Done**
- `prompts/induce_extract.md`: given one section, return `{"assertions": [{claim, type_guess, span}]}`. Genre enters
  only through placeholders filled from `genre_info.<genre>` in the config (document kind, author role); format
  examples come from an unrelated genre (a library's rules) so they cannot prime the schema's type labels.
- `packs/schema_induction.py` step 1: output models, prompt rendering, batched extraction, eye-test selection, and
  automatic checks (span is a verbatim quote, snake_case labels, no first-person claims, duplicates, share of the
  section's sentences covered by a span).
- `experiments/induce.py`, stage `eye_test`: a fixed selection of 20 dev sections per genre from 20 different
  documents, balanced over the first/middle/last third (ids and text hashes in
  `prompts/eval/induce_extract_eye_test.json`; a changed corpus is refused), per-genre jsonl, metrics, and a review
  page with spans highlighted in the text, side by side with an earlier run (`induce.eye_test.compare_with`).
- `tests/test_induce.py` (fake teacher, < 1 s). Plan Appendix D: lab runbook for the 72B confirmation.

**Iteration on the local teacher** (`mlx-community/Qwen2.5-7B-Instruct-4bit`, 40 sections, 6–10 min per version):

| Version | Hash | Spans verbatim (privacy / papers) | Verdict |
|---|---|---|---|
| v1 | `e7457d8b7f02` | 95.7% / 89.9% | faithful, but paper claims restate subject matter |
| **v2** | **`a756e28a41a1`** | 94.0% / 93.6% | **candidate**: claims are moves; residuals below |
| v3 | `9c3ede541a2a` | 96.4% / 93.7% | rejected: fixed neither residual, introduced example-label leakage |

**Found**
- **Negative finding (v1):** an assertion-extraction prompt that works for privacy policies fails for papers in a
  way no automatic check catches: every claim was faithful and grounded, but about the paper's *topic* ("MAML is used
  for cross-lingual NER"). Clustering those would have induced topic clusters, not genre types, and quietly broken
  schema induction for genre B. The fix (v2) asks for the *move* the document makes, keeping what defines the kind
  of point (party, data, purpose, duration, evidence) and dropping subject matter.
- **Residuals in v2 on the 7B:** a list of different recipient kinds (suppliers … government … "required by law")
  merged into one claim; one long related-work section still listed 12 cited works. v3's targeted fixes did not
  change the 7B's behaviour, so these look like the small model's ceiling. They are explicit pass criteria for the
  72B (Appendix D).
- **Example leakage is real:** in v3 the 7B copied the example labels onto unrelated claims. Few-shot examples for
  schema induction must stay off-genre and few; the v2 examples showed no leakage.
- About 6% of spans are not verbatim (mostly in sections with `[math]` placeholders). Spans are evidence, not the
  claim, so 3.2 can repair them by fuzzy matching; the 72B is expected to do better (pass bar ≥ 95%).

**Open**
- Confirm v2 on the 72B in the lab (one command, Appendix D), or iterate there. Then tick 3.1.
- Task 3.2's extraction pass is not implemented yet; it should exist before the lab so the session can run it
  straight after the confirmation.

## 2026-10-12 — Phase 2: corpora and the unified schema

**Done**
- 2.1 Corpora downloaded and recorded in `data/README.md` (URL, licence, version, size, citation): Princeton–Leuven
  SQLite release (3.29 GB `.xz` → 48.2 GB), OPP-115 v1.0 (99 MB), unarXive open subset 230324 (4.84 GB, Zenodo;
  a stalled transfer was resumed by a watchdog loop). arXiv per-version e-prints confirmed obtainable
  (`export.arxiv.org/e-print/<id>v<N>`), so OmitBench-revision is feasible for papers.
- 2.2 Every corpus inspected by hand; real schema, field names, flags and formats written into `data/README.md`.
- 2.3 `data/schema.py`: Document / Section / Sentence / Assertion / RevisionPair, char spans, jsonl(.gz) I/O.
- 2.4 Loaders: `privacy_pl` (metadata index first, texts fetched only for the sample), `opp115` (native segments,
  `Section.idx` = segment ID, consolidated labels at threshold 0.75), `unarxive` (one streaming pass through
  parallel `xz`, byte-level category prefilter, two-pass sampling), and `arxiv_versions` (cached, rate-limited e-print
  fetch + LaTeX → sections). Previews in `reports/figures/loader_check_<source>.html`.
- 2.5 `data/segment.py`: heading-aware segmentation (ATX, bold-line, numbered, ALL-CAPS, and plain title lines
  accepted only when a paragraph follows), table-of-contents removal, boilerplate removal (navigation, copyright,
  cookie banners near the document edges only), paragraph-chunk fallback, size normalisation, and a rule-based
  sentence splitter.
- 2.6 `data/dedup.py` (MinHash, LSH clusters, version-chain collapsing) and stratified sampling (year × category)
  in `data/build.py`; 2.6b `data/revisions.py`; 2.7 `data/splits.py` (hash-based, entity-disjoint, forced splits,
  cross-split near-duplicate resolution); 2.8 `data/build.py` as task `build_data` (`configs/data.yaml`).
- Tests: `test_data_schema.py`, `test_data_segment.py` (five documents per genre reproducing the real formats),
  `test_data_dedup_splits.py`, `test_data_loaders.py` (fixtures in the real formats, end-to-end builds of both
  genres). Smoke runs write to `data/processed_smoke/`, never over the real corpus. Loader previews are gitignored:
  they contain full third-party texts and the build regenerates them.

**Corpus (full build, seed 1234)**

| | privacy | papers |
|---|---|---|
| documents (train / dev / test) | 28,440 (22,300 / 3,043 / 3,097) | 17,644 (14,114 / 1,784 / 1,746) |
| sections / sentences | 248,181 / 1,697,753 | 373,478 / 4,981,129 |
| sections per doc (median) | 7 | 19 |
| dates | 1998-07 – 2019-12 | 2018-01 – 2022-12 |
| near-duplicate rate in the sample | 18.6% | 0.3% |
| dropped as cross-split twins | 1,949 | 37 |
| revision pairs (test entities) | 4,029 | 343 (v1 → last version) |
| build time on the M5 | 3.2 min | 4.0 min (+ 92 min one-off e-print fetch, now cached) |

**Sealed test splits** (sha256 of the sorted test doc ids, `data/processed/splits/<genre>.json`):
- privacy `23875735421b73d550af81c5cb49a39471c57c9bd6f2bc8a81c067e16b196c03` (identical across two rebuilds)
- papers `12bc037dec4506f0d082c87e11f55fabad963e1068f45b1345d58f1817207e22` (identical across three rebuilds)

**Found**
- **Sentence splitting:** pysbd measured at 333 sentences/s on the M5 (≈ 4 h for the corpus); spaCy's
  sentencizer at 50k/s but it splits after "Sec.", "Fig.", "et al.,". Replaced by a rule-based splitter with an
  abbreviation list: 425k sentences/s, correct on those cases. Plan 2.5 text updated.
- **Princeton–Leuven:** categories are Webshrinker lists in *alphabetical* order, so there is no primary category;
  we use the rarest listed one. 1,071,487 snapshots share 333,986 texts. After filters (analysis subcorpus, HTML,
  not parked) and one snapshot per site and year: 545,463 candidates over 103,020 sites.
- **Templates:** 18.6% of sampled policies are near-duplicates (Jaccard ≥ 0.85) of another sampled policy, i.e. many
  sites reuse one template. The cross-split guard drops near-copies of test documents from train/dev (1,949), and
  train keeps one document per cluster and year across different sites (176 dropped). This is why train is 22.3k,
  not 25k (still within the 20–30k scope).
- **OPP-115 protection:** a first build silently lost 10 of the 115 OPP-115 policies as near-twins of test documents.
  Human-labelled documents are now protected; their twins are dropped instead. All 115 are in dev.
- **Negative finding, papers:** the open subset (permissively licensed papers only) holds just 17,995 cs.CL/cs.LG
  papers from 2018 on, and 86% are from 2021–2022 (licence uptake). Genre B is therefore 17.6k documents, below the
  20–30k scope, and too thin before 2021 for time-window cohorts. The 2022 ACL-limitations drift analysis (Phase 10)
  cannot be done on this release (sources end 2022-12-31); unarXive 2024 (105 GB) is the recorded upgrade.
- **Residue:** after cleaning, archive URLs, Markdown links and HTML tags occur in 0.006% of privacy sentences.
  unarXive renders `\ref{}` as a bare `REF`, also glued ("TableREF", from `Table~\ref`); both now become `[ref]`
  (761 → 81 sentences with residue; "HREF" becomes "H [ref]", accepted). Run-in headings ("Acceptance of Terms. Welcome
  …") are not detected; such policies (9.4% have no detected heading) fall back to ~1,500-character chunks.
- **Paper revisions:** 607 of 1,746 test papers have more than one version up to 2022-12-31; 1,207 e-prints fetched
  (one request per 3 s, 92 min). Outcome per candidate: 342 pairs, 208 cosmetic only (< 5% change), 32 rewritten
  beyond the same-document bound (> 90%), 16 unparseable (PDF-only and similar), 9 fetch errors (343 pairs after the
  final text cleanup). arXiv 404s are cached as `.missing` markers so rebuilds never re-ask. Median change per pair
  0.29–0.64 by year versus 0.16–0.35 for policies: papers are revised far more heavily than policies.
- **Cohorts must exclude the query's own entity** (one snapshot per site *per year* means a site's other years are
  near-copies): added to plan 5.3.
- **Licence:** OPP-115 annotations are non-commercial; presence heads trained on them inherit that (recorded in
  `data/README.md`; blocks only the Phase 12 product).
- One request in testing carried the user's email in its User-Agent; the fetcher now sends a plain project UA
  (tested).

**Open**
- Disk: the 48 GB Princeton SQLite plus its 3.3 GB `.xz` stay in `data/raw/`; the `.xz` can be deleted to save
  space once nobody needs to re-verify the download.

## 2026-10-12 — Architecture revision: the lean/full tier design (decision 001)

**Decision (plan §0.2 edited first; full record in `reports/decisions/001-lean-architecture.md`, written for the report).**
Every swappable component now has a lean default that runs on the Mac and a full alternative behind one overlay
(`configs/full.yaml`). Changed: §0.2 #6 (teacher reached three ways: 7B MLX proxy locally, 72B vLLM on the server,
72B via OpenRouter), #7 (presence = logistic heads on frozen embeddings trained on OPP-115 + ~2k teacher labels;
ModernBERT-large becomes the full tier), #8 (lean late-interaction model answerai-colbert-small-v1), #9 (embedder
Qwen3-Embedding-0.6B; 8B is the full-tier ablation), #14 (template explanations from evidence; teacher rationale
opt-in), #16 (OmitBench-revision from version chains, validated on 50 docs/genre; blind 150×2 annotation is the
full tier), #18 (scope 20–30k documents per genre, one snapshot per entity per year, chains kept), #24, new #25
(tiers and upgrade path). §0.4 gains `data/revisions.py`, `bench/revision.py`, `configs/{full,openrouter}.yaml`,
`reports/decisions/`. §0.6 and Phases 2–12 rewritten for the new scope (what runs where; gate 5.9 now on the Mac;
Phase 9 validates the revision labels instead of annotating from scratch).

**Why.** Lab GPU access is session-based, so every teacher-in-the-loop step was a week of latency; API-only would
have cost $60–100 (class) and $300–750 (paper baselines); an 8B embedder needs a rented GPU to keep a demo online.
None of the contributions (task formalisation, OmitBench, MAM, packs, LLM gap) depends on model size. The revision
benchmark is cheaper *and* a stronger experiment (authors' own later additions as ground truth).

**Done**
- `configs/base.yaml` holds the lean defaults (`tier: lean`, `presence.head`, `presence.label_sources`,
  `explain.mode`, `bench.real.*`, `baselines.rag_peers`, `corpus.snapshots_per_entity_per_year`, teacher cache);
  `configs/full.yaml` flips all of them; `configs/openrouter.yaml` pins `deepinfra/fp8`, disables fallbacks,
  denies data collection, reads `OPENROUTER_API_KEY`; `configs/local.yaml` uses the MLX teacher.
- `models/teacher.py`: `MLXTeacher` (mlx-lm + Outlines structured output); on-disk response cache keyed by model
  build, messages, schema and sampling (all backends); `TEACHER_STATS` (calls, cache hits, tokens, served_by);
  `OpenAITeacher` sends OpenRouter `provider` routing and records the provider that served each call.
- `experiments/run.py` records `tier`, teacher stats and `teacher_calls` per run (metrics.json and log.csv).
- Stubs `data/revisions.py`, `bench/revision.py`; `pyproject.toml` extra `mac` (mlx 0.32.3, mlx-lm 0.32.0).
- Tests: every config × overlay combination resolves to the expected tier; no key-like strings in configs;
  OpenRouter provider pin and served provider; cache makes reruns free; MLX structured round trip (skipped if the
  model is not cached). `pytest` green.

**Found**
- Qwen2.5-7B-Instruct 4-bit under MLX on the M5: **24 tokens/s**, 1.3–3.9 s load, 2 s per structured call.
  Good for iteration; the 10k-section extraction pass would take ~30 h, so it stays on the 72B.
- OpenRouter (2026-10-12): `qwen/qwen-2.5-72b-instruct` has **one** provider (DeepInfra, fp8, 32k context,
  $0.36 / $0.40 per 1M tokens, structured outputs supported). Single-provider risk; the vLLM route is the backup.
  `meta-llama/llama-3.3-70b-instruct` has ten providers from $0.10 / $0.32. Frontier tiers: Claude Sonnet 5.5 and
  GPT-6.1-sol at $2 / $10, half price on `:batch`; Claude Haiku 5.5 at $0.10 / $0.50.
- Cost estimate (assumed section counts; Phase 2 will measure): class ≈ $10 lean (frontier baseline only) vs
  $60–100 original; paper baselines $100–250 with batch pricing and `rag_peers: 5` vs $300–750.
- Outlines' MLX adapter accepts `sampler=` and `max_tokens=` as mlx-lm kwargs; greedy = `make_sampler(temp=0)`.
- Overlay order matters: `--overlay local --overlay smoke` (compute first, mode last) so smoke's 1.5B teacher wins
  over local's MLX 7B. Documented in `smoke.yaml` and README.

**Open**
- Phase 2 must confirm arXiv v1/v2 full texts are obtainable (OmitBench-revision for genre B depends on it).
- Measure the 0.6B embedder's sections/second on the Mac in 5.1; it sets the overnight budget.
- University server `--device-info` still pending.

## 2026-10-12 — Phase 1: environment and skeleton

**Done**
- 1.1 Repository layout per plan §0.4, with a `.gitignore` for `data/raw`, `data/processed`, pack artefacts
  (`packs/*/{prototypes,classifier,index}`), `runs/*` except `log.csv`, checkpoints, `.venv`, `.env`. The repo root is
  this directory (`negative-space-reader/`) rather than a nested `negspace/` folder; the package/project name is `negspace`.
  `git init` done on branch `main`, nothing committed. Added `experiments/config.py` (config loading, not in §0.4).
- 1.2 Python 3.11.17 venv. Direct dependencies pinned in `pyproject.toml`; full freeze in `requirements.txt`. vLLM is the
  `gpu` extra, wandb the `tracking` extra.
- 1.3 `python -m experiments.run --device-info`. Local machine recorded in README and plan §0.6. Teacher size decided
  (below). University server still to be recorded.
- 1.4 `configs/base.yaml` with every tunable (band thresholds, presence threshold, HDBSCAN sizes, on-the-fly threshold
  left null until their phase); overlays `smoke`, `local`, `gpu`; one config per task (`pack_privacy`, `pack_papers`,
  `induce`, `distil`, `mam`, `bench`, `baselines`, plus `drift` and `serve` so every dispatch target has a config).
- 1.5 `experiments/run.py`: `inherit:` chains, `--overlay`, `--set` (unknown keys rejected, `+key` to add), seeds for
  python/numpy/torch, dispatch to the eight tasks, `runs/<name>_s<seed>/{config.yaml,metrics.json}` and a `runs/log.csv`
  line per run including failed and interrupted runs (with config hash, git commit, device, prompt hashes). An existing
  run directory is never overwritten (a timestamp suffix is added).
- 1.6 `models/teacher.py`: `Teacher` interface with `generate_structured(prompt, pydantic_model)` (+ batch and
  free-text variants); backends `VLLMTeacher` (in-process, xgrammar), `OpenAITeacher` (`vllm serve` or API models; key
  from an env var), `TransformersTeacher` (local stand-in; xgrammar HF logits processor, Outlines fallback). Failed
  validations retry with a doubled token budget. Prompt registry over `prompts/*.md` (YAML header `id`/`version`, sha256
  of the whole file); every rendered prompt's `v<version>:<hash>` is recorded per run. First prompt:
  `prompts/smoke_structured.md` v1.
- 1.7 `tests/test_skeleton.py`, `tests/test_run.py`, `tests/test_teacher.py`. **GATE: `pytest` green: 121 passed in
  8.2 s** on the M5, including the structured round trip on `Qwen/Qwen2.5-1.5B-Instruct` (MPS, xgrammar; ~3 s load,
  ~2 s generation).

**Decision recorded (plan §0.2 #6, edited in the plan first):** teacher = **Qwen2.5-72B-Instruct**, not
Llama-3.3-70B-Instruct. Reasons: ungated on the hub (Llama 3.3 is manually gated); the publisher ships an AWQ build
(`Qwen/Qwen2.5-72B-Instruct-AWQ`, 41.6 GB) that fits one 80 GB card; same family, tokenizer and chat template as the
1.5B smoke stand-in, so prompts iterated locally transfer. Precision follows the server: bf16 + tensor parallelism on
≥ 2×80 GB, AWQ on 1×80 GB / 2×48 GB (current `gpu.yaml` default), `Qwen2.5-32B-Instruct-AWQ` below 48 GB. Within the
options §0.2 #6 and §0.6 already allow; to be confirmed against the server's `--device-info`.

**Found**
- vLLM 0.31.0 (latest, 2026-10-05) pins `torch==2.13.0`, `xgrammar==0.2.7`, `transformers<5.18`. An unpinned install
  pulled torch 2.14.1 / xgrammar 0.2.8 / transformers 5.19.0, which would make `pip install .[gpu]` unresolvable; pins
  lowered to match vLLM so local and server share one dependency set.
- vLLM's structured-output API changed: 0.31.0 uses `structured_outputs_config={"backend": ...}` on the engine and
  `SamplingParams(structured_outputs=StructuredOutputsParams(json=...))` (no per-request backend; older
  `guided_decoding` is gone). Checked against the v0.31.0 source; vLLM cannot be installed on macOS, so
  `VLLMTeacher` is covered by a test against a fake with the same signatures. **It has not run on a GPU yet.**
- Hub check of every model id in the configs (all exist). **Negative finding:** `jinaai/jina-colbert-v2` (§0.2 #8) and
  `nvidia/NV-Embed-v2` (§0.2 #9 fallback) are **CC-BY-NC-4.0**: fine for the class and the paper, not for the product
  layer (Phase 12) without a commercial licence or a swap (e.g. `answerdotai/answerai-colbert-small-v1`, Apache-2.0,
  or bge-m3 dense for late interaction). `Qwen2.5-72B-Instruct` is under the Qwen licence (not Apache), the 32B
  fallback is Apache-2.0.
- Outlines 1.3.3 on transformers fails mid-generation ("No next state found") when the checkpoint's default sampling
  config is active (Qwen2.5 samples by default). Fixed by always passing explicit greedy settings at temperature 0;
  both xgrammar and Outlines then round-trip on MPS and CPU.
- The 1.5B stand-in is faithful on the easy field (retention_days = 90) but sloppy on judgement: it marked a
  cookies-only passage as `mentions_retention: true`. Fine for smoke; a reminder that 3.1's 20-section eye test must
  use the full teacher.
- xgrammar's HF logits processor masks on CPU for non-CUDA devices; it works on MPS with that copy overhead.

**Open**
- Run `python -m experiments.run --device-info` on the university GPU server; record it in README + §0.6 and confirm
  the teacher precision in `configs/gpu.yaml`.
- First GPU session: install `.[gpu]`, run `tests/test_teacher.py` with a vLLM round trip on the real teacher.
