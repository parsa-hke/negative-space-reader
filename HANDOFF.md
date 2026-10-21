# Session handoff — Phase 6 closed (6.6 failed for privacy, recorded); Phase 7 starts with 7.1

Written 2026-10-21 at the end of the 6.4/6.5 session. Read with: `CLAUDE.md`, `NEGATIVE_SPACE_PLAN.md` (§0.2 #14,
#20, #21, #23, #24; Phase 7), `CHANGELOG.md` (entries 2026-10-20/21 and 2026-10-21), decision record 007. Replace
this file at the end of the next session.

## Where things stand

- **Phase 6:** 6.1-6.5 ticked; 6.6 unticked with its outcome recorded (papers pass, privacy fail; Parsa: report as a
  negative finding, Phase 7 proceeds; §0.2 #9/#10 unchanged).
- **6.4 this session:** severity-weight ablation `bench_ablation_severity_v2` → `reports/tables/ablation_severity_<genre>.csv`
  (privacy: MRR up, false alarms up, gate still not passed; papers: no weights ≠ 1). Figures from
  `python -m analysis.figures` → `reports/figures/{bench_recall5,level1_minus_uniform,cohort_diagnosis}.png`.
  Cascade vs classifier-only presence **deferred (Parsa)**; full embedder not run (no full-tier pack).
- **6.5 this session:** stage `bench.stage: errors` (`analysis/error_analysis.py`), run `bench_errors_v3` on
  `bench_eval_v2` → `reports/tables/errors_<genre>.csv`. `bench_eval_v2` = `bench_eval_v1` plus
  `<run>/<genre>/untouched.csv`; all recorded tables unchanged. Main finding: missed types are the rare ones
  (Spearman of per-type recall@5 with prevalence 0.90 privacy, 0.56 papers).
- **Synth stage:** progress lines (timestamp, documents tried, ETA), atomic checkpoints, `bench.synth.resume_from`;
  checked with an interrupted smoke run.
- **Uncommitted:** everything from this session **and** the Markdown round-trip session's changes (`data/segment.py`,
  `engine/pipeline.py`, `bench/synth.py` docstring, pack manifests, analyse-gate tables, `tests/test_segment_roundtrip.py`,
  its CHANGELOG entry, two `gate_roundtrip_*` log rows). Nothing was committed: Parsa reviews first. `bench/synth.py`,
  `CHANGELOG.md` and `runs/log.csv` carry changes from both sessions.
- **Hub:** `runs` not pushed since this session's runs; `packs/*/pack.json` differ from the hub (the round-trip
  session's battery-power gate timings; rerun gate 5.9 on mains power for the report).
- Tests: 441 passed.

## Next session, in order

1. Checks below. Commit what Parsa approves (authored by Parsa only).
2. **7.1 Gradio app**, once Parsa approves the design below (or a changed one).
3. 7.3 frontier comparison (Claude Opus 5.5 via Message Batches, 100 synthetic instances per genre; prompt versioned
   in `prompts/`), then 7.4 report via `analysis/tables.py`.

## 7.1 design proposal (not yet approved)

- **Engine split, no genre code:** `analyse()` becomes `prepare(text, pack)` (segment, embed, presence: the slow part,
  cached per upload in the session) + `score(prepared, pack, cohort settings)` (cohort, expectation, rank, evidence:
  fast), so the cohort controls re-rank without re-embedding. `analyse()` stays as `score(prepare(..))`.
- **Ghost placement** (`engine/explain.py`, generic): a type's `section_hint` "under a heading like X" → after the
  section whose heading best matches X (token overlap), else "first/second half" / "near the start" → after the last
  section of that part; one ghost block per position, several absences stacked.
- **Layout (Gradio Blocks):** input row (paste / upload .txt .md .html, pack dropdown with "auto", examples from the
  test split of both genres via `document_markdown`); left 60%: the document as HTML with dashed ghost paragraphs
  ("N% of peers like this one state X; this document does not", prevalence, severity weight); right 40%, two tabs:
  **Cohort** (k, category filter, date-window slider, mean similarity, fallback flag) and **Evidence** (selected
  absence: peer count, up to five peer sentences with document ids, the template explanation).
- **Auto-detect:** both packs use the same embedder; pick the pack whose top-10 neighbours have the highest mean
  similarity, show the margin; below a threshold say "no matching pack" (on-the-fly mode is Phase 10).
- **Off by default, flagged when on (§0.2 #14, #21):** teacher rationale / suggested wording and "compare with a chat
  model" (baseline (e)); both send text off the machine, need an API key, and are hidden on a public Space. The
  chat-model toggle waits for the 7.3 prompt. Uploads are never stored (`serve.store_uploads: false`).
- **Wording:** prevalence, never verdicts ("non-compliant", "missing", "deficient" do not appear).
- **Tests:** ghost placement unit tests; the Blocks app built (not launched) on a tiny fake pack; `prepare`/`score`
  equals `analyse` on one corpus document. Launch: `experiments/run.py --config configs/serve.yaml` (task `serve`).
- **Open questions for Parsa:** is the 60/40 two-tab layout right? Severity shown as a number or as a badge? Should
  the CPU Space ship both packs (memory and pack size to be checked)?

## Watch

- Never push a hub bundle while a job writes into it.
- FAISS and PyTorch must not both initialise in one macOS process.
- `post_hoc_best` is not a gate result. Severity-weighted numbers are an ablation, not a gate result.
- Synth checkpoints (`checkpoint.pkl`) are deleted when a run finishes; an interrupted run's checkpoint must be pushed
  (`hub_sync push --bundles runs`) before the Colab runtime is gone, then resumed with `bench.synth.resume_from`.
- Commits are authored by Parsa only; no Claude co-author line.

## Checks before starting

```bash
.venv/bin/python -m pytest -q        # expected: 441 passed
git status --short                   # this session's and the round-trip session's changes, unless committed
.venv/bin/python -m experiments.hub_sync status --bundles runs,packs
```
