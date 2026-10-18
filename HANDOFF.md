# Session handoff — Phase 4 (presence detection), after session 1

Written 2026-10-15 at the end of Phase 4 session 1. Read with: `CLAUDE.md`, `NEGATIVE_SPACE_PLAN.md` (§0.2, Phase 4,
Appendix A), `CHANGELOG.md` (newest entry: "2026-10-15 (evening) — Phase 4 session 1"), `reports/decisions/001–003`.
Replace this file at the end of the next session.

## Where things stand

- **Oversized paper sections: fixed and verified.** Papers rebuilt (sections 373,478 → 380,736, none over 8,000
  chars; documents and the sealed test hash `12bc037d…` unchanged). Claims re-linked:
  `runs/relink_claims_s0/claims_papers.jsonl` (ids unchanged; `section_idx` is the new index). Use this file, not
  `runs/extract_32b_papers_s0/claims_papers.jsonl`, for anything that needs section indices.
- **Paper re-embedding: done (2026-10-17) on Colab.** The store `data/embeddings/Qwen__Qwen3-Embedding-0.6B/papers/
  sections` matches the rebuilt corpus (380,736 rows; 9,716 imported from an A100 run, checked against the Mac's
  encoder: cosine ≥ 0.99999). Route for future rebuilds: `embed.mode: export` → `notebooks/colab_embed.ipynb` →
  `embed.mode: import` (see `experiments/embed.py`).
- **4.1 privacy labels from OPP-115: done.** `runs/labels_opp115_s0/labels_privacy.jsonl` (3,792 sections × 35 types,
  each label with its OPP-115 annotation ids), counts in `reports/tables/presence_labels_privacy.csv`, the 23 teacher
  types in `runs/labels_opp115_s0/teacher_types_privacy.json`.
- **Label sets fixed:** `packs/<genre>/label_sets.json` (tracked): `human` (300 per genre) and `teacher` (2,000 per
  genre). The stage refuses to overwrite them (`distil.overwrite_label_sets`).
- **4.1 teacher labels: done (2026-10-18).** Run `runs/label_72b_s0_20261018-195259-515194/teacher_labels_<genre>.jsonl`
  (4,600 sections; papers teacher set 1,989/2,000, privacy 1,999/2,000; both reference sets 300/300). Labels are
  stored under the schema version they were made with (v1); readers project them (`packs/schema_versions.py`).
- **Papers pack is version 2 (47 types)**: `method_proposal` merged into `proposes_method`, the three comparison
  types merged into `compares_with_other_methods`; v1 in `packs/papers/history/schema.v1.yaml`. Privacy is version 1.
  Do not run it at the same time as the MPS re-embed if memory is tight (the labelling embedder runs on CPU, so it
  is safe in parallel; the 7B under MLX is not).
- **4.2 reference set: silver (Parsa, 2026-10-17, decision record 004).** No hand labels for now. The 300 fixed
  sections per genre are labelled by a second judge, Claude Haiku 5.5 (`configs/judge.yaml`), with the teacher's
  prompts; run `judge_haiku_s0*` → `teacher_labels_<genre>.jsonl` (≈ $1.50). Privacy also reports OPP-115 (35 types).
  The labelling page (https://claude.ai/artifact/Sy6Z8atpJ45WzJWHNkcKJi, all types, database collections
  `labels_papers` / `labels_privacy`) stays as the upgrade: hand labels later → point `distil.score.reference` at the
  read-back answers and recompute.
- Tests: 280 pass. `main` has commit `f2e2238` (session 1); everything after it is uncommitted. No git remote yet. Commits are authored by Parsa only (no Claude co-author line).

## Decisions taken this session (Parsa)

- 72B run: 5-section trial, then calibration of the second look (v2 chosen), then the full run (stopped, resume above).
- `data_collection_methods` / `data_collection_contests_surveys` (identical OPP-115 label_source): keep v1, decide
  after 4.2/4.3 whether to merge or teacher-label one in a version-2 pack.
- §0.2 #7 wording corrected (3,792 OPP-115 segments, not ~23k); decision record 001 likewise.
- Privacy human set: hand-label all 58 types (OPP-115 mappings proved loose for several types).

## Next steps (in order)

0. Finish the 72B labelling run if it is not done (command above; it resumes from the cache).
2. Scores done (CHANGELOG 2026-10-17/18): teacher vs judge privacy 0.775, papers 0.692 (v2); OPP-115 vs either
   model ≈ 0.60. Open for Parsa: (a) which labels each privacy type trains on (OPP-115 or teacher); (b) whether to
   sharpen the definitions of the still-confused pairs later (a relabelling run, ≈ $5); (c) gate 4.8 against the
   silver reference, to be decided with the classifier's numbers next to the teacher–judge ceiling; (d) the
   fit/evaluation split of the reference set before 4.3 (decision record 004).
3. 4.3 `engine/classifier.py` (`presence.head: linear`), training rows: privacy = OPP-115 labels (35 types) + teacher
   labels (23 types) on the teacher-set policies; papers = teacher labels on the teacher set. Human sets held out.
4. Watch list (unchanged): close neighbours kept apart in 3.4 (see CHANGELOG 2026-10-15); propose a version-2 pack if
   per-type F1 shows they cannot be told apart.

## Tools and keys

- OpenRouter: `.env` holds `OPEN_ROUTER_KEY`; the config reads `OPENROUTER_API_KEY` (prefix above; never print it).
  Pinned to DeepInfra fp8. Spend so far ≈ $0.80 (all Phase 3).
- Local 7B (MLX) for prompt iteration only. Running it while an MPS embedding job runs slows both about 2×.
- Worktree note: this session ran in `.claude/worktrees/negative-space-phase-4-344367`, with `.venv`, `.env`, `.cache`,
  `checkpoints`, `data/{raw,processed,embeddings}` and the old `runs/*` folders symlinked to the main checkout
  (`.git/info/exclude` lists them). New run folders of this session are real folders in the worktree's `runs/`:
  move them to the main checkout's `runs/` when merging (they are gitignored).

## Checks before starting

```bash
.venv/bin/python -m pytest -q        # expected: 280 passed
```
`packs/privacy/schema.yaml` version 1 (58 types); `packs/papers/schema.yaml` version 2 (47 types).
