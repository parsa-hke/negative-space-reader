# Session handoff — Phase 4 (presence detection), after session 1

Written 2026-10-15 at the end of Phase 4 session 1. Read with: `CLAUDE.md`, `NEGATIVE_SPACE_PLAN.md` (§0.2, Phase 4,
Appendix A), `CHANGELOG.md` (newest entry: "2026-10-15 (evening) — Phase 4 session 1"), `reports/decisions/001–003`.
Replace this file at the end of the next session.

## Where things stand

- **Oversized paper sections: fixed and verified.** Papers rebuilt (sections 373,478 → 380,736, none over 8,000
  chars; documents and the sealed test hash `12bc037d…` unchanged). Claims re-linked:
  `runs/relink_claims_s0/claims_papers.jsonl` (ids unchanged; `section_idx` is the new index). Use this file, not
  `runs/extract_32b_papers_s0/claims_papers.jsonl`, for anything that needs section indices.
- **Re-embedding of the 9,728 new paper chunks: NOT DONE.** Stopped at Parsa's request at batch 147/304; the store
  saves only at the end, so it must be rerun (≈ 1.5–2 h on MPS; run it alone, overnight is best). The store on disk
  is the old one (it still serves every unchanged section):
  `caffeinate -i .venv/bin/python -m experiments.run --config configs/embed.yaml --overlay local --name
  embed_papers_split --set pack=papers --set embed.reuse=true`
- **4.1 privacy labels from OPP-115: done.** `runs/labels_opp115_s0/labels_privacy.jsonl` (3,792 sections × 35 types,
  each label with its OPP-115 annotation ids), counts in `reports/tables/presence_labels_privacy.csv`, the 23 teacher
  types in `runs/labels_opp115_s0/teacher_types_privacy.json`.
- **Label sets fixed:** `packs/<genre>/label_sets.json` (tracked): `human` (300 per genre) and `teacher` (2,000 per
  genre). The stage refuses to overwrite them (`distil.overwrite_label_sets`).
- **4.1 teacher labels: approved and half-way set up, not finished.** Prompts `label_presence` v1 + second look
  `label_presence_check` **v2** (calibrated against OPP-115 on 100 privacy sections; v1 in `prompts/history/`).
  Parsa approved the full 72B run (≈ $6.8). It was stopped at Parsa's request after ≈ 1,000 of ≈ 8,500 calls; every
  answer is cached, so rerunning the same command continues for free where it stopped (≈ 2 h, ≈ $6 left):
  `set -a; . ./.env; set +a; export OPENROUTER_API_KEY="$OPEN_ROUTER_KEY"; caffeinate -i .venv/bin/python -m
  experiments.run --config configs/distil.yaml --overlay openrouter --name label_72b --set distil.stage=teacher_labels
  --set 'run.seeds=[0]' --set run.device=cpu --set induce.batch_size=24 --set models.teacher.concurrency=24`
  Output: `runs/label_72b_s0*/teacher_labels_<genre>.jsonl` (written per genre when its sets finish).
  Do not run it at the same time as the MPS re-embed if memory is tight (the labelling embedder runs on CPU, so it
  is safe in parallel; the 7B under MLX is not).
- **4.2 page published (version 2: all types in both genres, papers 50, privacy 58, by Parsa's choice):**
  https://claude.ai/artifact/Sy6Z8atpJ45WzJWHNkcKJi (private). Parsa labels there; answers
  are in its database, collections `labels_papers` and `labels_privacy`, one document per section:
  `{key, doc_id, idx, present, unsure, none, checked_all, note, schema_version, data, saved_at}`.
- Tests: 271 pass. Nothing is committed (Parsa commits; never add Claude as author or co-author).

## Decisions taken this session (Parsa)

- 72B run: 5-section trial, then calibration of the second look (v2 chosen), then the full run (stopped, resume above).
- `data_collection_methods` / `data_collection_contests_surveys` (identical OPP-115 label_source): keep v1, decide
  after 4.2/4.3 whether to merge or teacher-label one in a version-2 pack.
- §0.2 #7 wording corrected (3,792 OPP-115 segments, not ~23k); decision record 001 likewise.
- Privacy human set: hand-label all 58 types (OPP-115 mappings proved loose for several types).

## Next steps (in order)

1. Parsa labels on the page (300 + 300; privacy asks for 23 types per section, papers 50).
0. Finish the two stopped jobs (commands above): the 72B labelling run and the paper re-embed.
2. After Parsa's labels: read back the page answers (`ArtifactData list` of both collections with `out_dir`; write
   them as JSON lines), then `distil.stage: score` with `distil.score.reference` = that file and `name: human` →
   `reports/tables/teacher_vs_human_<genre>.csv` (final vs first pass = the second look's gain and cost). Also score
   the OPP-115-derived labels against the human set (privacy). Plan rule: teacher F1 < 0.85 on a type with ≥ 20 human
   positives → revisit that type before distillation. Then decide per privacy type whether training uses OPP-115 or
   teacher labels.
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
.venv/bin/python -m pytest -q        # expected: 271 passed
```
`packs/privacy/schema.yaml` and `packs/papers/schema.yaml`: `version: 1`, 58 and 50 types.
