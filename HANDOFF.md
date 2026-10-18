# Session handoff — start Phase 5 (packs, index, cohort, level-1 expectation)

Written 2026-10-18 at the end of Phase 4 (all eight tasks ticked; papers below its gate by Parsa's decision). Read with:
`CLAUDE.md`, `NEGATIVE_SPACE_PLAN.md` (§0.2, §0.6, Phase 5, Appendix A), `CHANGELOG.md` (entries of 2026-10-18 and
2026-10-17), `reports/decisions/001–005`. Replace this file at the end of the next session.

## Where things stand

- **Phase 4 is closed** (CHANGELOG 2026-10-18; decision records 004, 005). Presence model per pack = linear heads on
  the stored Qwen3-Embedding-0.6B section vectors (teacher labels, C 3) + ColBERT prototype scores, combined by a
  logistic layer (`packs/<genre>/cascade.json`; no reranker in the lean tier). Against the silver reference (judge
  Claude Haiku 5.5, eval half): privacy macro F1 0.619 (gate: 0.801 of the teacher's 0.773, pass), papers 0.497 (0.722
  of 0.688, not met, recorded). Profiles for the whole indexable corpus in `packs/<genre>/profiles/part-*.parquet`.
- **Schemas:** privacy version 1 (58 types); papers **version 2** (47 types; v1 in `packs/papers/history/schema.v1.yaml`;
  labels made under v1 are read through `merged_from`, `packs/schema_versions.py`).
- **Pack artefacts** (gitignored, synced through the HF store): `packs/<genre>/classifier/linear_teacher/`,
  `prototypes/`, `cascade.json`, `cascade_full.json` (with reranker, from `cascade_v1_s0`), `profiles/`.
- **Compute (plan §0.6, from Phase 5 on):** GPU work runs on **Colab Pro (A100)** through
  `notebooks/colab_runner.ipynb` (clone → install `requirements-colab.txt` → `hub_sync pull` → `experiments.run ...
  --overlay colab` → `hub_sync push`); the Mac pulls with `.venv/bin/python -m experiments.hub_sync pull --bundles
  runs,packs`. Tests and smoke stay on the Mac. Colab secrets Parsa must add once: `HF_TOKEN` (read+write on the
  dataset repo), `GH_TOKEN` (fine-grained, read-only contents of the GitHub repo), and API keys when a run needs them.
- **Stores:** GitHub `parsa-hke/negative-space-reader` (private; `main`); Hugging Face dataset
  `parsa-hke/negative-space-reader-data` (private; bundles in `configs/base.yaml` `hub:`). **Uploaded and verified 2026-10-19:** 514 files, 4.42 GB (data, embeddings, the teacher cache archive, packs incl. profiles, runs, the merged run log); every local bundle file matches the hub.
- **APIs:** OpenRouter (teacher, 72B; `.env` `OPEN_ROUTER_KEY` → `OPENROUTER_API_KEY`); **Anthropic API** for the
  frontier comparison (7.3: Claude Opus 5.5, Message Batches; $100 credit; needs `ANTHROPIC_API_KEY` in `.env` and in
  Colab secrets, and an `anthropic` backend in `models/teacher.py` written with the official SDK). Spend so far:
  OpenRouter ≈ $0.80 (Phase 3) + ≈ $8 (Phase 4 labelling, trial, calibration) + the judge ≈ $1.05.
- Tests: 329 pass. Commits are authored by Parsa only (no Claude co-author line).

## First job of Phase 5: document-level presence (a Phase 4 finding)

Profiles mark a type present when any section passes a section-level threshold, which inflates document-level
presence (papers: `states_own_limitations` in 96.6% of papers; privacy 20 of 58 types per policy). A type that looks
present everywhere can never be ranked as missing. Before level-1 expectation (5.4) and ranking (5.5), calibrate
presence at document level, test first, generic (no genre code): for example thresholds fitted on documents whose
every section is labelled (OPP-115 policies; papers have one labelled section per teacher-set paper, so propose an
approach and ask), or a noisy-OR over calibrated section probabilities with a per-type document threshold. Profiles
keep only the best section's probability, so storing every section's probabilities may be needed; rebuilding
profiles is a Colab job from now on.

## Phase 5 at a glance (plan Phase 5)

5.1 embedder (corpus embeddings exist; document embedding = mean of sections + title, §0.2 #9), 5.2 Index (FAISS +
Qdrant), 5.3 cohort (never the query's own entity), 5.4 level-1 expectation with Wilson intervals, 5.5 rank + evidence
(prototypes double as peer evidence), 5.6 `analyse()`, 5.7 pack builder, 5.8 explanations (template), 5.9 gate
(`analyse()` under 10 s on the Mac). For 5.9: papers presence on a new document costs ~0.5 s per section on the Mac
(the section embedding), so a 20-section paper is near the 10 s budget.

## Open items for Parsa (none blocks Phase 5)

- Papers presence upgrade (SetFit or ModernBERT fine-tune on Colab): the recorded upgrade for the unmet gate.
- Hand labels for the reference set (decision record 004): the upgrade from silver to human.

## Watch

- The 72B labelling needed up to 7 of 8 rate-limit retries; if OpenRouter is busier, raise `rate_limit_retries`.
- Close-neighbour types (CHANGELOG 2026-10-15/18) remain hard to separate; per-type F1 in
  `reports/tables/presence_cascade_<genre>.csv`.

## Checks before starting

```bash
.venv/bin/python -m pytest -q        # expected: 329 passed
```
`packs/privacy/schema.yaml` version 1 (58 types); `packs/papers/schema.yaml` version 2 (47 types).
