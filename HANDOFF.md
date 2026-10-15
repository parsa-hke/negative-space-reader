# Session handoff — start Phase 4 (presence detection)

Written 2026-10-15 at the end of the Phase 3 session (3.1–3.6 done; both schemas frozen). Read with: `CLAUDE.md`,
`NEGATIVE_SPACE_PLAN.md` (§0.2, Phase 4, Appendix A), `CHANGELOG.md` (newest entries first: 2026-10-15 and
2026-10-13), `reports/decisions/001–003`. Replace this file at the end of the next session.

## Where things stand

- **Phase 3 is complete (GATE 3.6 passed 2026-10-15).** `packs/privacy/schema.yaml` (58 types) and
  `packs/papers/schema.yaml` (50 types), `version: 1`, `status: frozen`, signed off by Parsa, coverage run
  `runs/coverage_72b_v4_s3` (unmapped sections after the second look: privacy 2%, papers 0%). **Frozen: any change
  is a new pack version, recorded in CHANGELOG.**
- Each type has `id, name, definition, positive_examples (2), near_miss, severity_weight, section_hint,
  label_source` (privacy: OPP-115 values or `teacher`), `source_clusters`; each schema has `seed_mapping` (every
  OPP-115 category / NeurIPS / ACL item → types, or an exclusion reason) and `pruned_clusters`.
- How the schemas were made: CHANGELOG 2026-10-13/15, decision record 003 (claim representation), review answers in
  `reports/schema_review/decisions/`, the review page https://claude.ai/artifact/BXtRWoATairWDT6uvZiweU (its database
  holds Parsa's answers; `analysis/schema_review_page.py` rebuilds its data).
- Tests: 248 pass. Nothing is committed (Parsa commits; never add Claude as author or co-author).

## First job of the next session: the oversized-section fix (Phase 2 bug, before any Phase 4 work)

The unarXive loader never applied `segment.max_section_chars`: 2,462 of 373,478 paper sections exceed 8,000
characters and 127 exceed 24,000 (one has 25,247 tokens). Phase 4 labels and classifies sections, so fix it first.
1. Test first (tiny fixture): a long unarXive section is split on paragraph breaks into chunks ≤ `max_section_chars`,
   like policy sections (`data/segment.py` already does this for genre A; reuse it, no genre-specific code).
2. Apply it in `data/loaders/unarxive.py`; rebuild papers with `configs/data.yaml` (through `experiments/run.py`).
3. **Check the sealed test split is unchanged** (hash `12bc037d…`; splitting sections does not change document ids,
   so it must be). Check document counts and section counts before/after; record both in CHANGELOG.
4. Re-link the extracted claims (`runs/extract_32b_papers_s0/claims_papers.jsonl`) to the new section indices by
   their verbatim `span` (97.7% are verbatim; report the rest). The frozen schema does not change; the claims feed
   prototypes and evidence later.
5. Re-embed only changed sections: `EmbeddingStore` is keyed by text sha1, so unchanged sections are reused (MPS,
   keep the 7B/MLX idle meanwhile; speed varies 3× with memory pressure).

## Phase 4 at a glance (plan Phase 4; lean tier)

- **4.1 labels.** Privacy: from `data/processed/privacy/opp115_labels.jsonl` through each type's `label_source`
  (format: one alternative per entry, `Category`, or `Category :: Attribute = Value`, `&` = all must hold; values are
  OPP-115's exact names). 23 of the 58 privacy types say `teacher` and need teacher labels. Papers: new prompt
  `prompts/label_presence.md`, ~2,000 train-split sections via the 72B (OpenRouter). **Lesson from 3.5:** with
  ~50–58 definitions in one prompt the 72B misses about 1 statement in 30; design labelling with the same remedy
  (retrieve the most similar types per section/sentence, or a second look on negatives) and measure it in 4.2.
- **4.2 human sanity set (300 sections per genre)** is the next big piece of Parsa's own work. A guided page like
  the 3.4 review page worked well (one section at a time, plain-language help, answers in the page database; avoid the
  two page faults found in 3.4: unnamed split halves and "checked" items with empty fields).
- **4.3–4.8** classifier (`presence.head: linear` first), prototypes, cascade, ablations, profiles, gate macro-F1
  ≥ 0.80.
- **Watch:** close neighbours that the review kept apart (papers: `performance_comparison`,
  `performance_comparison_across_other_works`, `method_comparison`, `reports_experimental_results`;
  `proposes_method`/`method_proposal`; privacy: `policy_coverage`/`states_policy_scope_and_commitment`,
  `provides_third_party_links`/`third_party_sites_and_their_policies`,
  `data_collection_methods`/`data_collection_contests_surveys`). If per-type F1 shows they cannot be told apart,
  propose a version-2 pack with merges; do not edit `schema.yaml` v1.

## Tools and keys

- OpenRouter: Parsa's `.env` holds `OPEN_ROUTER_KEY`; the config reads `OPENROUTER_API_KEY`. Prefix commands with
  `set -a; . ./.env; set +a; export OPENROUTER_API_KEY="$OPEN_ROUTER_KEY";` and never print the key. Pinned to
  DeepInfra fp8. Spend so far in Phase 3 ≈ $0.80 (naming ≈ $0.05, coverage ≈ $0.70).
- Local 7B (MLX) is fine for prompt iteration but too weak for classification against 50+ types (coverage: 40–50%
  false "none").
- New pinned dependencies this phase: `umap-learn` 0.5.12 (+ numba, llvmlite, pynndescent), `openpyxl` 3.1.5.
  LibreOffice is not installed (workbook formulas are computed when opened in Excel/Numbers).

## Checks before starting

```bash
.venv/bin/python -m pytest -q        # expected: 248 passed
```
`packs/privacy/schema.yaml` and `packs/papers/schema.yaml` exist with `version: 1`, 58 and 50 types.
