# Session handoff — finish Phase 5 (gate 5.9, teacher-mode check, wrap-up)

Written 2026-10-20 at the end of Phase 5 session 1. Read with: `CLAUDE.md`, `NEGATIVE_SPACE_PLAN.md` (§0.2, §0.6,
Phase 5 incl. the new 5.0), `CHANGELOG.md` (entry 2026-10-19), `reports/decisions/006-document-level-presence.md`
(and 004–005 for context). Replace this file at the end of the next session.

## Where things stand

- **Uncommitted work on the Mac** since `a0b47a6` (pushed): pack builder (`packs/builder.py` `build`/`pack_stages`,
  `experiments/build_pack.py`), gate stage `analyse_gate`, teacher explanation mode (`engine/explain.py`,
  `prompts/explain.md` v1), `models.embedder.query_batch_size: 4`, manifests `packs/<genre>/pack.json` (tracked),
  decision record 006, plan notes (4.7, 5.0), CHANGELOG. Tests: 393 pass. **Commit and push when Parsa asks**
  (authored by Parsa only, no Claude co-author line).
- **Document-level presence done (5.0, decision record 006):** 150 whole dev documents per genre labelled by the
  judge (Claude Haiku 5.5 via the Anthropic API, Message Batches, run `judge_docs_haiku_s0`, ≈ $2.60); document model
  `engine/doc_presence.py` (eval half: macro F1 privacy 0.687 → 0.754, papers 0.734 → 0.789; ECE ≈ 0.02); **profiles
  v2** in `packs/<genre>/profiles` (v1 in `profiles_v1`). Finding: v1 was miscalibrated in both directions, not
  uniformly inflated; `states_own_limitations` is really in ≈ 83% of papers.
- **Phase 5 built (tests first):** 5.2 index (FAISS + Qdrant), 5.3 cohort + report, 5.4 level 1 (Wilson), 5.5 rank +
  evidence, 5.6 `analyse()` (`engine/pipeline.py`, `packs/spec.py` `GenrePack.load`), 5.7 builder, 5.8 template +
  teacher mode, 5.9 gate stage. Indexes: privacy 22,300 / papers 14,114 documents.
- **Colab jobs done:** `section_probs_v1` (every section's probabilities, train + dev) and **`sentence_evidence_v1`**
  (5.3M sentences, A100, 2 h 27 min, commit `a0b47a6`): 74 parts in the hub under `packs/<genre>/evidence/`, **not yet
  pulled to the Mac**. Parsa may delete the Colab runtime; nothing is left there.
- Stores: hub `parsa-hke/negative-space-reader-data` (packs synced up to profiles v2 and the manifests; evidence came
  from Colab). Anthropic key in `.env` as `ANTHROPIC_API_KEY` (commands read it with grep, never print it).

## Next session, in order (≈ 45 min if the gate passes; 2–3 h if papers needs speeding up)

1. `hub_sync pull --bundles runs,packs`; check `packs/<genre>/evidence/` (45 / 29 parts).
2. Pack builder per genre: `--config configs/pack_<genre>.yaml --set build.adopt=true` (adopts `sentence_evidence`;
   every other stage should be skipped), which runs the **gate 5.9** (`analyse_gate`: 5 held-out dev documents per
   genre, template explanations, every document < 10 s on the M5). Look at a report's quotes by eye.
   Risk: very long papers. Measured: a 26-section paper took 15.0 s before two fixes (embedding 8.0 s → 2.6 s with
   `query_batch_size: 4`; query-time evidence 6.7 s → stored sentences).
3. One real call of `explain.mode: teacher` (5.8) on one document, the plan's teacher (72B, OpenRouter; `.env`
   `OPEN_ROUTER_KEY` → `OPENROUTER_API_KEY`), under a cent; check the wording stays within the quotes.
4. Record 5.1's Mac timing (from the gate), tick 5.1–5.9 in the plan, CHANGELOG, HANDOFF, commit/push on request.
5. Optional, if Parsa asks: related-work search + novelty memo (pasted prompt item 4).

## Watch

- FAISS and PyTorch must not both initialise in one macOS process (libomp clash); `engine/index.py` loads FAISS only
  for unfiltered approximate search. Keep it that way.
- Privacy cohorts repeat sites across years (88% distinct entities): possible Phase 6 ablation (one snapshot per
  entity). Level 1 uses bits (plan); mean probability is the Phase 6 ablation (bits saturate common types).
- The live dashboard (`analysis/progress_server.py`, launch config `progress`) shows any `runs/<name>.out` written
  in the last 24 h.

## Checks before starting

```bash
.venv/bin/python -m pytest -q        # expected: 393 passed
git status --short                   # expected: the uncommitted work listed above
```
