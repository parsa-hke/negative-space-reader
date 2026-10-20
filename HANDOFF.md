# Session handoff — start Phase 6 (OmitBench-synthetic, first results)

Written 2026-10-20 at the end of Phase 5 session 2. Read with: `CLAUDE.md`, `NEGATIVE_SPACE_PLAN.md` (§0.2 #1, #11,
#13–#15, #19; Phase 6), `CHANGELOG.md` (entries 2026-10-20 and 2026-10-19), `reports/decisions/006-document-level-presence.md`
(with its addendum). Replace this file at the end of the next session.

## Where things stand

- **Phase 5 closed.** Gate 5.9 passed on the M5 (MPS): privacy max 1.98 s, papers max 3.31 s per document
  (`reports/tables/analyse_gate_<genre>.csv`); both packs load from `packs/<genre>/pack.json`. Tasks 5.1–5.9 ticked.
  §0.2 #14 edited to record the evidence retrieval change of 2026-10-19 (stored sentence evidence, not the prototype
  index); no decision changed.
- **5.8 teacher mode checked** on one policy (run `explain_teacher_privacy_s0`, 72B via OpenRouter, ≈ $0.001): no
  invented specifics; suggested wordings often copy peer sentences verbatim (product note, not a bug).
- **New task `analyse`** (`experiments/analyse.py`, `configs/analyse.yaml`): named corpus documents → one Report JSON
  each in the run folder; teacher mode also lists unsupported words.
- **Lost file restored:** a hub pull on 2026-10-20 00:49 had emptied `runs/judge_docs_haiku_s0/teacher_labels_papers.jsonl`.
  Rebuilt from the teacher cache ($0, run `judge_docs_papers_from_cache_s0`); papers `doc_presence` and profiles v2
  refitted (0.13% of bits changed; macro F1 0.789 → 0.788). `hub_sync pull` now keeps local files newer than the
  hub's differing copy (test). CHANGELOG 2026-10-20 has the full account.
- **To push (if not done at the end of this session):** git commit + push (authored by Parsa only, no Claude
  co-author line), then `hub_sync push --bundles runs,packs`. **The hub still holds the empty papers labels file and
  the first papers profiles v2 until that push; push before any Colab job pulls packs.**
- Tests: 397 pass.

## Next session: Phase 6, in order

1. Checks below; `hub_sync status --bundles runs,packs` should show nothing differing (if the push was done).
2. **Design 6.1 before coding** (propose to Parsa first): which sentences "carry" type *t* in a test document.
   Test documents have **no stored section or sentence probabilities** (`section_probs` ran on train + dev only), and
   the full presence model on sentences costs ≈ 9 ms/sentence on the Mac. Likely a Colab job (`section_probs` +
   sentence probabilities over the test split, as `sentence_evidence_v1` did for train) before `bench/synth.py`.
   Then verify every deletion by re-running presence on the edited document (the type must now be absent at
   document level, `doc_presence`). Regimes: single type, three types, whole section; 500 instances each per genre.
3. 6.2 `bench/metrics.py` (recall@k, MRR, nDCG@10, false alarms on untouched documents against the document-level
   silver labels of decision record 006, ECE, bootstrap CIs), tests first.
4. 6.3 baselines (a) genre-uniform prevalence, (b) level-1 cohort prevalence → `reports/tables/bench_synth_<genre>.csv`.
5. 6.3b OmitBench-revision (privacy 4,029 / papers 343 revision pairs; time-windowed cohorts, §0.2 #19).
6. Gate 6.6: level 1 beats genre-uniform on recall@5 outside the CI, both genres.

Phase 6 ablations already recorded as candidates: mean probability instead of bits (`expectation.profile: prob`,
bits saturate near-universal types); one snapshot per entity in privacy cohorts (88% distinct entities); k ∈ {20,
50, 200, 1000}; category filter on/off.

## Watch

- FAISS and PyTorch must not both initialise in one macOS process (libomp clash); `engine/index.py` loads FAISS only
  for unfiltered approximate search. Keep it that way.
- Never push a bundle while a job is still writing into it (the cause of today's lost file). Pull now protects newer
  local files, but a push mid-run still puts a partial file on the hub.
- Pass `--set 'run.seeds=[0]'` to deterministic stages (base.yaml runs three seeds).
- Cosmetic: a Report's `count` is a float (188.0); the template formats it with `:g`.
- Optional, if Parsa asks: related-work search (GDPR completeness checking of privacy policies, checklist
  verification for papers, AbsenceBench and omission detection, set prediction) and a short novelty memo.

## Checks before starting

```bash
.venv/bin/python -m pytest -q        # expected: 397 passed
git status --short                   # expected: clean if the commit was made at the end of 2026-10-20
```
