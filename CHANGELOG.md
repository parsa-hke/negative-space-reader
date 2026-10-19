# Changelog

Date, what was done, what was found (negative findings included). Newest first.

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
