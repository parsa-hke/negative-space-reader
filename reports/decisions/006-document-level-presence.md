# Decision record 006: document-level presence from whole labelled documents

**Date:** 2026-10-20 · **Status:** adopted (lean tier) · **Plan sections changed:** none in §0.2 (#4 already says
presence "is detected at section level and aggregated to document level"; this record fixes *how*); notes added to
tasks 4.7 and Phase 5.

## Context

The omission score of §0.2 #1 multiplies the expectation of a statement type by (1 − P(present in the document)),
and the level-1 expectation is the share of a document's peers whose profile has the type present (§0.2 #11). Both
therefore need presence at *document* level, while the presence model of Phase 4 is trained and evaluated at
*section* level. The Phase 4 profiles aggregated with a rule: a document states a type when its best section passes
the section-level threshold (tuned for section-level F1).

At the end of Phase 4 the profiles looked inflated: research papers stated `states_own_limitations` in 96.6% of
cases and `states_limitations_of_existing_approaches` in 99.2%; privacy policies stated 20 of 58 types on average.
A type that seems present everywhere can never be ranked as conspicuously absent, so this had to be settled before
ranking. But "inflated" was a guess: no document in either corpus had every section labelled, so document-level
presence had never been measured. Research papers carried one labelled section each; OPP-115's privacy policies are
cut into about 33 short annotation segments per policy, against about 9 sections for a corpus policy, so a rule
fitted on them would not transfer.

## Decision

1. **Label whole documents.** 150 development-split documents per genre (stratified by year and category, only
   from sources the indexed corpus has, never in the schema-induction sample, another label set or the test split)
   were labelled section by section, every section, by the second judge of decision record 004 (Claude Haiku 5.5,
   the same two-pass prompts). A document states a type when one of its sections does. The calls went through the
   Anthropic API with the official SDK and Message Batches (half price): 4,227 sections, 8,454 calls, about $2.60.
   A route check showed the judge agreeing with its own earlier OpenRouter labels at micro-F1 0.94 (privacy) and
   0.85 (papers), against 0.78 / 0.69 for teacher vs judge.
2. **Fit a document-level model.** Per document and type, four or five features of the section probabilities (the
   best and second-best section probability on the logit scale, the log of their sum, the log of the number of
   sections, optionally a noisy-OR), one logistic model shared across types with a regularised intercept per type
   (partial pooling: 75 fitting documents cannot support a free model per type), regularisation chosen by
   cross-validation grouped by document, the feature set chosen by cross-validated log loss. Fitted on one half of
   the labelled documents, scored on the other; the deployed model is refitted on all of them.
3. **Profiles v2.** Every section's probabilities are stored once (`packs/<genre>/section_probs`, one Colab job),
   so the document model is applied, and can be refitted, on the Mac without a GPU. A profile's `prob_` is now the
   document-level probability, `bit_` that probability ≥ 0.5, and `best_` the old best-section probability.

## Evidence (evaluation half, 75 documents per genre, against the judge)

| | Privacy v1 | Privacy v2 | Papers v1 | Papers v2 |
|---|---|---|---|---|
| Macro-F1 (types with ≥ 5 positives) | 0.687 | **0.754** | 0.734 | **0.789** |
| Micro-F1 | 0.739 | **0.808** | 0.789 | **0.850** |
| Mean absolute error of per-type prevalence | 0.136 | **0.063** | 0.152 | **0.094** |
| Expected calibration error | 0.150 | **0.018** | 0.218 | **0.019** |
| Types per document (judge: 26.8 / 27.8) | 22.8 | 26.1 | 26.0 | 29.7 |

The diagnosis "inflated" was wrong in its general form. The v1 rule was **miscalibrated in both directions**: for
privacy it overstated 8 types by more than five points (`limits_internal_access` 71% vs 21% labelled) and understated
33 (`sends_marketing_communications` 27% vs 71%); for papers it overstated 16 (`gives_illustrative_example` 95% vs
55%) and understated 20 (`task_description` 28% vs 96%). The headline case was mostly real: the judge finds own
limitations stated in 82.7% of papers. Across the indexed corpus, v2 moves `limits_internal_access` from 69% to 21%,
`sends_marketing_communications` from 19% to 69% and `code_and_data_availability` from 14% to 39%.

## Options considered

| Option | Why not |
|---|---|
| Keep the best-section rule and raise its threshold | Moves every type the same way; the errors run in both directions. |
| Noisy-OR over calibrated section probabilities with a guessed threshold | Never below the best section, so it would inflate further; unfittable without document labels. |
| Fit on OPP-115 policies | A different unit (33 short segments per policy); privacy only. |
| Teacher (72B) labels for the documents | Possible; the judge was chosen because the section model's probabilities were calibrated against the judge's labels, and to spend the Anthropic credit rather than OpenRouter. |
| Hand labels | The upgrade path of decision record 004; 300 whole documents is beyond the schedule. |

## Consequences

- Every profile and every expectation now rests on judge labels at document level as well as at section level; the
  limitations section must say that shared blind spots of the judge are invisible here too (decision record 004).
- The labelled documents are also a document-level silver reference for the benchmark's false-alarm rate (§0.2
  #15).
- A presence bit at 0.5 saturates for near-universal types (`states_own_limitations`: bit set in 98.6% of papers,
  mean probability 0.82 ≈ the judge's 82.7%). Level-1 expectation stays the plan's bitvector mean; the mean
  probability (`expectation.profile: prob`) is the Phase 6 ablation.
- Residuals: papers v2 now overstates a few near-universal types (`claims_broader_applicability` 100% vs 67% on the
  evaluation half) and understates some rare ones (`generates_synthetic_data` 15% vs 36%).
- Upgrade: more labelled documents, hand labels, or a full-tier section model; the document model is refitted in
  seconds from the stored section probabilities (`distil.stage: doc_presence`, or the pack builder).

## Addendum 2026-10-20: papers labels restored from the cache

The papers judge labels were lost on the Mac (a hub pull replaced the finished file with a partial copy pushed
mid-run; `hub_sync` fixed) and rebuilt from the teacher cache at no cost. Sections with duplicated text now share one
answer (13,703 labels vs 13,698). Refitted on the restored labels, the same model is chosen; papers eval-half macro F1
**0.788** (above: 0.789), ECE **0.020** (0.019), prevalence error 0.094 (unchanged). The papers profiles v2 in use are
those of the refit (0.13% of bits differ from the first v2). Privacy is unaffected.
