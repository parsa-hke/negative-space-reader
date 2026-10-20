# Decision record 007: gate 6.6 fails for privacy; the deletion leak; Phase 7 proceeds

**Date:** 2026-10-21 · **Status:** adopted (lean tier) · **Plan sections changed:** task 6.6 (outcome and decision
recorded; edited before this record); notes on 6.1-6.4. No §0.2 decision changed: the cohort settings of §0.2 #9 and
#10 stay as they are.

## Context

Gate 6.6 asks level 1 (cohort prevalence, §0.2 #11) to beat genre-uniform prevalence on recall@5 by a margin
outside the bootstrap CI, in both genres. Criterion (Parsa, 2026-10-20): the paired bootstrap of the per-document
recall@5 difference on OmitBench-synthetic, single_type regime, 95% CI above zero. The plan says that a failure means
"the schema or the cohort retrieval is wrong; diagnose before Phase 7".

## Result

| Recall@5, level 1 − genre-uniform (paired, 95% CI) | Privacy | Papers |
|---|---|---|
| Synthetic, single_type (the gate) | +0.011 [−0.010, 0.033] **fail** | +0.108 [0.074, 0.141] **pass** |
| Synthetic, three_types | −0.015 [−0.028, −0.004] | +0.075 [0.059, 0.092] |
| Synthetic, whole_section | +0.004 [−0.015, 0.022] | +0.069 [0.042, 0.098] |
| Revision, the plan's filter (bootstrap over entities) | −0.010 [−0.023, 0.002] | +0.107 [0.046, 0.166] |
| Revision, every added type | −0.009 [−0.017, −0.002] | +0.041 [−0.003, 0.086] |

Both systems are far above random (privacy single_type: 0.385 / 0.396 vs 0.159). The revision filter selects
pairs with the level-1 quantity, which favours level 1, and privacy level 1 still does not win.

## Diagnosis (run `bench_diagnose_v1`, `reports/tables/cohort_diagnosis_<genre>.csv`)

Two measurements per cohort variant: how well cohort prevalence predicts which types an untouched test document
states (800 documents per genre: Brier score, within-document AUC of present over absent types), and recall@5 on
the synthetic instances with the cohort retrieved from the edited document's vector (what the system sees) and from
the original document's vector (before the deletion).

1. **Privacy cohorts carry little information about practices.** Against genre prevalence, the default cohort lowers
   the Brier score by 3.4% for privacy (0.176 → 0.170) and by 20% for papers (0.127 → 0.102); within-document AUC
   0.854 → 0.880 vs 0.922 → 0.959.
2. **Two default settings weaken privacy cohorts.** The category filter (site category: education, business,
   shopping, ...) does not track data practices, and the title part of the document vector (weight 0.5) mostly
   encodes the brand ("Privacy Policy | X"). Without both, privacy within-document AUC is **0.915** [0.912, 0.919]
   (papers 0.963). k (20, 50, 1000), one snapshot per entity and mean probability change little on their own.
3. **The deletion leaks into the cohort (main cause).** Level 1 retrieves the peers of the edited document, and a
   policy that loses a topic moves toward the policies that do not state it, so the deleted type's cohort prevalence
   drops exactly where it is needed. Privacy single_type, level 1 − genre-uniform:

   | Cohort | Edited vector | Original vector | Leak (original − edited) |
   |---|---|---|---|
   | Default | +0.011 [−0.011, 0.033] | +0.026 [0.005, 0.047] | +0.015 [0.005, 0.026] |
   | No title, no category filter | +0.013 [−0.020, 0.048] | **+0.069 [0.036, 0.102]** | **+0.056 [0.039, 0.073]** |

   The better the cohort, the larger the leak: a section-only vector reflects deletions more. Papers are shielded
   by the title and abstract, which no deletion touches (leak +0.009 [−0.001, 0.019] at the default). Real omissions
   behave the same way (an omitting document resembles other omitters), which is consistent with the revision result.

A variant found by searching about eleven settings on the test bench (no title, no category filter, k 50, one
snapshot per entity, mean probability) reaches +0.081 [0.047, 0.113] on privacy single_type. It was selected on test
data and is reported as post hoc, not as a gate result.

## Decision (Parsa, 2026-10-21)

Report the privacy failure as a negative finding and proceed to Phase 7. The cohort settings stay as fixed in §0.2
#9 and #10 (no selection on test data); papers support the research question on both benchmarks.

## Options considered

| Option | Why not (now) |
|---|---|
| Select title weight, category filter, k and snapshot rule per pack on dev documents (within-document AUC needs no labels), then rerun the test bench once | Needs §0.2 #9 and #10 to become per-pack tunables; leaves the leak, so the privacy gate may still fail. Kept as an upgrade. |
| Adopt the post-hoc variant | Selected on the test bench. |
| Change the bench so the cohort uses the original document | Hides a real weakness: an uploaded document only has its own (omitting) text. |

## Consequences

- The report states the privacy result plainly: neighbourhood prevalence from content embeddings is no better than
  genre prevalence for privacy policies, and the leak is the main reason; the papers result supports conditioning on
  peers.
- Level 2 (the masked-assertion model, Phase 8) conditions on the present types rather than on an embedding of the
  whole text; whether it also suffers from the leak is a Phase 8 question. A cohort anchored on what an omission
  cannot change (site metadata, the title for papers) is another route.
- Benchmark artefacts recorded with the result: deleted targets end just under the presence threshold (median
  P(present) after the edit 0.32 privacy, 0.45 papers), which halves their omission score for both systems
  (absence-only recall@5 0.03-0.06); papers three_types has 468 instances, not 500.
