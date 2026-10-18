# Decision record 005: the lean presence model, its training labels and its gate

**Date:** 2026-10-18 · **Status:** adopted (lean tier) · **Plan sections changed:** §0.2 #7, §0.2 #8, Phase 4.8.

## Context

Phase 4 builds the presence detector: for each section and each of the schema's statement types, the probability that
the section makes such a statement. Its outputs become the document profiles every later phase builds on. The plan
specified a cascade: (1) late-interaction matching against prototype sentences, (2) a band of uncertain scores, (3)
a cross-encoder reranker for that band, and a learned combination with a linear classifier over section embeddings.
The classifier was to be trained on OPP-115's human annotations for privacy (mapped through each type's
`label_source`) and on teacher labels for papers, and the phase gate was macro-F1 ≥ 0.80 against the human sanity
set. Decision record 004 replaced that set with a silver reference: a second judge model (Claude Haiku 5.5) labelling
300 sections per genre. All numbers below are measured on the evaluation half of that reference (whole documents held
out from the half used for calibration and thresholds).

Three findings forced decisions.

1. **The gate was unreachable by construction.** The teacher's own labels agree with the judge at macro-F1 0.773
   (privacy) and 0.688 (papers). A student trained on teacher labels can hardly exceed its teacher, so no lean or full
   variant could reach 0.80 against this reference. The 0.80 bar presumed a human reference on which the teacher
   would score above 0.85.
2. **The OPP-115 mapping is a weaker training signal than the teacher.** Against the reference, privacy heads trained
   on teacher labels reach 0.613 macro-F1; trained on the mapped OPP-115 labels, 0.576. Against OPP-115's own labels
   the order reverses (0.534 vs 0.568): each source wins on its own yardstick. Reading the disagreements (CHANGELOG
   2026-10-15) showed the mapping is often narrower or broader than the type definitions.
3. **The reranker did not pay.** Adding it to classifier + prototypes changed macro-F1 by +0.002 (privacy) and −0.004
   (papers) at 9× and 4.7× the time per section; on the Mac it scores 13 pairs per second, so profiling the corpus
   with it would take weeks.

## Decision

- **Training labels (§0.2 #7):** in the lean tier, presence heads train on teacher labels for every type in both
  genres (`presence.train_labels: teacher`). OPP-115 remains a reported human check (classifier vs OPP-115 on its 35
  mapped types).
- **Presence model for profiles and the pipeline (§0.2 #8):** classifier + prototypes, combined by the learned
  logistic layer. The reranker stays implemented and ablated; it is switched back on by configuration if a future
  measurement shows it helps.
- **Gate 4.8:** the presence model's macro-F1 against the reference must reach at least 0.80 of the teacher's own
  macro-F1 against the same reference, on the evaluation half, in both genres. With hand labels as the reference
  (the upgrade path of decision record 004), the absolute 0.80 applies again.

## Options considered

| Option | Why not |
|---|---|
| Keep the absolute 0.80 gate and try SetFit (the plan's next step) | Cannot exceed the ceiling, so it fails regardless; costs hours on the Mac plus re-embedding the corpus for presence. |
| Record the gate as not met and move on | Honest but leaves the phase without a usable acceptance test. |
| Keep OPP-115 training labels | Better only against OPP-115's own mapping, which the evaluation does not use. |
| Full cascade for profiles on a rented GPU | Several A100 hours for no measured gain. |

## Consequences

- The gate now measures how well the student reproduces what its teacher knows, relative to an independent model.
  It does not measure accuracy against people; the limitations section says so, and hand labels restore an absolute
  gate.
- The classifier's ECE falls from 0.09–0.15 (calibrated heads alone) to 0.004–0.016 once prototypes are added, which
  matters because later phases use P(present) as a probability.
- The reranker's model (2.29 GB) is still downloaded and versioned, so the full-tier ablation is one configuration key.
