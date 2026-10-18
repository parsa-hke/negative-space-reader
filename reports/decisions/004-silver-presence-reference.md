# Decision record 004: a silver presence reference instead of a hand-labelled sanity set

**Date:** 2026-10-17 · **Status:** adopted (lean tier) · **Plan sections changed:** §0.2 #15, §0.2 #25, Phase 4.2,
4.3, 4.5, 4.6, 4.8, Appendix B.

## Context

Presence detection (Phase 4) needs a reference: a set of sections whose present statement types are known, used to
measure the teacher's labels (4.2), calibrate and combine the presence models (4.3, 4.5), compare them (4.6), pass the
gate (4.8, macro-F1 ≥ 0.80) and, later, judge the benchmark's false-alarm rate (§0.2 #15). The plan made this a
hand-labelled sanity set: 300 sections per genre judged against the schema (50 types for research papers, 58 for
privacy policies), about 32,000 yes/no judgements.

Two things changed the picture.

1. **Effort.** The sections and a guided labelling page were ready, but at roughly two minutes per section the set
   would take the sole annotator 20 hours or more, which the project's schedule cannot absorb.
2. **The existing human labels are a weaker yardstick than assumed.** For privacy, OPP-115 (Wilson et al., 2016)
   already holds expert annotations, mapped to 35 of the 58 types through each type's `label_source`. A calibration
   on 100 sections showed the 72B teacher agreeing with these mapped labels at only micro-F1 0.53–0.55. The
   disagreement was concentrated in loosely mapped types, and reading the cases showed the teacher was usually right:
   OPP-115's attribute values often do not match the schema's definitions (log files with IP addresses are automatic
   usage collection; "companies that perform marketing services on our behalf" are service providers). Human labels
   of a different scheme, mapped mechanically, are not ground truth for this schema.

## Decision

In the lean tier the reference set is **silver**: the same 300 fixed sections per genre (`packs/<genre>/label_sets.json`)
labelled by a **second judge from another model family**, Claude Haiku 5.5 through OpenRouter, using exactly the
teacher's two-pass prompts (`label_presence` v1, `label_presence_check` v2). The teacher (Qwen2.5-72B-Instruct) is
measured against the judge, type by type. For privacy, OPP-115's human labels on the 35 mapped types are reported
alongside, both for the teacher and for the judge, which shows how each model relates to human annotation where it
exists. Hand labels on the guided page remain the **upgrade**: the reference is a configuration value
(`distil.score.reference`), so adding them later recomputes every number without code changes.

## Options considered

| Option | Annotator time | Why not (now) |
|---|---|---|
| Hand-label 300 sections per genre (the plan) | 20+ hours | Not feasible in the schedule. Kept as the upgrade. |
| Second judge, the annotator adjudicates only disagreements | 1–2 hours | Still annotator time this phase cannot spare; the cheapest human upgrade later. |
| Verify mode: the annotator confirms or rejects the teacher's labels | 3–4 hours | Anchors the annotator to the teacher, so recall would be overstated. |
| OPP-115 alone (privacy) | none | Loose mapping (above); covers 35 of 58 privacy types and no paper types. |
| **Second judge as silver reference** | none | Adopted. |

## Rationale

A second model from a different family makes errors that are partly independent of the teacher's, so agreement
between the two is evidence about label quality, and disagreement points at the types that need attention. It costs
about $0.40 for 600 sections. Keeping the same prompts makes the comparison about the models, not the instructions.
The lean/full structure of the project (decision record 001) already treats expensive inputs as upgrades behind
configuration; human labels fit that pattern.

## Consequences

- Every presence result states that it was measured against silver labels. Gate 4.8 becomes "macro-F1 ≥ 0.80 against
  the silver reference": a measure of agreement with an independent model, not of accuracy against people.
- **Shared blind spots are invisible.** Where both models make the same mistake, the reference agrees with the error.
  OPP-115 is the only human check (privacy, 35 types); research papers have none until hand labels are added. This
  goes into the limitations section.
- The labelling page, the fixed section sets and the scoring code (`distil.stage: score`) are unchanged and ready for
  the upgrade; the cheapest human upgrade is adjudicating only the teacher–judge disagreements.
- Watch item: the plan fits calibration (4.3) and the cascade's combination layer (4.5) on the same reference set it
  evaluates on (4.6, 4.8). A document-level split of the 300 sections into fitting and evaluation halves avoids
  optimistic numbers; to be decided before 4.3.
