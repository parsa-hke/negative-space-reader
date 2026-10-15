# Decision record 001: the lean/full tier architecture

**Date:** 2026-10-12 · **Status:** adopted · **Plan sections changed:** §0.2 #6, #7, #8, #9, #14, #16, #18, #24, new #25; §0.4; §0.6; Phases 2–12.

This record is written so that it can be inserted into the class report (design section and GenAI/engineering trade-off discussion) with light editing.

## 1. Context

The original design (plan of October 2026, pre-revision) sized every component for a university GPU server: a 72-billion-parameter teacher model labelling thousands of sections, an 8-billion-parameter embedding model over roughly a million privacy-policy snapshots, a fine-tuned ModernBERT-large presence classifier, teacher-written explanations on every query, and a human-annotated benchmark of 150 documents per genre read by two annotators each.

Three constraints made this design fragile:

1. **GPU access is session-based.** The server is available in scheduled lab sessions, not continuously. The original pipeline had at least three GPU jobs with human work between them (schema pruning, hand-labelling), so each cycle of "run, inspect, fix, rerun" could take a week.
2. **Cost.** Running the teacher and the frontier-model baselines through an API instead would cost roughly $60–100 for the class deliverable and $300–750 for the paper's baselines (OpenRouter prices, 2026-10-12).
3. **Deployment.** A demo that embeds uploaded documents with an 8B model needs a rented GPU to stay online; the class demo and any later public demo would otherwise be limited to pre-computed examples.

The question was whether the project's contributions depend on the large models, or only its plumbing does.

## 2. Decision

Adopt a **two-tier architecture**. Every swappable component is a configuration key with a **lean** default and a **full** alternative, selected by one overlay (`configs/full.yaml`). The engine, the benchmark generator, the masked-assertion model and the pack format are identical in both tiers, so "lean versus full" is itself a reported ablation rather than a fork of the code.

| Component | Lean tier (default) | Full tier (`--overlay full`) |
|---|---|---|
| Corpus scope | 20–30k documents per genre, stratified by year and category, one snapshot per entity per year; full version chains kept | same (scope is not tiered) |
| Embedder | Qwen3-Embedding-0.6B (Apache-2.0, CPU-servable) | Qwen3-Embedding-8B |
| Late-interaction matcher | answerai-colbert-small-v1 (Apache-2.0) | Jina ColBERT v2 (CC-BY-NC) |
| Presence classifier | Per-type logistic heads on frozen section embeddings, trained on existing human labels (OPP-115 for privacy, ~23k segments) plus ~2k teacher-labelled sections for papers; optional SetFit-style contrastive fine-tune | ModernBERT-large fine-tuned on 5k teacher labels per genre |
| Explanations | Template filled from evidence (prevalence, quoted peer sentences, attributed present types); no LLM on the hot path | Teacher-written rationale for the top 5, opt-in |
| Real-omission benchmark | **OmitBench-revision:** natural omissions from version chains, validated on 50 documents per genre by two annotators | Blind annotation of 150 documents per genre by two annotators |
| Teacher route | Same model (Qwen2.5-72B-Instruct) reached three ways: a 7B local proxy for prompt iteration; the 72B on the university server for the one recorded extraction pass; the 72B via OpenRouter (provider pinned, responses cached) when no GPU is available | same |

## 3. Options considered

- **Keep the original design and schedule around the lab.** Rejected: each human-in-the-loop step would wait for the next session; one missed session delays the submission.
- **Move everything to a paid API.** Rejected on cost, and because labelling through an API introduces provider and quantisation drift unless pinned and cached.
- **Drop the presence classifier and use the LLM to judge presence per query.** Rejected: it puts the most expensive model on the hot path, which violates the design principle in §0.2 #6, and it makes the system's output depend on a closed model.
- **Shrink the models only.** Insufficient alone: the annotation effort and the lab-bound teacher loops remained. The revision benchmark and the teacher tiering were needed for the time saving.

## 4. Rationale

The project's claimed contributions are (a) the formalisation of conspicuous omission as a ranking task with a label-free deletion benchmark, (b) the masked-assertion model that conditions expectation on what the document already says, (c) the genre-pack abstraction, and (d) the measured gap between a corpus-conditioned system and frontier chat models. None of these depends on model size. The masked-assertion model is a four-layer transformer over about fifty type tokens and trains in minutes on a laptop. What the large models bought was marginal accuracy in presence detection and cohort retrieval, and those are precisely the components with a clean fallback path.

The revision benchmark is not only cheaper; it is a stronger experiment. Deletion benchmarks test whether the system can recover what was removed; the revision benchmark tests whether it can predict what authors themselves later added. The longitudinal privacy-policy corpus exists for exactly this kind of analysis, and arXiv version histories offer the same signal for papers. The known confound (additions driven by regulation rather than by peers) is handled by computing expectation against peers from the same time window, and its magnitude is measured and reported.

Evidence gathered before deciding (all 2026-10-12): Qwen2.5-7B-Instruct 4-bit generates at 24 tokens/s on the Apple M5 under MLX (about 10 s per section, suitable for iteration, not for the 10k-section extraction pass); OpenRouter serves Qwen2.5-72B-Instruct from one provider at $0.36/$0.40 per million input/output tokens with structured output supported; every model id in both tiers exists on the Hugging Face hub, and the lean tier is Apache-2.0/MIT throughout while the full tier includes two CC-BY-NC models.

## 5. Consequences

**Gained.** Class-deliverable API spend of about $10 (the frontier baseline only) instead of $60–100; the university GPU becomes an accelerator for one session rather than a dependency for three; the human annotation effort drops from 300 double-annotated documents to a 100-document validation; the demo can run on a free CPU host; the lean tier ships under permissive licences; the lean/full comparison becomes reportable evidence for the classifier-comparison learning outcome.

**Lost or risked.** Presence detection with frozen embeddings and linear heads is expected to score a few F1 points below a fine-tuned ModernBERT-large; the gate (macro-F1 ≥ 0.80 on the human set) decides, and the full-tier classifier is one overlay away. A 0.6B embedder will retrieve peers somewhat less precisely than an 8B one; the ablation measures it. Training data for the masked-assertion model shrinks from hundreds of thousands to tens of thousands of documents, which is adequate for a model of its size. The revision benchmark needs a validation step and an honest paragraph about regulatory confounds. Overnight laptop jobs need checkpointing and resumption.

**Unchanged.** The research question, the task definition, OmitBench-synthetic, the masked-assertion model and its attribution-based explanations, the pack builder, cohort retrieval, the ghost-paragraph interface, and the frontier-model comparison.

## 6. Upgrade path

`python -m experiments.run --config configs/<task>.yaml --overlay full --overlay gpu` rebuilds a pack at the full tier. Pack manifests record the tier and the model id behind every artefact, and the builder skips stages whose inputs and models are unchanged, so moving a pack from lean to full re-embeds the corpus and retrains the classifier but reuses the schema, splits, prototypes selection logic and benchmark instances. Every run logs its tier, and every teacher call logs which model build served it, so results from the two tiers are never mixed silently.
