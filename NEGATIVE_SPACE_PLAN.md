# Negative-Space Reader: End-to-End Build Plan

Reading what a document conspicuously doesn't say · MAAI 902 Assessment 2 (NLP Solution Design) and beyond · Parsa Hosseini · UOWD · October 2026

This file is the working plan for the project and is written to be executed with Claude Code, one phase per session. Each task has a definition of done. Tasks marked **GATE** must be done and checked before the next phase starts. Decisions recorded in §0.2 are fixed; change them only by editing this file first and saying so in CHANGELOG.md.

The plan is written for the full project (class submission, research paper, product-grade system). Phases 1–7 produce the class deliverable; Phases 8–12 produce the paper and the product layer. No phase is throwaway: every artefact of the class version is reused by the later ones.

---

## 0. Project definition

### 0.1 Research question

Can a system detect what a document *conspicuously omits* by modelling what its genre peers assert, and does conditioning that expectation on the document's own content beat both genre-level prevalence and frontier chat models at the task?

Three sub-questions:

1. **Formalisation.** Can "conspicuous omission" be defined as a ranking task with a label-free evaluation (controlled deletion) that correlates with human judgement of real omissions?
2. **Conditional expectation.** Does a model that predicts P(assertion type *t* is present | everything else the document asserts, its embedding, its metadata) outperform neighbourhood prevalence P(*t* | nearest peers)?
3. **The LLM gap.** Language models are trained to predict what is present. Measured on the same benchmark, how far behind a corpus-conditioned system are frontier chat models at reasoning about absence, in open-ended and candidate-list settings?

### 0.2 Fixed decisions

Revised **2026-10-12** to the lean/full tier design (CHANGELOG; `reports/decisions/001-lean-architecture.md`). The research question, task, benchmark, MAM and pack abstraction are unchanged; the models plugged into the pipeline shrank to fit the Mac and a free demo, and the real-omission benchmark changed to one the corpus labels itself. Every lean choice has a full-tier upgrade behind one config overlay (#25).

| # | Decision | Choice |
|---|----------|--------|
| 1 | Task | **Conspicuous-omission detection.** Input: a document *d* and a genre corpus *C*. Output: a ranked list of assertion types absent from *d*, each with a score, evidence from peers, and an explanation. Formally, rank types *t* ∉ present(*d*) by score(*t*) = P(*t* expected | *d*, *C*) × (1 − P(*t* present in *d*)) × w(*t*), where w is an optional genre severity weight (default 1). |
| 2 | Genres in v1 | **Two, equal priority, built from the same pipeline:** (A) privacy policies; (B) research papers (arXiv cs.CL and cs.LG). A third genre is a stretch goal for the paper (§0.2 #22). |
| 3 | Genre-pack abstraction | A genre is a **pack**: `{corpus, schema, classifier, prototypes, index, config}`. The engine contains **no genre-specific code**. Anything that only works for one genre is a bug in the engine, not a special case. Adding a genre = running the pack builder, not writing code. |
| 4 | Unit of analysis | **Section-level.** Documents are segmented into sections; sections into sentences. Presence is detected at section level and aggregated to document level. Long documents are "more rows", never a different code path. |
| 5 | Assertion schema | **Frozen, per genre, 40–60 types.** Induced once: teacher LLM extracts open assertions from a stratified sample of ~500 peers → embed → cluster (HDBSCAN over embeddings) → teacher names clusters → human prune and merge to 40–60 types with a one-line definition, two positive examples and one near-miss each → **frozen**. Seeds: OPP-115's ten data-practice categories and their attributes (genre A); the NeurIPS paper checklist and ACL responsible-NLP checklist (genre B). Schema changes after freezing require a new pack version. |
| 6 | Teacher model | **Qwen2.5-72B-Instruct** (decided 2026-10-12 over Llama-3.3-70B: ungated, official AWQ build `Qwen/Qwen2.5-72B-Instruct-AWQ` fits one 80 GB card, same family and chat template as the small stand-ins). Structured output via **xgrammar** (fallback Outlines). Fallback if 72B is not servable: Qwen2.5-32B-Instruct. **Three ways to run it, same `Teacher` interface, chosen by compute overlay, never by code:** (i) `local.yaml`: a small Qwen (Qwen2.5-7B-Instruct 4-bit via MLX, or 1.5B via transformers) on the Mac for prompt iteration, the coverage check and smoke; (ii) `gpu.yaml`: the 72B on the UOWD server with vLLM (bf16 with tensor parallelism on ≥ 4×80 GB — corrected 2026-10-12 from "≥ 2×80 GB": 145 GB of bf16 weights leave no KV cache on 2×80 GB — AWQ on 1×80 GB or ≥ 2×48 GB, 32B-AWQ below 48 GB; `./lab_session.sh setup` applies this rule) for the final schema-induction pass and the genre-B labels; (iii) `openrouter.yaml`: `qwen/qwen-2.5-72b-instruct` through OpenRouter's OpenAI-compatible API, provider and quantisation pinned, every response cached to disk, key from `OPENROUTER_API_KEY`. The teacher is used for schema induction, a small label set, and optional explanations. It is **never on the per-query hot path**. **Every measured result states which tier produced its labels.** **Lab reality (2026-10-12, decision record 002):** the UOWD machine is Windows with 4 × RTX 5000 Ada 32 GB, where vLLM cannot run and a 72B is too slow for 15k calls in a day; the recorded schema-induction pass therefore uses the named fallback, **Qwen2.5-32B-Instruct Q4_K_M served by llama.cpp, one server per GPU**, and the prompt is confirmed on that model. The 72B via OpenRouter remains the fallback if the 32B fails the eye test. |
| 7 | Presence classifier | **Lean tier (default):** the pack's frozen embedder (#9) plus **one calibrated logistic head per type** over section embeddings, trained on teacher labels: a teacher-labelled set of ~2k sections per genre (one `gpu` or `openrouter` run; changed 2026-10-18 for genre A, decision record 005: OPP-115's human annotations mapped through `label_source` (3,792 segments of 115 policies) agree with the schema's definitions less well than the teacher does, so they are kept as a reported human check, not as training labels; `presence.train_labels: plan` restores the old rule). Optional **SetFit-style contrastive fine-tune** of the small embedder on the same labels, Mac-sized. **Full tier (`--overlay full`):** **ModernBERT-large** (fallback DeBERTa-v3-large) multi-label fine-tune on 5k teacher labels per genre, as originally planned; used if the lean head misses gate 4.8, or as an ablation. All variants: temperature-scaled on a held-out split so P(present) is a probability; one classifier per pack; selected by `presence.head: linear | setfit | finetune`. |
| 8 | Presence cascade | For each type *t* and each target section: (1) **prototype matching**: max similarity between the section's sentences and *t*'s prototype set using late interaction (lean: **answerai-colbert-small-v1**, Apache-2.0; full: **Jina ColBERT v2**, CC-BY-NC; fallback dense cosine with the pack embedder); (2) clear hit or clear miss stops here; (3) **borderline band** (thresholds fixed in Phase 4 from the distributions) goes to a **reranker** (**bge-reranker-v2-m3**, upgrade candidate Qwen3-Reranker) with the type definition as query. Classifier probability (#7) and cascade score are combined by a learned logistic layer. Band thresholds are fixed per pack and ablated. **Lean tier (2026-10-18, decision record 005):** the reranker is implemented and ablated, but the corpus profiles (4.7) and the pipeline use steps 1–2 plus the classifier ("classifier + prototypes"): in 4.6 the reranker added +0.002 / −0.004 macro-F1 at 5–9× the time; `presence.cascade.reranker: true` restores step 3. |
| 9 | Document and section embeddings | **Lean tier (default): Qwen3-Embedding-0.6B** (fallback bge-m3), both Apache/MIT, CPU-servable, so the index built with it can back a free demo. **Full tier: Qwen3-Embedding-8B** (fallback NV-Embed-v2, CC-BY-NC), run as the "does a big embedder help?" ablation when a GPU session allows. Section embeddings for cohort retrieval and schema induction; document embedding = mean of section embeddings plus a title/abstract embedding. The app must embed uploads with the **same model as the index**; the pack manifest records it. |
| 10 | Cohort (peer) retrieval | k nearest documents by document embedding, **k = 200 by default**, restricted to the same pack and, where metadata exists, the same category (arXiv primary category; app/website category for policies when available). ANN index: **Qdrant** (HNSW) in production, **FAISS** in development. Both behind one `Index` interface. |
| 11 | Expectation model, level 1 | **Neighbourhood prevalence.** Each indexed document has a profile: a bitvector over the schema from the presence cascade. Expectation for *d* = mean of the k cohort bitvectors. This is the baseline and the Week-6 model. |
| 12 | Expectation model, level 2 | **Masked-assertion model (MAM).** A set-transformer over the document's present-type tokens plus its embedding and metadata tokens, trained with random masking of present types to predict the masked ones (binary cross-entropy per type). At inference, P(*t* | present(*d*) \ {*t*}, emb, meta) for every absent *t*. Trained on the whole indexed corpus; no labels needed. This is the paper's method contribution. |
| 13 | Ranking | score(*t*) as in #1 with P(expected) from level 2 (level 1 when level 2 is unavailable). Ties broken by prevalence. Output the top 10 by default, all above a threshold on request. |
| 14 | Evidence and explanation | Every reported absence carries: the prevalence among cohort peers (count and percentage), up to five peer sentences that instantiate *t*, with document ids (retrieval changed 2026-10-19, recorded here 2026-10-20: each peer's best sentence for *t* under the full presence model, precomputed for the indexed corpus by stage `sentence_evidence`, instead of the prototype index, whose MaxSim selection quoted wrong sentences; CHANGELOG 2026-10-19), and, for the MAM, the three present types that most raised P(*t*) (gradient × input attribution over the type tokens). **The explanation is a template filled from this evidence** ("*N*% of peers like this one state *X*; this document does not. Peers typically say: …"): no LLM on the hot path, nothing that is not a quote. `explain.mode: teacher` adds a teacher-written one-line rationale and suggested wording for the top-N (N = 5), synthesised from the peer sentences only; in the UI it is **opt-in and flagged** when it sends the document off the machine. |
| 15 | Evaluation, label-free | **OmitBench-synthetic.** From held-out documents, delete all sections or sentences instantiating a sampled type *t*; the system must rank *t* highly. Metrics: recall@1/5/10, MRR, nDCG@10; **false-alarm rate** on untouched documents (fraction of top-5 absences that are in fact present, judged by the presence reference set of task 4.2: silver labels from a second judge model in the lean tier, hand labels as the upgrade; changed 2026-10-17 from "the human-labelled set", decision record 004); calibration (ECE) of P(expected). Three deletion regimes: single type, three types, and "delete a whole section". |
| 16 | Evaluation, real omissions | **OmitBench-revision (primary).** Natural omissions from version chains: if version *n*+1 of a document adds a type *t* that its cohort peers *at the time of version n* already stated, then version *n* conspicuously omitted *t*. Genre A from the longitudinal policy corpus; genre B from arXiv v1→v2 pairs where both versions are available (verify in Phase 2). Expectation is always computed against a time-windowed cohort (#19) to separate peer norms from law changes. **Human validation:** 50 documents per genre, two annotators, judging whether each revision-added type was conspicuously missing; agreement (Krippendorff's α) and precision of the revision labels reported. The original 150 × 2 blind annotation remains the full-tier option (`bench.real.source: annotation`) if validation precision is below 0.7. Evaluation only, never training. |
| 17 | Baselines | (a) genre-uniform prevalence (no cohort); (b) level-1 cohort prevalence; (c) level-2 MAM; (d) frontier chat model, open-ended ("what is missing?"), parsed to schema types by the teacher; (e) frontier chat model, candidate-list (given the 40–60 type names, which are missing?); (f) a chat model given the same cohort peers in context (RAG baseline, bounded by context length). Models for (d)–(f): GPT-4-class and Claude-class APIs plus the open teacher, all with fixed prompts recorded in the repo. |
| 18 | Corpora | **Sources unchanged. Genre A:** Princeton–Leuven Longitudinal Corpus of Privacy Policies (Amos et al., 2021; ~1M snapshots) plus OPP-115 (Wilson et al., 2016) for presence labels and the sanity set. **Genre B:** unarXive (verify release and licence) restricted to cs.CL and cs.LG, 2018 onward; arXiv metadata for categories, dates and version history. **Scope:** **20–30k documents per genre**, stratified by year and category, **one snapshot per entity per year** (Appendix B: a million snapshots are mostly the same policies repeated), with the full version chain kept for #16 and #19. Lean/full differ in model size, not in corpus; `corpus.max_documents_per_genre` is the only knob. Record every URL, licence and citation in `data/README.md` before use. |
| 19 | Time axis | Each indexed document carries a date. The cohort can be restricted to a date window, which makes expectation drift measurable. Events of interest: GDPR enforcement (May 2018) for genre A; ACL's mandatory limitations section (2022) for genre B. |
| 20 | On-the-fly mode | For a document from a genre with no pack: detect the nearest pack by embedding; if the similarity is below a threshold, retrieve a cohort from the general index and induce a temporary schema from that cohort alone with the teacher. Slower (minutes), uncalibrated, clearly labelled as such in the UI. Demo feature and discovery tool, not a measured system. |
| 21 | Ethics stance | The system reports "**N% of peers like this one state X; this document does not**", never "non-compliant" or "deficient". Every claim is backed by visible peer evidence. Genre norms encode what dominant actors do; the UI shows prevalence, not verdicts. Uploaded documents are processed locally and never stored. The limitations section of the paper is written before the results section. |
| 22 | Stretch genres (paper only) | SEC 10-K risk-factor sections, clinical-trial registrations (ClinicalTrials.gov), open-source READMEs. Each is a pack-builder run; none is required for the class. |
| 23 | Engineering | Python 3.11, PyTorch, Hugging Face `transformers`, `sentence-transformers`, vLLM, Qdrant/FAISS, single YAML config with inheritance and overlays, one entry point `experiments/run.py`, fixed seeds, every run logged to `runs/log.csv`, 10-minute smoke mode, a test per module. Demo: **Gradio** for the class; **FastAPI + a small web front-end** for the product layer. |
| 24 | Class submission scope | Phases 1–7 complete on the **lean tier**: both packs built; level-1 expectation; cascade presence detection; OmitBench-synthetic on both genres; OmitBench-revision constructed (validation in Phase 9); the frontier-model comparison on the candidate-list setting only; Gradio demo with the ghost-paragraph view and the cohort evidence panel, hostable on a CPU Space; report. |
| 25 | Tiers and the upgrade path | Every swappable component (teacher route, embedder, late-interaction model, presence head, explanation mode, real-omission source, presence reference set (silver | human; added 2026-10-17)) is a **config key, not code**. `configs/base.yaml` holds the **lean** defaults (runs on the Mac overnight, API spend ≈ $10 for the class); `configs/full.yaml` is one overlay that flips every key to the **full** tier (the 2026-10-12 pre-revision design). Pack manifests record the tier and model ids of every artefact; `packs/builder.py` rebuilds only the artefacts whose inputs changed. Upgrading = `--overlay full` plus one GPU session; the engine, the benchmark and the MAM are identical in both tiers, so lean-vs-full is itself a reported ablation. Decision record: `reports/decisions/001-lean-architecture.md`. |

### 0.3 Deliverables

**Class (MAAI 902 Assessment 2, individual, 20%, Week 6):**
- Technical report, 3,000–4,000 words, covering the seven brief items (§11.1 maps them).
- Working prototype (Gradio) and a demonstration.
- GitHub repository with README and commented code.

**Paper (target: an ARR cycle, or EMNLP/NAACL main track; datasets-and-benchmarks track as the fallback):**
- OmitBench (synthetic + real) for two to three genres, released with the schemas.
- The MAM method and the frontier-model gap result.

**Product (after the paper has a result):**
- API and web UI; genre packs as the unit of sale; on-prem deployment for confidential documents; continuous monitoring of a document set against its drifting cohort.

### 0.4 Repository layout

```
negspace/
  CLAUDE.md                     # instructions for Claude Code (see §0.5)
  NEGATIVE_SPACE_PLAN.md        # this file
  CHANGELOG.md
  README.md
  pyproject.toml / requirements.txt
  configs/
    base.yaml                   # every tunable
    data.yaml                   # Phase 2 data build (task build_data)
    smoke.yaml                  # 200 documents per genre, tiny models
    pack_privacy.yaml           # genre A pack build
    pack_papers.yaml            # genre B pack build
    induce.yaml                 # schema induction
    distil.yaml                 # classifier distillation
    mam.yaml                    # masked-assertion model
    bench.yaml                  # OmitBench runs
    baselines.yaml              # LLM baselines (API keys from env)
    gpu.yaml / local.yaml       # compute overlays (paths, batch sizes, teacher route)
    openrouter.yaml             # teacher via OpenRouter (provider pinned, responses cached)
    full.yaml                   # the full tier: flips every lean default to the large model (§0.2 #25)
  data/
    raw/                        # downloaded corpora, untouched (gitignored)
    processed/                  # jsonl per genre: documents, sections, sentences
    README.md                   # sources, URLs, licences, actual field names
    schema.py                   # Document / Section / Sentence / Assertion dataclasses
    loaders/
      privacy_pl.py             # Princeton–Leuven corpus
      opp115.py
      unarxive.py
      arxiv_versions.py         # version-specific arXiv e-prints -> sections (revision pairs, genre B)
    segment.py                  # section + sentence segmentation, boilerplate removal
    dedup.py                    # near-duplicate removal (MinHash) and version collapsing
    splits.py                   # document-level splits, disjoint by source entity
    revisions.py                # version chains -> (version n, version n+1) pairs for OmitBench-revision
    build.py                    # CLI: raw → processed jsonl + stats
  packs/
    spec.py                     # GenrePack dataclass; load/save; versioning
    schema_induction.py         # teacher extraction → clustering → naming → freeze
    prototypes.py               # prototype selection per type
    profiles.py                 # bitvector profile per indexed document
    builder.py                  # CLI: build a pack end to end from a config
    privacy/                    # schema.yaml, prototypes/, classifier/, index/  (artefacts gitignored except schema.yaml)
    papers/
  engine/
    embed.py                    # embedding models behind one interface
    index.py                    # Index interface: FAISS and Qdrant backends
    cohort.py                   # peer retrieval with category and date filters
    presence.py                 # cascade: prototypes → band → reranker; combine with classifier
    classifier.py               # ModernBERT multi-label student; calibration
    expectation.py              # level 1 (prevalence) and level 2 (MAM inference)
    rank.py                     # scoring, thresholds, top-N
    evidence.py                 # peer sentences, attribution, counts
    explain.py                  # teacher rationale for top-N
    pipeline.py                 # analyse(document, pack) → Report
    onthefly.py                 # pack detection and temporary schema
  models/
    teacher.py                  # vLLM client, structured output, prompt registry
    mam.py                      # masked-assertion set-transformer
    reranker.py
  experiments/
    run.py                      # single entry point
    build_pack.py
    induce.py
    distil.py
    train_mam.py
    bench.py                    # OmitBench synthetic + real
    baselines.py                # frontier-model runs
    drift.py                    # time-window experiments
  bench/
    synth.py                    # controlled deletion generator
    revision.py                 # OmitBench-revision: revision-added types vs time-windowed cohort
    real/                       # annotation guidelines, raw annotations, adjudication
    metrics.py                  # recall@k, MRR, nDCG, false-alarm, ECE, bootstrap CIs
  app/
    gradio_app.py               # class demo
    api.py                      # FastAPI (product layer)
    web/                        # front-end (product layer)
  analysis/
    tables.py                   # every table in the report from runs/
    figures.py
    error_analysis.py
  prompts/                      # every teacher/baseline prompt, versioned, with a hash
  tests/
  reports/
    figures/ tables/
    report.md                   # class report draft
    decisions/                  # decision records, written for insertion into the report
    paper/                      # LaTeX, later
  runs/                         # metrics.json per run + log.csv (gitignored except log.csv)
```

### 0.5 CLAUDE.md (put this at the repo root)

```
# Project: Negative-Space Reader (MAAI 902 Assessment 2 → paper → product)
Read NEGATIVE_SPACE_PLAN.md before any task. Work one phase at a time.
Rules:
- Never change a decision in §0.2 without editing the plan first and saying so.
- The engine has no genre-specific code. A genre is a pack (config + data). If a step only works for one genre, fix the engine.
- Every new module gets a test in tests/ that runs in under 10 s with tiny models.
- Every experiment is launched through experiments/run.py with a config in configs/.
- Every run writes runs/<name>/metrics.json and one line to runs/log.csv.
- Smoke mode (configs/smoke.yaml: 200 documents per genre, small models) must pass before any full run.
- Every teacher/baseline prompt lives in prompts/ with a version and hash; runs record the hash.
- Do not invent corpus field names or model checkpoint names: inspect the raw files and the hub, and write what is actually there.
- Schemas are frozen after Phase 3; a change is a new pack version, recorded in CHANGELOG.
- Plots go in reports/figures/, tables in reports/tables/, both regenerated by scripts.
- Keep a running CHANGELOG.md: date, what was done, what was found, including negative findings.
- Never add Claude as author or co-author of commits: no `Co-Authored-By: Claude` trailer, no "Generated with Claude Code" line, in commit messages or PR descriptions. Commits are authored by the user only.
```

### 0.6 Compute

To be recorded in Phase 1 with `python -m experiments.run --device-info`.

- **Local (Apple M5, 24 GB unified, MPS): the primary machine in the lean tier.** Development, tests, data build, segmentation, dedup, corpus embeddings with the 0.6B embedder (overnight), presence heads, prototypes, index, profiles, MAM training, OmitBench, demo. Teacher iteration with Qwen2.5-7B-Instruct 4-bit under MLX (**measured 2026-10-12: 24 tokens/s, 1.3 s load**; ≈ 10 s per section, so fine for the 20-section eye test, ≈ 30 h for the 10k-section extraction pass, which therefore goes to the 72B).
- **University GPU server: an accelerator, not a dependency.** One session covers the final schema-induction pass and the genre-B labels with the 72B teacher (vLLM), and, if wanted, the full-tier ablations (8B embedder, ModernBERT-large). Model, count and VRAM to be recorded; decides the teacher precision (70B needs ~140 GB in bf16, ~40 GB with AWQ; 32B fits one 80 GB card in bf16).
- **OpenRouter:** the same 72B teacher through an API, ≈ $5–15 for the class; lets the interactive teacher loops run from home when no GPU is available (§0.2 #6 route iii).
- **Google Colab Pro (A100) — the primary GPU from Phase 5 on (changed 2026-10-18; Parsa has a Pro subscription with units to spare; replaces "free tier, optional, no Pro units planned"):** every GPU-shaped job (corpus embeddings, presence over new documents, profiles, the full-tier ablations: 8B embedder, SetFit / ModernBERT-large) runs on Colab through one runner notebook (`notebooks/colab_runner.ipynb`): it clones the private GitHub repo, pulls data and artefacts from a private Hugging Face dataset repo, runs `experiments/run.py` with `--overlay colab`, and pushes the outputs back; the Mac pulls them. Tests and smoke stay on the Mac. **Hugging Face Pro:** the private dataset repo above, and a ZeroGPU Space as the demo host option (7.1). Earlier free-tier use: the 2026-10-17 paper re-embed (A100, 140 s).
- Same code on all three; `configs/gpu.yaml` and `configs/local.yaml` overlay paths, batch sizes and model sizes. Every long job checkpoints and resumes.

**Recorded in Phase 1.3 (2026-10-12):**

| Target | Measured / stated | Role |
|---|---|---|
| Local | `--device-info`: Apple M5, 10 cores, 24.0 GiB unified (17.8 GiB usable by MPS), macOS 27.0.1, Python 3.11.17, torch 2.13.0, MPS yes, CUDA no | dev, tests, data build, smoke; teacher stand-in Qwen2.5-1.5B-Instruct (transformers + xgrammar on MPS) |
| UOWD lab machine | `--device-info` 2026-10-12: **Windows Server 2022, 4 × NVIDIA RTX 5000 Ada Generation, 32 GiB each (128 GiB, sm_8.9)**, 112 CPU cores, 1023 GiB RAM, driver 581.42, torch 2.13.0+cu126; no git, no admin; day sessions only | teacher via llama.cpp (32B, one server per GPU), corpus embeddings (decision record 002) |
| Colab Pro | not measured; nominal A100 40 GB or L4 24 GB, allocation varies per session (run `--device-info` in each session). **Runs Colab's own Python and scientific stack** (Python 3.13 with Colab's numpy/pandas/pyarrow/torch on 2026-10-19), with the project's model libraries (transformers, sentence-transformers, ...) pinned on top by `requirements-colab.txt`; the package is not installed there (pyproject pins the Mac's 3.11). The smoke comparison against the Mac checks that this changes no result. | fallback for distillation and MAM training; too small for the 72B teacher (L4: 32B-AWQ at most) |

Teacher size (§0.2 #6): Qwen2.5-72B-Instruct, served as AWQ (`configs/gpu.yaml` default) unless the server has ≥ 4×80 GB, in which case bf16 with tensor parallelism (`./lab_session.sh setup` picks automatically). Confirm against the server row above before the lab session.

**What runs where (lean tier):**

| Step | Mac | GPU session or OpenRouter |
|---|---|---|
| Phase 2 data build, dedup, splits, revision pairs | all | |
| 3.1 prompt eye test, 3.5 coverage check | 7B iterations | confirm on the 72B |
| 3.2 claim extraction, 500 docs/genre | | 72B, one pass (≈ $13 on OpenRouter, free on the server) |
| 3.3 clustering | all | |
| 4.1 presence labels | genre A from OPP-115 (no teacher) | genre B: ~2k sections with the 72B (≈ $2) |
| 4.3 presence heads, 4.4 prototypes, 4.5 band, 4.7 profiles | all (0.6B embedder, overnight) | full-tier ablations if time |
| Phase 5 index, pipeline | all | |
| Phase 6 OmitBench-synthetic and -revision | all | |
| 7.3 frontier comparison | | one frontier model via OpenRouter batch (≈ $10) |
| Phase 8 MAM | all (minutes per seed) | |

### 0.7 Glossary

- **Assertion type**: one entry of a pack's schema, e.g. `retention_period`, `third_party_recipients`, `limitations_stated`, `random_seeds_reported`.
- **Present set** present(*d*): the assertion types detected in *d* above the calibrated threshold.
- **Profile**: the bitvector (or probability vector) of present types for an indexed document.
- **Cohort**: the k nearest indexed documents to *d* under the pack's filters.
- **Prevalence** prev(*t* | cohort): the fraction of cohort documents with *t* present.
- **Conspicuous omission**: an absent type with high expectation; the system's output.
- **Ghost paragraph**: the UI rendering of an omission at the position where peers usually place it.

---

## Phase 1 — Environment and skeleton

- [x] 1.1 Create the repo `negspace` with the layout in §0.4. `.gitignore` for `data/raw`, `data/processed`, `packs/*/{prototypes,classifier,index}`, `runs/*` except `log.csv`, checkpoints, `.venv`, `.env`.
- [x] 1.2 Python 3.11 environment. Pin: `torch`, `transformers`, `sentence-transformers`, `vllm` (GPU only; optional extra), `xgrammar`, `outlines`, `qdrant-client`, `faiss-cpu`, `scikit-learn`, `hdbscan` (or `scikit-learn`'s HDBSCAN), `datasketch`, `pandas`, `numpy`, `pyyaml`, `pydantic`, `gradio`, `fastapi`, `uvicorn`, `pytest`, `tqdm`, `matplotlib`. Optional: `wandb`.
- [x] 1.3 `python -m experiments.run --device-info`: print devices, VRAM, CUDA, torch. Record all three compute targets in README and §0.6. **Decide the teacher size (§0.2 #6) and record it.** *(Local recorded; university-server row in §0.6 pending its `--device-info` output.)*
- [x] 1.4 `configs/base.yaml` with every tunable: paths, model ids (teacher, embedder, late-interaction, reranker, student), k for cohorts, schema size bounds, band thresholds (null until Phase 4), MAM hyper-parameters, bench settings, seeds. Overlays: `smoke`, `gpu`, `local`.
- [x] 1.5 `experiments/run.py`: loads config with `inherit:` and `--overlay`, `--set key=value`, sets all seeds, dispatches to `build_data | build_pack | induce | distil | train_mam | bench | baselines | drift | serve` (`build_data` added in Phase 2), writes `runs/<name>_s<seed>/{config.yaml,metrics.json}` and a `log.csv` line for every run including failed ones.
- [x] 1.6 `models/teacher.py`: vLLM client with a `generate_structured(prompt, pydantic_model)` call; an OpenAI-compatible client so the same code can hit an API model for baselines; prompt registry reading `prompts/*.md` with a version header and a content hash. Smoke stand-in: a 1.5B instruct model locally.
- [x] 1.7 `tests/test_skeleton.py` imports every module; `tests/test_teacher.py` round-trips a structured call against the stand-in. **GATE: `pytest` green.**

---

## Phase 2 — Corpora and the unified schema

- [x] 2.1 Download the Princeton–Leuven corpus, OPP-115 and unarXive (plus arXiv metadata **including version history**). Record URLs, licences, versions, sizes and citations in `data/README.md`. If a corpus requires a request form, file it on day one and use the public sample meanwhile. Record whether arXiv v1/v2 full texts are obtainable for genre B (needed by §0.2 #16); if not, OmitBench-revision for genre B uses the metadata-level signal (added sections by heading) or is dropped with the reason logged.
- [x] 2.2 Inspect each corpus by hand (open 5 documents each). Document the actual field names: how documents, versions, dates and categories are stored; how section structure is represented (unarXive has it; policies need segmentation); character encodings; boilerplate patterns; **how versions of the same entity are linked**. Paste the real field names into `data/README.md`. **Map every OPP-115 category and attribute to its segment-level annotation format** (these become genre A's presence labels in 4.1).
- [x] 2.3 `data/schema.py`:
  ```
  Document: id, genre, source, entity_id (website / arXiv id), version, date, category, title, meta: {}
  Section:  doc_id, idx, heading, text, char_span
  Sentence: doc_id, section_idx, idx, text
  Assertion (Phase 3+): doc_id, section_idx, type_id, prob, evidence_sentence_ids
  ```
- [x] 2.4 One loader per corpus → `Document` + `Section`s. Each loader has a test that round-trips three documents and writes an HTML preview with section boundaries drawn (`reports/figures/loader_check_<source>.html`).
- [x] 2.5 `data/segment.py`: heading-aware section segmentation for policies (rule-based on heading patterns plus a fallback on paragraph breaks), sentence segmentation (**rule-based with an abbreviation list**: pysbd measured at 333 sentences/s, too slow for ~3M sentences; spaCy's sentencizer splits after "Sec.", "Fig.", "et al.,"; CHANGELOG 2026-10-12), boilerplate removal (cookie banners, navigation, LaTeX residue). Unit tests on five hand-checked documents per genre.
- [x] 2.6 `data/dedup.py`: MinHash near-duplicate detection across documents; collapse policy versions into one entity with a version chain; drop exact duplicates. **Sample the indexable corpus to `corpus.max_documents_per_genre` (20–30k) stratified by year and category, one snapshot per entity per year; keep the full version chain of every sampled entity.** Report duplicate rates and the sampling table per genre in `reports/tables/dedup.csv` and `reports/tables/corpus_sample.csv`.
- [x] 2.6b `data/revisions.py`: from the version chains, emit (version *n*, version *n*+1, date_n, date_n+1) pairs where the two texts differ materially (not boilerplate); genre B from arXiv version pairs if 2.1 found them. Manifest in `data/processed/<genre>/revisions.jsonl`; counts per year in `reports/tables/revisions.csv`.
- [x] 2.7 `data/splits.py`: splits disjoint by entity (website for policies; arXiv id, with author overlap ignored, for papers). Train (indexable corpus) / dev (schema induction samples, calibration) / test (OmitBench documents, never indexed). **Every entity with a revision pair used in OmitBench-revision is in test, and no version of it is indexed.** Manifests in `data/processed/splits/`.
- [x] 2.8 `data/build.py` → `data/processed/<genre>/{documents,sections,sentences}.jsonl` and `reports/tables/corpus_stats.csv`: documents, sections per document, sentences per section, date range, category distribution, per genre.
- [x] 2.9 **GATE: both genres build end to end in smoke mode (200 documents each) and at the 20–30k scope on the Mac; stats, sampling and revision tables exist; the test split is sealed (hash recorded in CHANGELOG).** *(Passed 2026-10-12. Genre B is 17.6k documents, below the scope: the open subset holds only 18k permissively licensed cs.CL/cs.LG papers from 2018 on; recorded as a negative finding, upgrade = unarXive 2024.)*

---

## Phase 3 — Schema induction (the teacher's main job)

The schema is the backbone of everything downstream. It is induced, pruned, frozen, and versioned. Iterate with the small local teacher (`--overlay local`); run the one extraction pass with the 72B (`--overlay gpu` in the lab session, or `--overlay openrouter` from home). Every artefact records which teacher tier produced it.

- [x] 3.1 `prompts/induce_extract.md`: given a section, return a JSON list of assertions, each `{claim: str, type_guess: str, span: str}`, where `claim` is a normalised one-sentence paraphrase of what the section asserts. Test on 20 sections per genre by eye with the local small teacher; iterate the prompt until the extraction is faithful; **confirm the final version on the 72B** on the same 20 sections (the small model is for iteration, not for the record). Record the final version hash. *(2026-10-12: iterated v1→v3 on the 7B; v2 `a756e28a41a1` kept, v3 rejected. Confirmed on the recorded teacher, the 32B (decision 002), run `eye_v2_32b_s0`: spans 95.3% / 97.8% verbatim, 0 failed; full by-eye review of all 40 sections against Appendix D on 2026-10-13: passed with three recorded residuals (related-work claims misattributed and over-split, off-genre boilerplate, over-split proofs); accepted by Parsa. Ticked 2026-10-13.)*
- [x] 3.2 `packs/schema_induction.py` step 1: run the 72B teacher over a stratified sample of 500 dev documents per genre (stratify by date and category); responses cached so a rerun is free. Store every extracted claim with its source section and the teacher tier. *(2026-10-12, lab, 32B per decision 002: privacy 500/500 documents, 21,046 claims; papers 303 complete + 1 partial of 500, 44,579 claims, stopped at the session deadline. 2026-10-13: Parsa accepted the 303 complete papers as the genre-B sample; the partial paper `unarxive:2110.04947` (212 claims) is dropped at load time. Ticked 2026-10-13.)*
- [x] 3.3 Step 2: embed claims; cluster with HDBSCAN (min cluster size tuned so 60–120 clusters emerge); for each cluster, the teacher proposes a name, a one-line definition, and picks three representative claims. Write `packs/<genre>/schema_draft.yaml`. *(2026-10-13: claims embedded under an instruction prompt, centred on their document's mean and reduced with UMAP before HDBSCAN (decision record 003; PCA left 68–95% noise and plain embeddings clustered papers by topic). Run `cluster_v1_s0`: privacy 74 clusters, 15.5% noise; papers 80 clusters, 44.4% noise. Named by the 72B via OpenRouter (`name_72b_<genre>_s0`, prompt `induce_name` v2). Ticked 2026-10-13.)*
- [x] 3.4 Step 3, **human pass**: merge, split and prune to 40–60 types. For each type write: `id`, `name`, `definition`, `positive_examples` (2), `near_miss` (1), `severity_weight` (default 1.0; for privacy, mark the types that correspond to a legal requirement in GDPR Art. 13/14 or CCPA with 1.5), `section_hint` (where peers usually place it, used by the ghost-paragraph view), and **`label_source`: for genre A, the OPP-115 category/attribute values that count as a positive (this mapping is what makes 4.1 teacher-free; types with no OPP-115 mapping are listed and get teacher labels)**. Cross-check against the seeds: every OPP-115 category and every NeurIPS/ACL checklist item should map to at least one type or be explicitly excluded with a reason. *(2026-10-15: done by Parsa on a guided review page (artifact; answers in `reports/schema_review/decisions/`), with four follow-up resolutions in chat; privacy 58 items, papers 50; every OPP-115 category and every NeurIPS/ACL item mapped, except LLM-use declarations (NeurIPS 16, ACL E1), excluded because they post-date the corpus.)*
- [x] 3.5 Schema review: have the teacher (local small for iteration, 72B for the recorded run) classify 100 random sections against the schema and inspect the ones that match nothing; add or widen types if a recurring assertion has no home. Freeze as `packs/<genre>/schema.yaml` with `version: 1`. *(2026-10-15: recorded run `coverage_72b_v4_s3` (72B, fresh seed): sections with an unmapped statement after the second look, privacy 2% (first pass 9%), papers 0% (first pass 2%). Earlier runs (privacy 18%, 17%, 14%) led to 15 widened definitions and the second-look method; see CHANGELOG.)*
- [x] 3.6 **GATE: both schemas frozen, 40–60 types each, every type has definition, examples, near-miss, section hint; coverage check in 3.5 shows < 5% of sections with an assertion that maps to no type. Record both schemas' type lists in CHANGELOG.** *(Method fixed 2026-10-15, Parsa's decision after four coverage runs: the teacher first maps each statement of a held-out section to a type or "none"; every "none" then gets a second look, in which the teacher sees only the 6 types most similar to the statement (by embedding), with definitions and examples, and decides again. The gate counts sections with a statement that is still "none" after the second look; the first-pass rate is reported alongside. A briefly tried `background_or_filler` label was withdrawn the same day: on papers it absorbed real background statements. See CHANGELOG 2026-10-15.)* *(Passed 2026-10-15: `packs/privacy/schema.yaml` and `packs/papers/schema.yaml`, version 1, signed off by Parsa, run `freeze_v1_s0`.)*

---

## Phase 4 — Presence detection

- [x] 4.1 Presence labels. **Genre A: derive from OPP-115's segment annotations through the schema's `label_source` mapping (no teacher).** **Genre B: `prompts/label_presence.md`** (given a section and the schema, return the set of present type ids with an evidence span each) run with the 72B on ~2,000 train-split sections, stratified by section position and length, plus any genre-A types with no OPP-115 mapping. Full tier: 5,000 teacher-labelled sections per genre (`--overlay full`). Every label carries its source (`opp115`, `teacher:<tier>`, `human`). *(2026-10-15, in progress: genre A OPP-115 labels done (`labels_opp115_s0`, 3,792 sections, 35 types); label sets fixed in `packs/<genre>/label_sets.json` (human 300, teacher 2,000 per genre; privacy sets from disjoint OPP-115 policies); teacher labels with two passes (`label_presence` v1 + second look `label_presence_check` v2, calibrated against OPP-115) on all types of both genres, run `label_72b_s0` (72B, OpenRouter). Finding: OPP-115 labels mapped through `label_source` are loose for several types; see CHANGELOG.)*
- [x] 4.2 Presence reference set *(rewritten 2026-10-17, decision record 004; was "Human sanity set: 300 sections per genre labelled by hand")*: the 300 fixed sections per genre in `packs/<genre>/label_sets.json` (`human`). **Lean: silver labels** from a second judge of another model family (Claude Haiku 5.5 via OpenRouter, the same two-pass prompts as the teacher), the reference for every type; for genre A, OPP-115's human labels on its 35 mapped types are reported alongside (teacher vs OPP-115, judge vs OPP-115). **Upgrade: hand labels** on the guided page (`reports/human_labels/page/`), then `distil.score.reference` points to them and every number is recomputed. Measure teacher agreement against the reference (`reports/tables/teacher_vs_<reference>_<genre>.csv`). **If teacher F1 < 0.85 on any type with ≥ 20 positives, revisit that type's definition before distillation.** State in every result which reference it was measured against. *(2026-10-18: silver reference built (judge Claude Haiku 5.5, 600 sections). Teacher vs judge: privacy micro F1 0.775; papers 0.671, after which papers pack version 2 merged two near-duplicate groups (50 → 47 types, labels projected, no relabelling): 0.692. See CHANGELOG.)* *(2026-10-15: labelling page published; Parsa chose to hand-label all 58 privacy types as well (not only the 23 without an OPP-115 mapping), so the human set also measures the OPP-115-derived labels.)*
- [x] 4.3 `engine/classifier.py`, selected by `presence.head`: **`linear` (lean default)**: per-type logistic heads on the pack embedder's section embeddings, class-weighted, temperature-scaled on the reference set (4.2); **`setfit`**: contrastive fine-tune of the small embedder on the same labels, then the heads; **`finetune` (full tier)**: ModernBERT-large multi-label fine-tune (fallback DeBERTa-v3-large), 10% held out for early stopping. Report per-type P/R/F1 and ECE for every variant run, three seeds; the lean-vs-full comparison is the LO2 classifier-comparison evidence. **If `linear` misses gate 4.8, run `setfit` on the Mac before spending a GPU session on `finetune`.** *(2026-10-18: `linear` built (`heads_linear_v2_s0`): teacher labels for every type (decision record 005), C 3 chosen on the reference fit half; eval-half macro-F1 against the silver reference privacy 0.617, papers 0.508 (ceiling 0.773 / 0.688); ECE after temperature 0.061 / 0.124. Deterministic, so one seed is recorded. `setfit` not run: Parsa recorded papers below the gate instead (4.8).)*
- [x] 4.4 `packs/prototypes.py`: for each type, select 10–30 prototype sentences from labelled positives (any source) by k-means over their late-interaction embeddings (centroid-nearest per cluster). Store with document ids so they double as evidence. *(2026-10-18: answerai-colbert-small-v1; candidates = the teacher's verbatim evidence spans on training sections; privacy 1,031, papers 1,139 prototypes; run `prototypes_v1_s0`.)*
- [x] 4.5 `engine/presence.py`: the cascade (§0.2 #8). Plot the distribution of max prototype similarity for labelled-positive and labelled-negative sections per type; set the band thresholds at the overlap region per pack; record them in the pack config and the plan. *(2026-10-18: privacy [0.9341, 0.9508), papers [0.9210, 0.9459) (5% quantile of labelled positives, 95% of negatives; ColBERT MaxSim, answerai-colbert-small-v1); `configs/pack_<genre>.yaml`. The reranker is implemented but not used by the lean presence model (decision record 005).)* Learned logistic combination of classifier probability and cascade score, fitted on the reference set (4.2).
- [x] 4.6 Ablations on the reference set (4.2): classifier only, prototypes only, prototypes + reranker, full cascade. Table `reports/tables/presence_ablation_<genre>.csv`. Latency per section for each. *(2026-10-18: runs `cascade_v1_s0` (all variants, reranker included) and `cascade_v2_s0` (final heads, no reranker). The reranker adds +0.002 / −0.004 macro-F1 at 5–9× the time; classifier + prototypes is the lean model: privacy 0.619, papers 0.497, ECE 0.003 / 0.013.)*
- [x] 4.7 `packs/profiles.py`: run presence over the whole train split per genre → profile vectors (probabilities and thresholded bits) stored as Parquet. Resumable in chunks (the Mac runs this overnight). *(2026-10-18: run `profiles_v1_s0` on the M5 (MPS), classifier + prototypes: privacy 22,300 documents, 192,596 sections, 134 sections/s; papers 14,114 documents, 305,088 sections, 65 sections/s; 1 h 42 min. Negative finding: document-level presence is inflated (privacy 20.2 of 58 types per document, papers 26.8 of 47; `states_own_limitations` in 96.6% of papers) because a document counts a type when any section passes a section-level F1 threshold; to fix before omissions are ranked (Phase 5).)* *(2026-10-20, decision record 006: profiles v2 use a document-level model fitted on 150 whole labelled dev documents per genre (judge labels); the v1 rule was miscalibrated in both directions, not uniformly inflated; v1 kept as `packs/<genre>/profiles_v1`.)*
- [x] 4.8 **GATE (changed 2026-10-18, decision record 005; was "macro-F1 ≥ 0.80 against the reference set"): the presence model's macro-F1 against the reference set of 4.2 is at least 0.80 × the teacher's own macro-F1 against the same reference (the ceiling a student of the teacher can reach; with hand labels as the reference, the absolute 0.80 applies again), on the eval half, both genres, with the presence model the profiles use; profiles computed for the whole indexable corpus; throughput recorded (sections/second, and on which machine). If the lean head fails the gate after `setfit`, switch `presence.head: finetune` in the pack config, log it, and use the lab session; nothing downstream changes.** *(2026-10-18, run `gate_v1_s0`, `reports/tables/presence_gate.csv`: privacy 0.619 / ceiling 0.773 = **0.801, pass**; papers 0.497 / 0.688 = **0.722, not met**, recorded as a negative finding by Parsa's decision (diagnosis: a frozen whole-section embedding cannot separate papers' fine-grained types; SetFit on Colab is the recorded upgrade); profiles complete for the whole indexable corpus; throughput above. Phase 4 closed with papers below its gate.)*

---

## Phase 5 — Packs, index, cohort, level-1 expectation

- [x] 5.0 *(added 2026-10-20)* Document-level presence (decision record 006): whole-document judge labels (stage `select_documents`, `configs/judge_anthropic.yaml`), every section's probabilities (`section_probs`, Colab), document model (`engine/doc_presence.py`, stage `doc_presence`), profiles v2.
- [x] 5.1 `engine/embed.py`: embedding model behind one interface with batching, fp16 (fp32 on MPS where needed), caching to disk keyed by (model id, text hash), resumable over the corpus. Document embedding as in §0.2 #9. Time the 0.6B embedder on the Mac (sections/second) and record it. *(2026-10-20, Mac M5, MPS, fp16, query time (one document at a time, `models.embedder.query_batch_size: 4`), from gate 5.9: privacy 15.0 sections/s (44 sections in 2.9 s), papers 10.2 sections/s (117 longer sections in 11.5 s). The corpus embeddings were made on the lab GPU (`embed_0.6b_s0`, 586 / 345 sections/s); the laptop CPU runs ≈ 1.5 sections/s (2026-10-12).)*
- [x] 5.2 `engine/index.py`: `Index` interface with `add`, `search(vector, k, filters)`, `save`, `load`; FAISS backend (HNSW) and Qdrant backend (payload filters for genre, category, date). Tests on 1,000 vectors check both backends return the same top-10 on exact search. *(2026-10-19: same top-10 on 1,000 vectors under four filter settings. macOS: faiss-cpu and PyTorch bundle clashing libomp builds, so FAISS is loaded only for unfiltered approximate search; exact and filtered search is a NumPy pass.)*
- [x] 5.3 `engine/cohort.py`: cohort retrieval with category and date-window filters; **never returns documents of the query's own entity** (the corpus holds one snapshot per entity per year, so a site's other years would otherwise be its "peers"); falls back to unfiltered when the filtered pool is smaller than k. Report cohort similarity statistics (`reports/figures/cohort_similarity_<genre>.png`). *(2026-10-19, stage `cohort_report`, 500 dev queries per genre, k 200: mean peer similarity 0.783 / 0.745, own entity in a cohort 0 / 0, category fallback 5.6% / 0%. Privacy cohorts hold several yearly snapshots of the same other site (88.4% distinct entities): possible Phase 6 ablation.)*
- [x] 5.4 `engine/expectation.py` level 1: prevalence over cohort profiles; with Wilson confidence intervals so small cohorts don't produce overconfident expectations. *(2026-10-19: bit mean, as planned; mean probability (`expectation.profile: prob`) is the Phase 6 ablation, because bits saturate near-universal types, decision record 006.)*
- [x] 5.5 `engine/rank.py`, `engine/evidence.py`: scoring per §0.2 #1 and #13; evidence per #14 (counts, peer sentences via the prototype index, section hint). *(2026-10-19: peer sentences come from stored sentence evidence (stage `sentence_evidence`, Colab job `sentence_evidence_v1`: the best sentence of every indexed document per type under the full presence model), not from the prototype index: prototype MaxSim picked wrong sentences (CHANGELOG 2026-10-19).)*
- [x] 5.6 `engine/pipeline.py`: `analyse(text_or_path, pack) → Report` with `Report = {present: [...], absences: [{type, score, prevalence, ci, evidence, explanation?}], cohort: {...}, timings}`. Serialises to JSON. *(2026-10-19; task `analyse` (`configs/analyse.yaml`, 2026-10-20) runs it on named corpus documents and saves each Report.)*
- [x] 5.7 `packs/builder.py`: one command builds a pack from a config: load corpus → segment → embed → (schema must exist) → profiles → prototypes → index → `pack.json` manifest with versions, hashes, **the tier, and the model id behind every artefact**. Each stage is skipped when its inputs and model ids are unchanged, so `--overlay full` rebuilds only what the full tier changes. Run it for both genres. *(2026-10-20: run for both genres (`pack_privacy_adopt_s0`, `pack_papers_restored_s0`): unchanged stages skipped, `sentence_evidence` adopted; manifests `packs/<genre>/pack.json`.)*
- [x] 5.8 `engine/explain.py`: `explain.mode: template` (default) fills the §0.2 #14 template from the evidence; `explain.mode: teacher` adds a teacher rationale and suggested wording for the top-N, synthesised from the retrieved peer sentences only (the prompt forbids inventing content not present in the evidence; prompt in `prompts/explain.md`). Both produce the same `Report` fields. *(2026-10-20, run `explain_teacher_privacy_s0`: one policy, top 5, Qwen2.5-72B via OpenRouter (DeepInfra fp8), 2,157 + 462 tokens ≈ $0.001. By eye: no invented fact, number, law or period; three suggested wordings copy peer sentences verbatim; one rationale draws on the type definition (given in the prompt), not the quotes. The automatic check (words found in neither the quotes nor the type card) flags 3 function words.)*
- [x] 5.9 **GATE: `analyse()` runs end to end on a held-out document from each genre in under 10 seconds on the Mac (CPU or MPS) with template explanations; both packs load from their manifests. Record the time on each machine available.** *(Passed 2026-10-20 on the Mac (M5, MPS), 5 held-out dev documents per genre after one warm-up call: privacy max 1.98 s (17 sections), median 0.69 s; papers max 3.31 s (32 sections), median 3.07 s, embedding ≈ 85% of it. Both packs load from their manifests. Only the Mac was timed (the demo host is timed in 7.1). `reports/tables/analyse_gate_<genre>.csv`. Rerun 2026-10-21 on mains power after a battery-power rerun (runs `gate59_mains_<genre>_s0`): privacy max 1.37 s, median 0.35 s; papers max 3.28 s, median 3.11 s.)*

---

## Phase 6 — OmitBench-synthetic and the first results

- [x] 6.1 `bench/synth.py`: for each test document and each sampled type present in it, delete the sentences (or the whole section) carrying that type, verified by re-running presence on the edited document (the type must now be absent). Three regimes per §0.2 #15. 500 instances per regime per genre. Store the edited documents and the ground-truth deleted types. *(2026-10-20, run `bench_synth_v1` on Colab; test-split probabilities from Colab job `bench_probs_v1`. Parsa's choices: types drawn uniformly from the document's present types; ground truth = every type that goes from present to absent (sampled targets plus collateral); whole_section = every section carrying the sampled type. A sentence carries t when its presence probability passes the cascade threshold; while a target is still present, the edit escalates (next most likely carriers, then whole sections), at most 50% of the characters removed, up to 3 draws per document, one document per entity per regime; edits work on the stored sections. Instances: privacy 500 / 500 / 500, papers 500 / **468** / 500 (three_types: every test paper tried; diffuse presence in long papers, Parsa: keep the rules and report the shortfall). Per-type acceptance: `reports/tables/bench_synth_acceptance_<genre>.csv`.)*
- [x] 6.2 `bench/metrics.py`: recall@k, MRR, nDCG@10, false-alarm rate, ECE, bootstrap 95% CIs over documents. *(2026-10-20; plus the paired bootstrap of the gate, bootstrap over groups (entities, for revision chains) and a within-document AUC (decision record 007). False alarms and ECE on the 150 judge-labelled dev documents per genre (decision record 006) with out-of-fold document presence.)*
- [x] 6.3 `experiments/bench.py`: run baselines (a) genre-uniform prevalence and (b) level-1 cohort prevalence on both genres and all regimes. Table `reports/tables/bench_synth_<genre>.csv`. *(2026-10-20, run `bench_eval_v1`, w = 1; reference rows random and absence-only. Recall@5 single_type, level 1 vs genre-uniform: privacy 0.396 vs 0.385, papers 0.622 vs 0.514.)*
- [x] 6.3b `bench/revision.py`: **OmitBench-revision.** For each revision pair from 2.6b: run presence on both versions; the revision-added types are those present in *n*+1 and absent in *n*; keep a pair only if at least one added type has cohort prevalence ≥ 0.5 among peers dated before version *n* (time-windowed cohort, §0.2 #19). Instance = (version *n*, the qualifying added types). Report counts per genre and year, and the prevalence distribution of added types, in `reports/tables/revision_bench_<genre>.csv`. Run (a) and (b) on it. *(2026-10-20, runs `bench_revision_probs_v1` (Mac) and `bench_revision_v1`: privacy 1,678 of 4,029 pairs, papers 137 of 343; (a) uses the corpus prevalence of the same date window; bootstrap over entities; also a system-neutral variant (every added type), because the plan's filter selects with the level-1 quantity. Recall@5 level 1 − genre-uniform: privacy −0.010 [−0.023, 0.002], papers +0.107 [0.046, 0.166]. Table `bench_revision_<genre>.csv`.)*
- [x] 6.4 Ablations: k ∈ {20, 50, 200, 1000}; category filter on/off; cascade vs classifier-only presence; **lean vs full embedder (0.6B vs 8B) if a GPU session allowed the full-tier pack**. Figures in `reports/figures/`. *(2026-10-21: k, category filter, title weight, one snapshot per entity and mean probability were run on the test bench as the diagnosis of gate 6.6 (`cohort_diagnosis_<genre>.csv`, decision record 007), not as a selection. Severity weights on (`rank.use_severity_weights: true`), run `bench_ablation_severity_v2`, paired against `bench_eval_v1`: `reports/tables/ablation_severity_<genre>.csv` (papers has no weight ≠ 1, so no change). Figures `bench_recall5.png`, `level1_minus_uniform.png`, `cohort_diagnosis.png` from `python -m analysis.figures`. **Deferred (Parsa, 2026-10-21):** cascade vs classifier-only presence end to end (the instances were verified with the cascade, which biases the comparison; 4.6 compared the two at section level). Lean vs full embedder: not run, no full-tier pack.)*
- [x] 6.5 Error analysis: which types are missed, which false alarms recur; by document length, date, category. `analysis/error_analysis.py` → `reports/tables/errors_<genre>.csv`. *(2026-10-21, stage `bench.stage: errors`, run `bench_errors_v3` on `bench_eval_v2` (which reproduces `bench_eval_v1` and also records the untouched documents' rankings).)*
- [ ] 6.6 **GATE: level-1 beats genre-uniform prevalence on recall@5 by a margin outside the bootstrap CI on both genres. If not, the schema or the cohort retrieval is wrong; diagnose before Phase 7.** *(2026-10-21, criterion (Parsa, 2026-10-20): paired bootstrap of the recall@5 difference on single_type, 95% CI above zero. **Papers passed** (+0.108 [0.074, 0.141]); **privacy failed** (+0.011 [−0.010, 0.033]). Diagnosed (decision record 007, `reports/tables/cohort_diagnosis_<genre>.csv`): privacy cohorts carry little information about practices, the category filter and the title part of the document vector weaken them further, and the deletion itself moves the edited policy toward peers that omit the type (the leak), which removes most of what a better cohort gains. **Edited 2026-10-21 (Parsa's decision):** the gate is recorded as failed for privacy and Phase 7 proceeds; the privacy result is reported as a negative finding, the cohort settings stay as fixed in §0.2 #9 and #10, and level 2 (Phase 8) is where the leak is addressed.)*

---

## Phase 7 — Class demo and report

- [x] 7.1 `app/gradio_app.py` (runs on CPU with the lean pack, so it can live on a free Hugging Face Space): upload or paste a document; pick a pack or let auto-detect choose; output in three panels: (i) the document with **ghost paragraphs** inserted at the section hint of each top absence (dashed outline, prevalence, severity); (ii) the **cohort panel** (k, similarity, category and date filters, time-window slider); (iii) the **evidence panel** for a selected absence (peer count, five peer sentences with document ids, teacher rationale, suggested wording). Toggle: "compare with a chat model" runs baseline (e) on the same document and shows both outputs side by side. *(2026-10-21, not ticked: built and verified in the browser on one policy and one paper (run `serve_s0_20261021-114046-082824`, Mac M5 / MPS: privacy 0.74 s, papers 2.20 s per analysis): engine split `prepare` / `score` (the cohort controls re-rank without embedding again; prepare + score equals analyse on corpus documents of both packs), ghost placement from the section hint (`engine/explain.py`), auto-detect (`engine/onthefly.py`), cohort panel (k, category filter, date window), evidence panel; teacher rationale behind `serve.allow_teacher` (off by default, flagged). Evening: redesigned (dark, animated, lens cursor and trail; `app/static/`), pinned examples with the cached 7.3 answers (`app/examples.py`), the chat-model toggle (three columns for benchmark examples, cached answers only), ghost click opens the evidence; guided demo (steps, narration, progress, challenge/reveal for benchmark examples) and an About tab explaining the project with numbers copied from the tables (`app/content.py`, `app/about.py`); Parsa's palette (cream / teal / light teal / orange), approved. Space bundle `python -m app.space_bundle` (305 MB, corpus-free; CPU variant run locally: 4.6 s per paper on the Mac's CPU, 6.1 GB peak RSS; ZeroGPU variant for Parsa's Pro account, untested locally). **Live 2026-10-21:** https://huggingface.co/spaces/parsa-hke/negative-space-reader (public, ZeroGPU; four builds: 60-char description, Python 3.10 pins, pydantic range, LFS patterns for npz/parquet). On ZeroGPU: policy 1.95 s, paper 0.5 s; moved the same day to `cpu-basic` because ZeroGPU charges each visitor's own quota (Parsa): policy 9.6 s, paper 35.8 s, results identical. Ticked.)*
- [ ] 7.2 Record a 3–5 minute walkthrough video of the demo on one privacy policy and one paper. *(2026-10-25: shot list, narration and timings planned in `reports/demo_video_plan.md`: the Etilize policy, the FANS paper, and the MarIA benchmark instance as the reveal ending (level 1 three of five removed types in the top five; Opus two, Qwen none; recorded ranks). Examples pinned in `serve.examples` / `serve.bench_examples` and shipped in `app/examples/`. Recording pending.)*
- [x] 7.3 Frontier-model comparison, candidate-list setting only (baseline (e)) on 100 synthetic instances per genre; one frontier API model (**Claude Opus 5.5 via the Anthropic API, Message Batches at 50% off, ≈ $10–20**; route changed 2026-10-18 from OpenRouter: Parsa has $100 of Anthropic API credit; called with the official `anthropic` SDK) plus the open teacher. Table `reports/tables/llm_gap_candidate_<genre>.csv`. *(2026-10-21: prompt `prompts/llm_gap_candidate.md` v1 and stage `baselines.stage: candidate` written; dry run `llm_gap_candidate_dry_s0`: 100 single_type instances per genre, ≈ 1.6 M input tokens, ≈ $8 by Message Batches at an assumed 2,500 output tokens per request. Run 2026-10-21 after Parsa's approval (`llm_gap_candidate_opus_s0`, ≈ $4.2): recall@5 Opus 5.5 vs level 1 vs genre-uniform, privacy 0.242 / 0.376 / 0.375, papers 0.235 / 0.655 / 0.465; Opus below both, CIs excluding zero. Open teacher Qwen2.5-72B via OpenRouter (`llm_gap_candidate_qwen_s0`, ≈ $0.40): 0.137 / 0.181, below both; one papers prompt beyond its 32k context counted as a miss; 60-69% of its lists follow catalogue order.)*
- [ ] 7.4 `reports/report.md`, 3,000–4,000 words, structured per §11.1. Every number copied from `reports/tables/` by `analysis/tables.py`, never typed. *(2026-10-25, drafted and polished, awaiting Parsa's read: `analysis/tables.py` writes the fragments and `values.json`; `analysis/report.py` assembles `reports/report.md` from `reports/report_source.md` (placeholders for every value, fragment and citation; a test keeps them in step); 3,991 words all-in, 3,865 for Sections 1–10; checker zero errors; 30 verified IEEE references; `reports/report.docx` exported. See CHANGELOG. 2026-10-25 later: Parsa's title-block edit applied to the source, report and .docx rebuilt; `python -m analysis.report --words` added: 3,989 words all-in, 3,835 for Sections 1–10.)*
- [ ] 7.5 GenAI acknowledgement section as required by the subject outline: tools used, for what, with the prompt-history practice described. *(2026-10-25, drafted at the end of the report from the practice in CLAUDE.md; Parsa confirms the claims before submission.)*
- [x] 7.6 README: one-command setup, one-command pack build in smoke mode, one-command bench, expected runtimes and hardware. Fresh-clone test: smoke passes from a clean checkout. *(2026-10-25: README rewritten (setup, the three levels a clone can run, the pipeline as runs with recorded durations, hardware, models). The smoke overlay now redirects `paths.packs` and `paths.embeddings` under `data/processed_smoke/` and defines the smoke pack build as the label-free stages `embed`, `index`, `cohort_report` on the 200-document corpus (the presence stages need the recorded teacher labels of the real corpus). Fresh clone in a scratch folder: install 60 s, 484 tests passed (1 skipped, MLX), smoke pack builds 28.6 s / 43.4 s; one engine bug found and fixed (the builder did not create the folder of a pack built from nothing; test). The bench smoke is the eval stage on the recorded synth instances, 51 s, reproducing `bench_synth_<genre>.csv` exactly. Ticked.)*
- [ ] 7.7 Tag `v0.1-class`; submit the report, repository link and demo per the outline. **GATE: submitted.** *(2026-10-25: tag message, commands and the submission checklist prepared in `reports/submission_v0.1-class.md`; package version 0.1.0. Open point: the repository is private and must be visible to the marker.)*

---

## Phase 8 — The masked-assertion model (MAM)

- [ ] 8.1 `models/mam.py`: inputs = present-type tokens (learned embedding per type, order-free), a projected document embedding token, metadata tokens (category, year bucket), a `[CLS]`; a 4–6 layer transformer encoder; output head = one logit per type. Training: for each indexed document, mask 15–40% of present types (random rate per example), predict the masked ones with BCE; also add 10% "no-mask" examples where the target is the full present set so the model learns prevalence. Negative types are also supervised (target 0) with a down-weight so the model doesn't collapse to "everything present".
- [ ] 8.2 `experiments/train_mam.py`: train per pack on the train-split profiles (20–30k documents per genre, masking gives many examples per document); early stopping on dev-split masked-type AUROC; three seeds; minutes per seed on the Mac. Calibrate with temperature scaling on dev.
- [ ] 8.3 `engine/expectation.py` level 2: inference for all absent types in one forward pass; attribution of each P(*t*) to present types by gradient × input over the type embeddings (the "because you say X" explanation).
- [ ] 8.4 Bench: MAM vs level-1 on OmitBench-synthetic (all regimes) and OmitBench-revision, both genres. Ablations: without document embedding; without metadata; with k-cohort profiles added as extra tokens (a hybrid). **This is the paper's main table.**
- [ ] 8.5 Qualitative: 20 conditional omissions per genre of the form "says A → 90% of peers that say A also say B → doesn't say B", checked by hand for sense.
- [ ] 8.6 **GATE: MAM beats level-1 on recall@5 and nDCG@10 outside the CI on at least one genre, or a negative result is documented with error analysis (still a paper, framed differently).**

---

## Phase 9 — Validating OmitBench-revision and the LLM gap

- [ ] 9.1 Annotation guidelines (`bench/real/guidelines.md`): what "conspicuously missing" means, worked examples, what to do when the schema has no fitting type. Pilot on 10 revision instances, revise.
- [ ] 9.2 **Validate the revision labels:** 50 instances per genre, two annotators each, judging for every revision-added type whether version *n* conspicuously omitted it (yes / no / not a schema type). Report Krippendorff's α and the precision of revision labels against the adjudicated judgement. **If precision < 0.7 on a genre, fall back to the full-tier blind annotation (`bench.real.source: annotation`, 150 × 2) for that genre.** Store raw labels and adjudications separately. Also record the confound check: fraction of added types that coincide with a known regulatory date.
- [ ] 9.3 Bench on OmitBench-revision: level-1, MAM, and the LLM baselines (d)–(f) with fixed prompts from `prompts/`, three runs each for sampling variance, batch pricing where available, cohort peers in (f) capped at `baselines.rag_peers`. Open-ended outputs are mapped to schema types by embedding similarity to type definitions (teacher mapping as the full-tier option); report the mapping's agreement with a human on 100 items.
- [ ] 9.4 Calibration analysis: ECE and reliability diagrams for MAM vs the LLMs' verbalised confidence. Analysis of what LLMs miss: by type prevalence, by whether the type is conditional on another, by document length.
- [ ] 9.5 **GATE: OmitBench-revision results for every system in one table with CIs; the LLM-gap figure drawn; the validation precision and α reported next to them.**

---

## Phase 10 — Drift, on-the-fly mode, and a third genre

- [ ] 10.1 `experiments/drift.py`: for genre A, expectation vectors from cohorts restricted to 2016, 2017, 2019, 2021, 2023; plot per-type prevalence over time and the shift at GDPR; same for genre B around the 2022 limitations mandate. Figure `reports/figures/drift_<genre>.png`. Then the demo's time slider reads from the same code.
- [ ] 10.2 `engine/onthefly.py`: pack detection by embedding similarity to each pack's centroid; below threshold, cohort from the general index and a temporary schema via steps 3.2–3.3 on 100 cohort documents; clearly flagged in the Report. Measure on held-out documents from the two packs with the packs hidden: how often does the temporary schema recover the frozen types (Jaccard over matched definitions)?
- [ ] 10.3 Third genre (pick one from §0.2 #22 by corpus availability): run the pack builder unchanged; any code change needed is logged as an engine bug and fixed. Bench synthetic; 50-document real annotation if time allows.
- [ ] 10.4 **GATE: a third pack builds with zero engine changes, or every change is recorded and justified.**

---

## Phase 11 — Paper

- [ ] 11.0 Write the limitations section first (per §0.2 #21): one community's norms; schema induction bias; English only; deletion benchmark vs real omissions; **revision labels confounded by regulation and by authors' own drift (mitigated by time-windowed cohorts, quantified in 9.2)**; small-model presence detection in the lean tier; dual use (optimising a document to *hide* omissions from the detector).
- [ ] 11.1 Sections: Introduction (absence as a first-class signal; the LLM gap); Task definition; OmitBench; Method (packs, cascade, MAM); Experiments (synthetic, real, LLM baselines, drift, third genre); Analysis; Related work (compliance checking, checklist verification, contrastive/counterfactual retrieval, set prediction, LLM negation and absence reasoning); Ethics; Limitations.
- [ ] 11.2 Reproducibility: release packs' schemas, OmitBench (synthetic generator, revision-pair manifests, validation annotations), prompts with hashes, cached teacher responses, configs for both tiers, seeds; model weights for the presence heads and MAMs.
- [ ] 11.3 Internal review: have someone outside the project run the fresh-clone smoke and read the paper cold.
- [ ] 11.4 Submit to the chosen venue; preprint on arXiv.

---

## Phase 12 — Product layer

- [ ] 12.1 `app/api.py`: FastAPI with `POST /analyse` (document, pack, options), `GET /packs`, `GET /absence/{id}/evidence`; async job queue for long documents; auth; rate limits; no document persistence by default.
- [ ] 12.2 `app/web/`: upload, ghost-paragraph reader, cohort and evidence panels, time slider, export to PDF/JSON, "compare with chat model" toggle. Reuse the Gradio panels' logic.
- [ ] 12.3 On-prem packaging: Docker Compose (API, Qdrant, vLLM); a `pack` bundle format signed and versioned; an offline mode with the open teacher only. **Licence check before shipping:** the lean tier is Apache/MIT throughout; the full tier's Jina ColBERT v2 and NV-Embed-v2 are CC-BY-NC and need a commercial licence or a swap.
- [ ] 12.4 Monitoring mode: watch a set of documents and re-run against a cohort that refreshes monthly; alert on new conspicuous omissions as norms drift.
- [ ] 12.5 First vertical (choose by conversations with users): contract review against market standard, journal editorial checks, or compliance narratives. Pack building is the service; the engine is the moat.

---

## Appendix A — Working with Claude Code on this project

- Start every session with: "Read NEGATIVE_SPACE_PLAN.md and CHANGELOG.md. We are on Phase N, task N.M."
- Say which tier a task is for. The default is lean; `--overlay full` is the upgrade and should be used deliberately, never by accident.
- Ask for one task at a time, and for the test before the implementation when a module is new.
- After any corpus inspection task, ask it to paste the actual field names it found into `data/README.md`.
- Before any GPU job, ask it to run smoke mode with the stand-in models and show the metrics file.
- Every prompt change is a new version in `prompts/`; ask it to rerun the 20-section eye test (3.1) after any change.
- Ask it to update `CHANGELOG.md` at the end of each session with what changed and what was found, including negative findings.
- When a result is surprising, ask for the three most likely bugs that would produce it before accepting it. For this project the usual suspects are: test documents leaking into the index; presence thresholds set on the same data they're evaluated on; a schema type that is really two types.

## Appendix B — Things that will go wrong

- The policy corpus is huge and noisy; version collapsing (2.6) will decide whether the cohort is "200 policies" or "200 snapshots of 30 policies". Check cohort entity diversity.
- unarXive section structure is imperfect; LaTeX residue leaks into sentences. Budget cleaning time; measure it.
- Schema induction will produce types that are the same assertion phrased two ways; the human pass (3.4) is where the project is won or lost. Do not skip it or rush it.
- The teacher will be confidently wrong on a few types; the reference set (4.2) catches it. Don't distil before 4.2. With silver labels (lean tier) it catches only the errors the second judge does not share; OPP-115 is the one human check (genre A, 35 types), and hand labels are the upgrade.
- A 70B teacher may not fit; the 32B fallback is in §0.2 #6. Don't let model size block Phase 3.
- Presence false negatives become omission false positives. The false-alarm rate is as important as recall; report both always.
- The MAM can learn to predict prevalence and nothing more. The ablation "without document embedding" and the conditional-omission check (8.5) tell you whether it learned anything conditional.
- LLM baselines are sensitive to prompt wording; fix prompts early, version them, and report variance across three runs.
- Checkpoint names and licences on the hub change; verify each one on the day and record it.
- OmitBench-revision will contain additions driven by law (GDPR, CCPA) rather than by peers. The time-windowed cohort handles the ranking; 9.2 measures how much of the benchmark it is. Report it, don't hide it.
- The lean presence head may be good at common types and weak at rare ones. Per-type F1 in 4.3 tells you which; prototypes + reranker in the cascade are there to catch exactly those.
- Overnight Mac jobs die (sleep, memory). Every corpus-scale stage checkpoints in chunks and resumes; `caffeinate` the run.

## Appendix C — Mapping to the Assessment 2 brief and rubric

| Brief item / rubric criterion | Weight | Where it is answered |
|---|---|---|
| Problem definition and justification for NLP | 20% | §0.1, §0.2 #1; omissions have no text to search for, so detection requires semantic modelling of genre expectations |
| Design of the NLP pipeline / architecture | 25% | §0.2 #3–#14, #25; Phases 4–5; ablations in 4.6 and 6.4; the lean/full decision record (`reports/decisions/001`) as the design-trade-off discussion |
| Data considerations and pre-processing | 15% | §0.2 #18; Phase 2 (segmentation, dedup, version collapsing, entity-disjoint splits) |
| Ethical and responsible AI | 15% | §0.2 #21; Phase 11.0; the UI design in 7.1 (evidence, not verdicts) |
| Clarity, structure, presentation | 15% | §11.1 structure; every number from a table; demo video |
| Use of tools, frameworks, platforms | 10% | Hugging Face (transformers, sentence-transformers, hub models), vLLM, Qdrant/FAISS, Gradio, FastAPI; each named with its role in the report |
| Input data description | brief | Phase 2.8 corpus stats table |
| Pre-processing and embedding techniques | brief | Phase 2.5–2.6; §0.2 #9 |
| Proposed models / pipelines | brief | §0.2 #6–#13 |
| Tool stack | brief | §0.2 #23 |
| Expected output and value proposition | brief | §0.2 #1, #14; §0.3 |
| Learning outcomes LO1–LO3 | — | LO1: Phase 2; LO2: classifier comparison 4.3 and presence ablation 4.6; LO3: Phase 5 pipeline |

## Appendix D — Lab session runbook (UOWD GPU, day sessions, nothing overnight)

Written 2026-10-12 for the first session. Every step is one command; every long step resumes from the teacher's
response cache if the session ends early. Copy results home before leaving.

**Shortcut:** `./lab_session.sh setup | download | vllm-check | eye | extract | pack` runs the steps below in order
(the script picks the teacher build from the GPU report). The bundle `lab_bundle/negspace_lab.tar.gz` holds the code,
the processed documents and sections, the split manifests and the 7B reference run.

**Before leaving home**
1. Commit the repository (or copy the folder to a USB drive). The eye-test selection
   (`prompts/eval/induce_extract_eye_test.json`) and the prompt versions are tracked files.
2. Copy `data/processed/` (≈ 2.5 GB, gitignored) and `runs/eye_v2_s0/` (the 7B reference output) to the drive.
3. If the server may lack internet, also put the 72B weights on the drive:
   `hf download Qwen/Qwen2.5-72B-Instruct-AWQ --local-dir <drive>/Qwen2.5-72B-Instruct-AWQ` (41.6 GB).

**Session start (≈ 15 min)**
1. `python3.11 -m venv .venv && .venv/bin/pip install -e ".[gpu]"` (no Python 3.11? `uv python install 3.11`).
2. `export HF_HOME=<large disk>/hf` so model weights do not fill the home directory.
3. Start the 72B download in a second terminal now: `.venv/bin/hf download Qwen/Qwen2.5-72B-Instruct-AWQ`.
4. `.venv/bin/python -m experiments.run --device-info`. Paste the output into README "Compute" and §0.6 at home.
   Set the teacher precision in `configs/gpu.yaml` from the VRAM (table in that file).
5. Put `data/processed/` and `runs/eye_v2_s0/` into the repository folder on the server.

**vLLM check with a small model (≈ 5 min, the vLLM path has never run on CUDA)**
`python -m experiments.run --config configs/induce.yaml --overlay gpu --name vllm_check --set pack=privacy
--set models.teacher.id=Qwen/Qwen2.5-1.5B-Instruct --set models.teacher.quantization=null --set models.teacher.dtype=bfloat16`
Pass: `[ok]`, and `runs/vllm_check_s0/eye_test.html` shows 20 privacy sections with extractions.

**Task 3.1 confirmation on the 72B** *(superseded for the Windows lab machine by `lab_session.py` and the 32B, decision record 002)*
`python -m experiments.run --config configs/induce.yaml --overlay gpu --name eye_v2_72b --set induce.eye_test.compare_with=runs/eye_v2_s0`
Open `runs/eye_v2_72b_s0/eye_test.html` (the 72B on the left, the 7B on the right, spans highlighted in the text).
Accept `induce_extract` v2 when all of these hold:
- every claim is stated in its section (read all 40; no inference, no outside knowledge);
- paper claims are moves ("the authors report…", "define…", "compare…"), not subject-matter facts;
- different kinds of parties, purposes or conditions are separate claims (e.g. service providers vs authorities);
- background and related work are summarised in a few claims, not one per cited work;
- `metrics.json`: span grounding ≥ 0.95 in both genres, no empty sections, first-person claims 0;
- no claim reuses an example label (`loan_period`, `late_fee_policy`, `overdue_sanction`) for unrelated content.
If one fails: copy the prompt to `prompts/history/induce_extract.v2.md` (already there), edit
`prompts/induce_extract.md`, bump `version`, rerun the command with `--name eye_v4_72b --set
induce.eye_test.compare_with=runs/eye_v2_72b_s0`. A 72B rerun takes minutes, so iterate in the session.

**Before leaving**
Copy home: `runs/eye_*_72b_s0/`, `runs/vllm_check_s0/`, the new lines of `runs/log.csv` (append them to the home
`runs/log.csv` without the header), and `.cache/negspace/teacher/` (the response cache: reruns become free).
At home: record the accepted version and its hash in CHANGELOG and `prompts/history/README.md`, tick 3.1.

**No GPU?** The same confirmation from home through OpenRouter (≈ $0.05 for 40 sections): `export
OPENROUTER_API_KEY=…` then the 3.1 command with `--overlay openrouter` instead of `--overlay gpu`. Record that the
served build was DeepInfra fp8, not AWQ.
