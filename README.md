# Negative-Space Reader (`negspace`)

Detecting what a document *conspicuously omits*, by modelling what its genre peers assert.
MAAI 902 Assessment 2 (NLP Solution Design) → paper → product. Parsa Hosseini, UOWD, 2026.

Given a document and a corpus of its genre, the engine ranks the kinds of statement the document lacks by how many of
its 200 closest peers make them, and shows the peers' own sentences as evidence. Two genres ship as **packs** (privacy
policies, 58 kinds of statement; cs.CL/cs.LG papers, 47): the engine has no genre-specific code. Nothing is a verdict:
every number is a count of peers and every quote is theirs.

- **Demo:** https://huggingface.co/spaces/parsa-hke/negative-space-reader (public, free CPU hardware; it sleeps after
  two idle days and wakes on the first visit).
- **Report:** [reports/report.md](reports/report.md) (`reports/report.docx` is the submitted copy).
- **Status:** Phases 1–7 of [NEGATIVE_SPACE_PLAN.md](NEGATIVE_SPACE_PLAN.md) on the lean tier, tag `v0.1-class`.
  Progress and findings, including negative ones, are in [CHANGELOG.md](CHANGELOG.md); design decisions written for the
  report are in [reports/decisions/](reports/decisions/). **Starting a work session? Read [HANDOFF.md](HANDOFF.md).**

## One-command setup

Python 3.11 is required (pinned in `pyproject.toml`; `python3.11` must be on the path).

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e .
```

Then the tests, which need no data and no model weights beyond two optional tiny ones:

```bash
.venv/bin/python -m pytest -q
```

Optional extras: `.[mac]` (Apple Silicon: `mlx-lm` for the local 7B teacher), `.[gpu]` (Linux + CUDA: vLLM),
`.[tracking]` (wandb). `requirements.txt` is the transitive freeze of the development machine, for the record.

## What a fresh clone can run

The corpora, embeddings, pack artefacts, run folders and caches are never committed (`.gitignore`): they are third-party
text or rebuildable. Three levels of "it works", from nothing to everything:

| Level | Needs | Command | Time on the M5 |
|---|---|---|---|
| Tests | the clone | `.venv/bin/python -m pytest -q` | ≈ 25 s (485 tests) |
| Smoke pack build | the 200-document smoke corpus (38 MB, see below) | `python -m experiments.run --config configs/pack_privacy.yaml --overlay local --overlay smoke` (same for `pack_papers`) | privacy ≈ 25 s, papers ≈ 42 s |
| Smoke bench | the recorded bench artefacts (`hub_sync pull --bundles runs,packs`) | `python -m experiments.run --config configs/bench.yaml --set bench.stage=eval --set 'bench.eval.synth=runs/bench_synth_v1_s0/{genre}'` | 51 s; reproduces `reports/tables/bench_synth_<genre>.csv` exactly |
| Everything | the raw downloads (57 GB) or the hub bundles | the runs in the table further down | hours to days, see below |

**The smoke corpus** (`data/processed_smoke/`, 200 documents per genre, written by the data build with the `smoke`
overlay) needs the raw downloads listed in [data/README.md](data/README.md):

```bash
python -m experiments.run --config configs/data.yaml --overlay local --overlay smoke
```

**The smoke pack build** runs the label-free stages of the pack builder (`embed`, `index`, `cohort_report`) on that
corpus with the smoke models (`BAAI/bge-small-en-v1.5`), and writes only under `data/processed_smoke/` (`packs/`,
`embeddings/`, `tables/`, `figures/`): the real packs are never touched. The presence stages (`heads`, `prototypes`,
`cascade`, …) need the recorded teacher labels of the real corpus and are not part of smoke. Smoke must pass before
any full run (CLAUDE.md); the smoke overlay goes last so that its models win: `--overlay local --overlay smoke`.

**Fresh-clone test (2026-10-25):** `git clone`, the one-command setup (60 s with cached wheels; the first
install downloads about 3 GB of wheels), `pytest` (484 passed, 1 skipped: the MLX teacher test, since the `.[mac]` extra
was not installed; 2 min 8 s on a cold cache), the smoke corpus copied in, both smoke pack builds (28.6 s and 43.4 s,
model loading included). It found and fixed one bug: the builder did not create the pack folder of a
pack built from nothing.

**The private data repo.** Everything gitignored but needed to reproduce a run travels through one private Hugging Face
dataset repo (`hub.repo` in `configs/base.yaml`), as named bundles:

```bash
python -m experiments.hub_sync pull --bundles data,embeddings,packs,runs
```

`status` compares without writing; `push` uploads what differs (never while a job writes into a bundle).

## Running things

Every experiment goes through one entry point with a config from `configs/`:

```bash
python -m experiments.run --device-info
```

```bash
python -m experiments.run --config configs/induce.yaml --overlay local --overlay smoke --set pack=papers
```

- `--overlay NAME` applies `configs/NAME.yaml` on top (repeatable, in order; compute overlay first, `smoke` last).
  - **Compute:** `local` (Mac, MPS, MLX teacher), `lab` (the UOWD machine, llama.cpp teacher), `colab` (CUDA, fp16),
    `gpu` (a Linux server with vLLM), `openrouter` (72B teacher through OpenRouter; `OPENROUTER_API_KEY`),
    `llm_gap_anthropic` / `judge_anthropic` (Claude through the Anthropic SDK; `ANTHROPIC_API_KEY`).
  - **Mode:** `smoke` (200 documents per genre, tiny models, one seed, everything under `data/processed_smoke/`),
    `full` (the full tier: every tiered key flipped at once, plan §0.2 #25).
- `--set dotted.key=value` overrides one key (YAML-parsed; unknown keys are rejected, `+key` adds one); `--seed N`;
  `--name` names the run; `--print-config` shows the resolved config.
- Tasks: `build_data | build_pack | embed | induce | distil | train_mam | bench | analyse | baselines | drift | serve`.
- Each (config, seed) writes `runs/<name>_s<seed>/{config.yaml,metrics.json}` and one line to `runs/log.csv`, failed
  and interrupted runs included: tier, config hash, git commit, device, the version and hash of every prompt rendered,
  which model build served every teacher call. Of `runs/`, only `log.csv` is tracked.

### The pipeline, as runs (lean tier)

Recorded durations from `runs/log.csv`; the hardware column names where the recorded run happened.

| Step (plan) | Config and stage | Hardware | Duration |
|---|---|---|---|
| Corpus build, both genres (Phase 2) | `data.yaml --overlay local` | M5 | 3–4 min per genre, plus ≈ 1.5 h once for the arXiv e-prints of the revision pairs (one request per 3 s, cached) |
| Schema induction: extraction (3.1–3.3) | `induce.yaml --overlay lab`, `induce.stage=extract` | lab, Qwen2.5-32B Q4_K_M on 4 GPUs | 107 min for 500 policies; 179 min for 303 papers (stopped at the session deadline) |
| Schema induction: cluster, name, review, freeze (3.3–3.6) | `induce.yaml`, stages `cluster`, `name`, `candidate`, `freeze` | M5 / OpenRouter | minutes each; naming ≈ 8–10 min per genre through OpenRouter |
| Teacher labels, ≈ 2k sections per genre (4.1–4.2) | `distil.yaml --overlay openrouter`, `distil.stage=teacher_labels` | OpenRouter, Qwen2.5-72B | ≈ 1.5 h both genres, ≈ $7 |
| Silver reference labels (4.2) | `distil.yaml --overlay judge_anthropic` | Anthropic API, Claude Haiku 5.5 | ≈ 10 min, ≈ $1 (plus ≈ $2.6 for the document-level set) |
| Corpus embeddings (5.1) | `embed.yaml --overlay lab` | lab, one GPU | 29 min both genres (Qwen3-Embedding-0.6B) |
| Presence heads and cascade thresholds (4.3–4.5) | `pack_<genre>.yaml` stages `heads`, `prototypes`, `cascade` | M5 | seconds; cascade ≈ 2 min |
| Section probabilities and sentence evidence (Phase 5) | `pack_<genre>.yaml` stages `section_probs`, `sentence_evidence` | Colab A100 (`colab_jobs.yaml`) | 30 min and 2.4 h, both genres |
| Document presence, profiles, index, evidence (5.1–5.6) | `pack_<genre>.yaml` stages `doc_presence`, `index`, `evidence` | M5 | seconds per stage |
| Gate 5.9: analyse held-out documents | `pack_<genre>.yaml` stage `analyse_gate` | M5, MPS | 0.74 s per policy, 2.2 s per paper |
| OmitBench-synthetic: test probabilities, instances, scoring (6.1–6.3) | `bench.yaml`, stages `probs`, `synth`, `eval` | Colab A100; eval on the M5 | 23 min; 1.8 h; 51 s |
| OmitBench-revision (6.3b) | `bench.yaml`, stages `revision_probs`, `revision` | M5 | 1.8 h; 3.5 min |
| Cohort diagnosis, error analysis, severity ablation (6.4–6.6) | `bench.yaml`, stages `diagnose`, `errors`; `rank.use_severity_weights=true` | M5 | 9 min; 5 s; 52 s |
| Frontier comparison, candidate-list setting (7.3) | `baselines.yaml --overlay llm_gap_anthropic` / `--overlay openrouter` | Anthropic Message Batches; OpenRouter | 12 min, ≈ $4.2 (Claude Opus 5.5); 4.7 min, ≈ $0.40 (Qwen2.5-72B) |
| The demo (7.1) | `serve.yaml` | M5 | app ready in ≈ 15 s; on the Space's free CPU 9.6 s per policy, 36 s per paper |

A full pack rebuild is one command per genre; the builder skips every stage whose inputs, settings and model ids are
unchanged, and a stale GPU stage on a machine without CUDA stops and names its Colab job instead of running for hours:

```bash
python -m experiments.run --config configs/pack_privacy.yaml --name pack_privacy
```

### The teacher

One `Teacher` interface, several backends, chosen by the compute overlay, never by code ([models/teacher.py](models/teacher.py)):

| Backend | Overlay | Model | Use |
|---|---|---|---|
| `mlx` | `local` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | prompt iteration, coverage checks |
| `openai` (llama.cpp server) | `lab` | `Qwen2.5-32B-Instruct` Q4_K_M, one server per GPU | the recorded schema-induction pass (decision record 002) |
| `openai` (OpenRouter) | `openrouter` | `qwen/qwen-2.5-72b-instruct`, provider pinned | teacher labels, cluster naming, the open-model baseline |
| `anthropic` | `judge_anthropic`, `llm_gap_anthropic` | Claude Haiku 5.5, Claude Opus 5.5 | silver reference labels; the frontier baseline |
| `vllm` | `gpu` | `Qwen/Qwen2.5-72B-Instruct-AWQ` | a CUDA server with vLLM, if one is available |
| `transformers` | `smoke` | `Qwen/Qwen2.5-1.5B-Instruct` | tests and smoke |

All backends return validated pydantic objects and cache every response on disk (`models.teacher.cache_dir`), so a rerun
is free and byte-identical. Prompts live in `prompts/*.md` with a version header; their hashes are logged per run. The
teacher is never on the per-query path: the demo calls no API.

## The demo

```bash
python -m experiments.run --config configs/serve.yaml
```

Opens http://127.0.0.1:7860: paste or upload a document, or pick one of the pinned examples (two corpus documents and
four OmitBench instances with the recorded chat-model answers, `app/examples/`, rebuilt by `python -m app.examples`
when tables or runs change). Ghost paragraphs mark the ranked absences where peers usually make the statement; a click
opens the evidence panel; the cohort panel re-ranks by k, category and date window without re-embedding; the comparison
toggle shows the recorded answers of Claude Opus 5.5 and Qwen2.5-72B for a benchmark document. The About tab's numbers
come from `reports/tables/` through `app/examples/results.json`, never typed.

`python -m app.space_bundle` assembles the Hugging Face Space in `space/` (gitignored): the code, the configs, the
examples and the subset of each pack the app reads, with no corpus. It copies; uploading is a separate, deliberate step.

## The report

Every number in the report comes from `reports/tables/*.csv`:

```bash
python -m analysis.tables && python -m analysis.figures && python -m analysis.report
```

`analysis/tables.py` writes the table fragments and `values.json`; `reports/report_source.md` holds the prose with
value, fragment and citation placeholders; `analysis/report.py` assembles `reports/report.md` (`--check` fails when it
is stale, `--words` counts the prose; a test keeps the committed report in step). Figures go to `reports/figures/`.

## Compute

Recorded with `python -m experiments.run --device-info` (plan §0.6).

| Target | Hardware | Role in the lean tier |
|---|---|---|
| Local | Apple M5, 10 cores, 24 GiB unified memory, macOS 27, torch 2.13.0, MPS | primary: data build, heads, index, bench scoring, demo, tests, smoke |
| UOWD lab machine | Windows Server 2022, 4 × RTX 5000 Ada 32 GiB, torch 2.13.0+cu126; no vLLM | schema extraction (llama.cpp), corpus embeddings; `lab_session.py` (decision record 002, plan Appendix D) |
| Google Colab Pro | A100 40 GB | the long presence and bench stages (`configs/colab_jobs.yaml`, `notebooks/`) |
| OpenRouter, Anthropic API | API | the 72B teacher from home; silver labels; the frontier baseline (about $20 of API spend for the whole class project, CHANGELOG) |
| Hugging Face Space | `cpu-basic`, 2 vCPU, 16 GB | the public demo |

## Models

All checked on the Hugging Face hub (CHANGELOG 2026-10-12, 2026-10-18).

| Role | Lean (default) | Full (`--overlay full`) | Smoke |
|---|---|---|---|
| Teacher | `Qwen/Qwen2.5-72B-Instruct` (recorded pass: `Qwen2.5-32B-Instruct` Q4_K_M, decision 002) | same | `Qwen/Qwen2.5-1.5B-Instruct` |
| Embedder | `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0) | `Qwen/Qwen3-Embedding-8B` | `BAAI/bge-small-en-v1.5` |
| Late interaction | `answerdotai/answerai-colbert-small-v1` (Apache-2.0) | `jinaai/jina-colbert-v2` (CC-BY-NC) | same as lean |
| Reranker (ablation only in the lean tier) | `BAAI/bge-reranker-v2-m3` | same | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Presence classifier | calibrated logistic heads on the embedder, teacher labels (decision 005) | `answerdotai/ModernBERT-large` fine-tune | heads |
| Frontier baseline | Claude Opus 5.5 (Anthropic API), Qwen2.5-72B (OpenRouter) | same | 2 instances |

The lean tier is Apache-2.0/MIT throughout. OPP-115's annotations (non-commercial) are a reported human check, not
training data, since decision 005. The full tier's CC-BY-NC models are fine for the class and the paper, not for the
product layer.

## Layout

The repository layout is plan §0.4: `data/` (loaders, segmentation, schema), `packs/` (schemas and the pack builder),
`engine/` (embedding, presence, cohort, expectation, ranking, evidence, explanation), `models/` (the teacher), `bench/`
(OmitBench), `experiments/` (the entry point, one module per task, the hub sync), `app/` (Gradio demo, examples, Space
bundle), `analysis/` (tables, figures, report), `configs/`, `prompts/`, `tests/`, `reports/`. Corpora, pack artefacts,
checkpoints, caches and run folders are gitignored; `runs/log.csv` is tracked.
