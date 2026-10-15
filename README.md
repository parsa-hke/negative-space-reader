# Negative-Space Reader (`negspace`)

Detecting what a document *conspicuously omits*, by modelling what its genre peers assert.
MAAI 902 Assessment 2 (NLP Solution Design) → paper → product. Parsa Hosseini, UOWD, 2026.

**Starting a session? Read [HANDOFF.md](HANDOFF.md) first** (current state, artefact locations, open decisions). The working plan is [NEGATIVE_SPACE_PLAN.md](NEGATIVE_SPACE_PLAN.md); progress and findings are in [CHANGELOG.md](CHANGELOG.md); design decisions written for the report are in [reports/decisions/](reports/decisions/).
Status: **Phases 1 and 2 done; task 3.1 iterated locally** (72B confirmation pending, plan Appendix D); architecture revised to the lean/full tier design (decision 001). Modules from Phase 3 on are stubs until their task.

## The two tiers

Every swappable component is a config key with a **lean** default and a **full** alternative (plan §0.2 #25). The engine, benchmark and MAM are the same in both; only the models plugged in differ.

| Component | Lean (default) | Full (`--overlay full`) |
|---|---|---|
| Embedder | `Qwen/Qwen3-Embedding-0.6B` | `Qwen/Qwen3-Embedding-8B` |
| Late interaction | `answerdotai/answerai-colbert-small-v1` | `jinaai/jina-colbert-v2` (CC-BY-NC) |
| Presence classifier | logistic heads on frozen embeddings, labels from OPP-115 + ~2k teacher labels (`presence.head: linear`, `setfit`) | ModernBERT-large fine-tune on 5k teacher labels (`finetune`) |
| Explanations | template from evidence (`explain.mode: template`) | teacher rationale, opt-in (`teacher`) |
| Real-omission benchmark | OmitBench-revision + 50-doc validation (`bench.real.source: revision`) | 150 × 2 blind annotation (`annotation`) |
| Corpus | 20–30k documents per genre, both tiers | |

The lean tier runs on the Mac overnight and costs about $10 in API calls for the class (the frontier-model baseline). Full details, rationale and trade-offs: [reports/decisions/001-lean-architecture.md](reports/decisions/001-lean-architecture.md).

## Setup

Python 3.11 is required (pinned in `pyproject.toml`).

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .
```

Optional extras:

| Extra | Where | What |
|---|---|---|
| `.[mac]` | Apple Silicon | `mlx-lm` for the local teacher (Qwen2.5-7B-Instruct 4-bit) |
| `.[gpu]` | Linux + CUDA | vLLM (pins `torch==2.13.0`, which matches ours) |
| `.[tracking]` | anywhere | wandb |

`requirements.txt` is the full transitive freeze of the local environment, kept for the record.

## Running things

Every experiment goes through one entry point with a config from `configs/`:

```bash
python -m experiments.run --device-info
```

```bash
python -m experiments.run --config configs/induce.yaml --overlay local --overlay smoke --set pack=papers
```

- `--overlay NAME` applies `configs/NAME.yaml` on top (repeatable, in order; put the compute overlay first and `smoke` last).
  - **Compute:** `local` (Mac, MPS, MLX teacher), `gpu` (server, vLLM teacher), `openrouter` (72B teacher through OpenRouter; needs `OPENROUTER_API_KEY` in the environment).
  - **Mode:** `smoke` (200 documents per genre, tiny models, one seed), `full` (the full tier).
- `--set dotted.key=value` overrides one key (value parsed as YAML). Unknown keys are rejected; prefix with `+` to add one.
- `--seed N` limits the run to seed N; otherwise every seed in `run.seeds` runs.
- `--print-config` prints the resolved config and exits.
- Tasks: `build_pack | induce | distil | train_mam | bench | baselines | drift | serve`.
- Each (config, seed) writes `runs/<name>_s<seed>/{config.yaml,metrics.json}` and appends a line to `runs/log.csv` (failed and interrupted runs too). The line records the tier, config hash, git commit, device, the version and hash of every prompt rendered, and which model build served every teacher call.

### The teacher

One `Teacher` interface, four backends, chosen by the compute overlay, never by code ([models/teacher.py](models/teacher.py)):

| Backend | Overlay | Model | Use |
|---|---|---|---|
| `mlx` | `local` | `mlx-community/Qwen2.5-7B-Instruct-4bit`, 24 tok/s on the M5 | prompt iteration, coverage checks |
| `vllm` | `gpu` | `Qwen/Qwen2.5-72B-Instruct-AWQ` | the recorded extraction pass, genre-B labels |
| `openai` | `openrouter` | `qwen/qwen-2.5-72b-instruct`, provider pinned to `deepinfra/fp8`, fallbacks off | the same pass from home, ≈ $13 |
| `transformers` | `smoke` | `Qwen/Qwen2.5-1.5B-Instruct` | tests |

All backends return validated pydantic objects from `generate_structured(prompt, Model)` and cache every response on disk (`models.teacher.cache_dir`), so a rerun is free and byte-identical. Prompts live in `prompts/*.md` with a version header; their hashes are logged per run.

## Data (Phase 2)

Sources, licences and the real field names are in [data/README.md](data/README.md). Raw files go in `data/raw/<source>/` (about 57 GB in total, mostly the Princeton SQLite). Download commands and URLs are listed there.

Build both genres (about 7 minutes on the M5 plus the arXiv e-print fetch, which is cached and rate-limited):

```bash
python -m experiments.run --config configs/data.yaml --overlay local
```

Smoke build (200 documents per genre, writes to `data/processed_smoke/`):

```bash
python -m experiments.run --config configs/data.yaml --overlay local --overlay smoke
```

Outputs per genre in `data/processed/<genre>/`: `documents.jsonl`, `sections.jsonl`, `sentences.jsonl`, `revisions.jsonl` with `revisions/` (the version texts), and for privacy `opp115_labels.jsonl`. Split manifests with the sealed test hash are in `data/processed/splits/`; tables in `reports/tables/` (`corpus_stats`, `corpus_sample`, `dedup`, `revisions`); loader previews in `reports/figures/loader_check_<source>.html`.

| | privacy | papers |
|---|---|---|
| documents (train / dev / test) | 28,440 (22,300 / 3,043 / 3,097) | 17,644 (14,114 / 1,784 / 1,746) |
| sections / sentences | 248k / 1.7M | 373k / 5.0M |
| years | 1998–2019 | 2018–2022 |

## Tests

```bash
.venv/bin/python -m pytest
```

About 15 s on the M5 (plus the two model tests). Two tests load real models and are skipped if the weights are not in the Hugging Face cache:

```bash
.venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct
```

```bash
.venv/bin/hf download mlx-community/Qwen2.5-7B-Instruct-4bit
```

## Compute

Recorded with `python -m experiments.run --device-info` (plan §0.6).

| Target | Hardware | Role in the lean tier |
|---|---|---|
| Local | Apple M5, 10 cores, 24.0 GiB unified memory (17.8 GiB usable by MPS), macOS 27.0.1, torch 2.13.0, MPS, no CUDA | primary: data build, embeddings, presence heads, index, MAM, bench, demo; 7B teacher for iteration |
| UOWD lab machine | Windows Server 2022, 4 × RTX 5000 Ada 32 GiB (128 GiB), 112 cores, 1 TB RAM, driver 581.42, torch 2.13.0+cu126; no vLLM | teacher via llama.cpp (Qwen2.5-32B Q4_K_M, one server per GPU), corpus embeddings; `lab_session.py` (decision record 002) |
| OpenRouter | API | the 72B teacher from home; frontier baselines |
| Google Colab Pro | A100 40 GB or L4 24 GB, varies | fallback; CUDA rehearsal for the vLLM path |

**Teacher (plan §0.2 #6): `Qwen/Qwen2.5-72B-Instruct`.** Precision on the server depends on its VRAM: bf16 with tensor parallelism on ≥ 4×80 GB (145 GB of weights); `-AWQ` (41.6 GB) on 1×80 GB or ≥ 2×48 GB (the `gpu.yaml` default); `Qwen/Qwen2.5-32B-Instruct-AWQ` below 48 GB.

## Models

All checked on the Hugging Face hub on 2026-10-12. Weight sizes are the safetensors/bin totals.

| Role | Lean | Full | Smoke |
|---|---|---|---|
| Teacher | `Qwen/Qwen2.5-72B-Instruct` (Qwen licence; 145 GB / AWQ 41.6 GB); local proxy `mlx-community/Qwen2.5-7B-Instruct-4bit` | same | `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) |
| Teacher fallback | `Qwen/Qwen2.5-32B-Instruct` (Apache-2.0; 65.5 GB / AWQ 19.3 GB) | same | |
| Embedder | `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0, 1.2 GB); fallback `BAAI/bge-m3` (MIT) | `Qwen/Qwen3-Embedding-8B` (Apache-2.0, 15.1 GB); fallback `nvidia/NV-Embed-v2` (**CC-BY-NC**) | `BAAI/bge-small-en-v1.5` |
| Late interaction | `answerdotai/answerai-colbert-small-v1` (Apache-2.0) | `jinaai/jina-colbert-v2` (**CC-BY-NC**) | same as lean |
| Reranker | `BAAI/bge-reranker-v2-m3` (Apache-2.0); upgrade candidate `Qwen/Qwen3-Reranker-8B` | same | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Presence classifier | heads on the embedder | `answerdotai/ModernBERT-large` (Apache-2.0); fallback `microsoft/deberta-v3-large` (MIT) | `answerdotai/ModernBERT-base` |

The lean tier is Apache-2.0/MIT throughout. The full tier's two CC-BY-NC models are fine for the class and the paper but cannot ship in the product layer (Phase 12) without a commercial licence or a swap.

## Layout

The repository layout is plan §0.4. `experiments/config.py` (config loading) was added alongside `run.py`. Corpora, pack artefacts (prototypes, classifier, index), checkpoints, caches and run directories are gitignored. Of `runs/`, only `runs/log.csv` is tracked.
