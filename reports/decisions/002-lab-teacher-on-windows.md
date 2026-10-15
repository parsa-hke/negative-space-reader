# Decision record 002: serving the teacher on the Windows lab machine

**Date:** 2026-10-12 · **Status:** adopted · **Plan sections changed:** §0.2 #6, §0.6, Appendix D.

## Context

The first lab session revealed the hardware: a Windows workstation with **4 × NVIDIA RTX 5000 Ada Generation
(32 GB each, 128 GB total)**, driver 581.42 (CUDA 13.0), 544 GB free disk, no git, and no administrator rights. The
plan assumed a Linux server running vLLM. The session is a single day; jobs cannot run overnight. Schema induction
(task 3.2) needs about 15,400 teacher calls (500 dev documents per genre, every section), roughly 5–6 million
generated tokens.

## Decision

- Serve the teacher with **llama.cpp** (build b11280, Windows CUDA 12.4 binaries; no installation or admin rights),
  through its OpenAI-compatible server, which enforces the JSON schema of the extraction output (verified with the
  same build before the session).
- Run the bulk extraction with **Qwen2.5-32B-Instruct, Q4_K_M** (19.9 GB), the plan's named fallback teacher:
  **one server per GPU**, 8 parallel slots each, requests spread over the four servers.
- Confirm the extraction prompt (task 3.1) on **the 32B itself**, since it is the model that produces the record.
- Use the same session for the corpus embeddings (Qwen3-Embedding-0.6B on all four GPUs) and, time permitting, the
  8B-embedder ablation, so that no GPU work remains for later phases.

## Options considered

| Option | Why not |
|---|---|
| vLLM (as planned) | Does not run on native Windows; WSL needs administrator rights. |
| Qwen2.5-72B, Q4 (44 GB), split across GPUs | Fits, but one model spread over 2–4 consumer GPUs is pipeline-bound: an estimated ~150 generated tokens/s in total, i.e. 10+ hours for the extraction. Not possible in a day. |
| Transformers + 4-bit quantisation | Slower still for batched generation; a 145 GB download. |
| The 72B through OpenRouter | ≈ $15 and the same model as planned; kept as the fallback if the 32B fails the eye test. |

## Consequences

- The schema is induced from 32B claims. Every claim records the serving build
  (`llama.cpp-b11280/Qwen/Qwen2.5-32B-Instruct-GGUF/q4_k_m`), so this is visible in all downstream artefacts.
- Quality risk is measured, not assumed: the 32B eye test is reviewed against the same pass criteria as the 72B
  would have been. A cheap cross-check against the 72B (≈ $0.05 via OpenRouter, 40 sections) can be run from home.
- Throughput depends on llama.cpp batching on consumer GPUs; the extraction prints an ETA, processes documents in a
  fixed pseudo-random order (an early stop leaves a random subset) and resumes from the response cache.
- After this session the remaining work is laptop-sized; the only teacher calls left (schema coverage check and
  genre-B presence labels, ~2,200 calls) depend on the schema written at home and can use OpenRouter (≈ $3).
