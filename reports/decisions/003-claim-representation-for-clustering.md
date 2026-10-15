# Decision record 003: how claims are represented for clustering

**Date:** 2026-10-13 · **Status:** adopted · **Plan sections changed:** none (§0.2 #5 stays "embed → cluster with
HDBSCAN"; this record fixes *how* claims are embedded and reduced, Phase 3.3).

## Context

Schema induction clusters the claims that the teacher extracted from 500 privacy policies (21,046 claims) and 303
research papers (44,367 claims), so that each cluster can become one assertion type. The plan targets 60–120
clusters per genre before the human pass. Two problems surfaced in the first runs, both measured on the same claims
with Qwen3-Embedding-0.6B and HDBSCAN.

1. **Density.** Reducing the 1024-dimensional embeddings with PCA to 32 dimensions left **68–80% of privacy claims
   and 92–95% of paper claims** as HDBSCAN noise across 24 parameter settings per genre, and no papers setting
   reached 60 clusters (run `cluster_pca32_s0`). A schema induced from a quarter of the claims would miss the rarer
   assertion types, which are the ones omission detection most needs.
2. **Topic instead of genre.** With UMAP (10 dimensions) the noise fell to 17% for privacy and 39% for papers
   (run `cluster_umap10_s0`), and the privacy clusters read as genre types (sharing with third parties, children
   under 13, opt-out, retention). The paper clusters did not: most were research topics (federated learning, crop
   phenology from remote sensing, EEG data), several drawn from fewer than ten papers. The extraction prompt asks for
   the *move* ("the authors describe …") and the claims are phrased that way, but the embedding of a claim is
   dominated by what follows the verb. Clustering those vectors would have induced a topic taxonomy for genre B,
   which the omission task cannot use: no paper is expected to be about federated learning.

## Decision

Claims are clustered in a representation built in three generic steps, none of which names a genre:

1. **Instruction-aware embedding.** Each claim is embedded as a query under the instruction "Given a statement
   extracted from a {document kind}, identify the kind of statement it is (what the document does or discloses),
   ignoring its specific subject matter" (`prompts/induce_cluster_embed.md`, versioned and hashed like every
   prompt; the document kind comes from the genre's configuration).
2. **Within-document centring.** Each claim vector has its document's mean claim vector subtracted
   (`induce.cluster.center_by_document: 1.0`) and is re-normalised. Whatever all claims of one document share is
   its subject matter; the kind of statement varies within a document. Removing the shared part keeps topic from
   deciding the clusters.
3. **UMAP to 10 dimensions** (cosine, 15 neighbours, `min_dist` 0) before HDBSCAN, whose minimum cluster size and
   minimum samples are swept; the setting with 60–120 clusters, no cluster above 10% of the claims, and the least
   noise is kept. Every setting tried is recorded in the run's metrics.

## Evidence

Papers, UMAP-10, HDBSCAN minimum cluster size 120, measured on the same 44,367 claims. "Spread" is the
claim-weighted mean share of documents a claim's cluster draws from (a genre type recurs across documents; a topic
does not).

| Representation | Clusters | Noise | Spread | Clusters in < 5% of documents |
|---|---|---|---|---|
| plain embedding | 91 | 39% | 0.19 | 23 |
| + centring | 74 | 42% | 0.30 | 5 |
| + instruction | 62 | 41% | 0.42 | 3 |
| + instruction + centring | 57 | 38% | 0.58 | 1 |

With both steps, the paper clusters are the moves a reviewer checks for: hyperparameters (217 of 303 papers),
datasets, results against baselines, limitations (186), future work (229), code availability (168), data splits,
hardware and software used, runs repeated over seeds, ablations, funding, acknowledgements. A few topical clusters
remain (each drawn from a small share of documents) and are flagged for the human pass. On privacy policies,
centring changed nothing measurable (17.7% vs 17.8% noise, spread 0.356 vs 0.359), as expected for a genre with
little topical variation, so the same pipeline serves both genres.

## Options considered

| Option | Why not |
|---|---|
| PCA before HDBSCAN | 68–95% noise; papers never reach 60 clusters. |
| Cluster the extraction labels (`type_guess`) instead of claims | The labels are inconsistent across documents ("data_sharing", "third_party_sharing", "disclosing_practice") and some copy the prompt's examples; they stay a naming hint. |
| Re-extract paper claims with a stricter prompt | A new extraction pass costs a lab session or ≈ $15 and the 32B already follows the "move" instruction in its wording; the topic sits in the embedding, not in the claim. |
| Drop clusters drawn from few documents automatically | Kept as a flag (document share) for the human pass rather than a rule: a rare but real genre type would be lost silently. |

## Consequences

- A new pinned dependency, `umap-learn` 0.5.12 (with `pynndescent`, `numba`, `llvmlite`).
- Claim vectors are stored per instruction version (`data/embeddings/<model>/<genre>/claims__<prompt>_v<n>_<hash>/`),
  so a change of instruction never mixes vectors.
- Noise remains substantial for papers (≈ 40%). Noise claims are not discarded: the review workbook samples them so
  the human pass can spot a recurring statement with no cluster, and the coverage check (3.5) measures what the
  frozen schema misses on unseen sections.
- The representation is part of the pack's provenance (the schema draft records embedder, instruction hash,
  centring, reduction and HDBSCAN setting) and is reusable for a third genre without code changes.
