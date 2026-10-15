---
id: induce_cluster_embed
version: 1
description: >
  Schema induction step 2 (plan §3.3): the instruction under which claims are embedded for clustering
  (Qwen3-Embedding's query-side format, "Instruct: ...\nQuery: "). It asks for the kind of statement and away from
  subject matter: with plain embeddings, paper claims clustered by research topic (CHANGELOG 2026-10-13).
  The text that follows "Query: " is the claim.
---
Instruct: Given a statement extracted from a {{document_kind}}, identify the kind of statement it is (what the document does or discloses), ignoring its specific subject matter
Query:
