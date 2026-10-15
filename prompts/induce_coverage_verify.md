---
id: induce_coverage_verify
version: 1
description: >
  Schema review (plan §3.5), second look: one statement the first pass mapped to "none", shown with only the
  candidate types most similar to it (by embedding), each with its definition and examples. The teacher picks the
  type it belongs to or confirms "none". Counters the first pass missing types in a long catalogue.
---
A statement was taken from a {{document_kind}} (section heading: {{section_heading}}). It is about {{author_role}}.

Statement: {{statement}}

Candidate types, each with its definition and examples of statements that belong to it:
{{candidates}}

Does the statement belong to one of these types? It belongs if it is an instance of the type, a narrower case of
it, or a negative form of it ("does not ..."), even if its wording differs from the examples. Answer "none" only
if none of the types fits.

Return only the JSON object {"type": "<one candidate id, or none>"}.
