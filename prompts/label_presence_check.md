---
id: label_presence_check
version: 2
description: >
  Presence labels (plan §4.1), second look: the section again, with only the few types most similar to it (by
  embedding of the section and of its first-pass statements) that the first pass did not find, each with its
  definition and examples. The teacher says which of them the section states, with a verbatim span. Counters the
  first pass missing types in a 50-type catalogue (3.5: about one statement in 30).
  v2: shows the types already found and counts a candidate only when the section explicitly states it; v1 added
  22-40% of labels at about 60% precision in the 72B trial, mostly by reusing a passage already labelled with a
  better-fitting type (CHANGELOG 2026-10-15).
---
A section of a {{document_kind}} is shown below (heading: {{section_heading}}). Statements are about
{{author_role}}.

Section text:
"""
{{section_text}}
"""

This section has already been labelled with these types:
{{found}}

Candidate types, each with its definition and examples of statements that belong to it:
{{candidates}}

Which candidate types does the section also state? Count a candidate only if the section explicitly states what its
definition describes: an instance of it, a narrower case of it, or a negative form of it ("does not ..."), even if
the wording differs from the examples. Do not count a candidate because the section touches on its topic, because it
could be inferred, or because a passage that is already covered by one of the types above is loosely related to it.
A passage can count for a candidate as well only if it clearly states that candidate's content too. For each
candidate stated, copy the shortest passage from the section, word for word, that shows it. Return an empty list
if the section states none of them.

Return only the JSON object {"present": [{"type": "<candidate id>", "span": "<passage>"}, ...]}.
