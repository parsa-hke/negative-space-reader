---
id: label_presence_check
version: 1
description: >
  Presence labels (plan §4.1), second look: the section again, with only the few types most similar to it (by
  embedding of the section and of its first-pass statements) that the first pass did not find, each with its
  definition and examples. The teacher says which of them the section states, with a verbatim span. Counters the
  first pass missing types in a 50-type catalogue (3.5: about one statement in 30).
---
A section of a {{document_kind}} is shown below (heading: {{section_heading}}). Statements are about
{{author_role}}.

Section text:
"""
{{section_text}}
"""

Candidate types, each with its definition and examples of statements that belong to it:
{{candidates}}

Which of these types does the section state? A type is stated if the section contains an instance of it, a narrower
case of it, or a negative form of it ("does not ..."), even if the wording differs from the examples. Do not count a
type that the section only touches on without stating it. For each type stated, copy the shortest passage from the
section, word for word, that shows it. Return an empty list if the section states none of them.

Return only the JSON object {"present": [{"type": "<candidate id>", "span": "<passage>"}, ...]}.
