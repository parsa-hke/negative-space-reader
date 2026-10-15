---
id: induce_coverage
version: 3
description: >
  Schema review (plan §3.5), first pass: list what one held-out section states and map each statement to one type of
  the candidate schema, or to "none" when no type fits. Every "none" then gets a second look (induce_coverage_verify).
  v3 = v1's two outcomes plus v2's explicit mapping rules (narrower cases and negative forms belong to the type);
  v2's "background_or_filler" label was withdrawn: on papers it absorbed real background statements.
---
You are checking whether a catalogue of statement types covers what documents of one genre say. The genre is:
{{document_kind}}. Statements are about {{author_role}}.

The catalogue (type id: definition):
{{type_catalogue}}

Read the section below. List each distinct statement it makes (at most {{max_assertions}}), and for each one give
the id of the catalogue type it is an instance of. A statement that is a narrower case of a type, or a negative form
of it ("does not ..."), belongs to that type. Use "none" only when no type in the catalogue fits the statement, not
even loosely. Ignore navigation text and headings on their own; everything else the section states counts.

For each statement return:
- "claim": one sentence saying what the section states, in general terms.
- "type": the catalogue id, or "none".

Section heading: {{section_heading}}
Section text:
"""
{{section_text}}
"""

Return only the JSON object {"assertions": [...]}.
