---
id: induce_coverage
version: 1
description: >
  Schema review (plan §3.5): list what one held-out section asserts and map each assertion to one type of the
  candidate schema, or to "none" when no type fits. The share of sections with at least one "none" is the coverage
  gate (< 5%, §3.6). The answer's type field is constrained to the schema's ids plus "none".
---
You are checking whether a catalogue of statement types covers what documents of one genre say. The genre is:
{{document_kind}}. Statements are about {{author_role}}.

The catalogue (type id: definition):
{{type_catalogue}}

Read the section below. List each distinct statement it makes (at most {{max_assertions}}), and for each one give
the id of the catalogue type it is an instance of. Use "none" only when no type in the catalogue fits the
statement, not when the fit is loose: a statement that is a narrower case of a type belongs to that type.
Ignore navigation text and headings on their own. Everything else the section states counts, including
boilerplate such as copyright lines or product disclaimers: map it to a type or mark it "none".

For each statement return:
- "claim": one sentence saying what the section states, in general terms.
- "type": the catalogue id, or "none".

Section heading: {{section_heading}}
Section text:
"""
{{section_text}}
"""

Return only the JSON object {"assertions": [...]}.
