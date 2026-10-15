---
id: label_presence
version: 1
description: >
  Presence labels (plan §4.1), first pass: list what one section states and map each statement to one type of the
  frozen schema, or "none", with a verbatim evidence span. The set of types mapped is the section's present set.
  Built on induce_coverage v3 (the 3.5 method) plus the span the plan asks for. A second look
  (label_presence_check) then re-asks about the few types most similar to the section that this pass did not find.
---
You are labelling which kinds of statement a section of a document makes. The genre is: {{document_kind}}.
Statements are about {{author_role}}.

The catalogue of statement types (type id: definition):
{{type_catalogue}}

Read the section below. List each distinct statement it makes (at most {{max_assertions}}), and for each one give
the id of the catalogue type it is an instance of. A statement that is a narrower case of a type, or a negative form
of it ("does not ..."), belongs to that type. When two types could fit, choose the one whose definition fits more
precisely. Use "none" only when no type in the catalogue fits the statement, not even loosely. Ignore navigation
text and headings on their own; everything else the section states counts. Statements the section attributes to
other works or other parties count only as what this document does with them (for example, reviewing prior work).

For each statement return:
- "claim": one sentence saying what the section states, in general terms.
- "type": the catalogue id, or "none".
- "span": the shortest passage copied word for word from the section that shows the statement.

Section heading: {{section_heading}}
Section text:
"""
{{section_text}}
"""

Return only the JSON object {"statements": [...]}.
