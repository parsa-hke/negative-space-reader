---
id: induce_coverage
version: 2
description: >
  Schema review (plan §3.5): list what one held-out section states and map each statement to one type of the
  candidate schema, to "none" when no type fits, or to "background_or_filler" when it only explains general
  background or is promotional filler. Gate: < 5% of sections with a "none" (§3.6, clarified 2026-10-15); the share
  with "none" or "background_or_filler" is reported as the strict rate. v2 adds the third label (v1 counted
  explanations of cookies, P3P, CAN-SPAM and promotional pledges as unmapped: runs coverage_72b_s0, coverage_72b_v2_s1).
---
You are checking whether a catalogue of statement types covers what documents of one genre say. The genre is:
{{document_kind}}. Statements are about {{author_role}}.

The catalogue (type id: definition):
{{type_catalogue}}

Read the section below. List each distinct statement it makes (at most {{max_assertions}}), and for each one give
one of:
- the id of the catalogue type it is an instance of. A statement that is a narrower case of a type, or a negative
  form of it ("does not ..."), belongs to that type;
- "none" when the statement makes a point about {{author_role}} or the document (a practice, commitment, right,
  obligation, condition, finding or other point a reader could look for) and no catalogue type fits it, not even
  loosely;
- "background_or_filler" when the statement only explains general background that is not specific to
  {{author_role}} (for example what a technology, standard or law is in general), or is promotional or courtesy
  language that makes no such point. When unsure between "none" and "background_or_filler", use "none".
Ignore navigation text and headings on their own.

For each statement return:
- "claim": one sentence saying what the section states, in general terms.
- "type": the catalogue id, "none", or "background_or_filler".

Section heading: {{section_heading}}
Section text:
"""
{{section_text}}
"""

Return only the JSON object {"assertions": [...]}.
