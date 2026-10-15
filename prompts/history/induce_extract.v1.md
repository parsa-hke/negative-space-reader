---
id: induce_extract
version: 1
description: >
  Schema induction step 1 (plan §3.1): list the assertions one section of a document makes, as normalised claims
  with a type guess and a verbatim supporting span. Genre enters only through the placeholders.
---
You are reading one section of a {{document_kind}}.

List every assertion this section makes. An assertion is one distinct point the text states: a practice, a
commitment, a fact, a definition, a limitation, a result, a right, a refusal. Statements that something is NOT done
are assertions too.

For each assertion return:
- "claim": one sentence that paraphrases the point in neutral, general terms. Refer to the document's author as
  "{{author_role}}" and never by name. Do not copy specific names, numbers or product names unless the point is
  about them; write it so the same point made in another document would read the same.
- "type_guess": a short snake_case label (2 to 4 words) for the kind of point, for example "loan_period" or
  "late_fee_policy". Name the kind of statement, not its specific content.
- "span": an exact quote copied from the section text that supports the claim, at most 30 words.

Rules:
- Only what the section states. Do not infer, do not add background knowledge.
- One assertion per distinct point; do not merge separate points into one claim.
- Ignore navigation text, headings on their own, and contact details unless the section is about how to contact.
- At most {{max_assertions}} assertions. If the section asserts nothing, return an empty list.

Example (from an unrelated document, a public library's lending rules):
Section: "Books may be borrowed for three weeks. We do not charge fees for late returns, but borrowing is
suspended until the item is back."
Output:
{"assertions": [
 {"claim": "The library lends books for a fixed period.", "type_guess": "loan_period", "span": "Books may be borrowed for three weeks."},
 {"claim": "The library does not charge fees for late returns.", "type_guess": "late_fee_policy", "span": "We do not charge fees for late returns"},
 {"claim": "The library suspends borrowing until overdue items are returned.", "type_guess": "overdue_sanction", "span": "borrowing is suspended until the item is back"}
]}

Section heading: {{section_heading}}
Section text:
"""
{{section_text}}
"""

Return only the JSON object {"assertions": [...]}.
