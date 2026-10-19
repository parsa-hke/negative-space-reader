---
id: explain
version: 1
description: >
  Teacher explanation for one reported absence (plan §0.2 #14, task 5.8; explain.mode: teacher, opt-in): a one-line
  rationale and a suggested wording, written only from the peer sentences given. The ethics stance of §0.2 #21
  holds: prevalence among peers, never a verdict ("non-compliant", "missing", "deficient").
---
You help the author of a {{document_kind}} see what comparable documents usually say. Statements are about
{{author_role}}.

Among {{n}} comparable documents, {{count}} ({{pct}}%) make a statement of this kind, and this document does not:
"{{type_label}}": {{definition}}

Sentences these comparable documents use (quoted exactly, with their document ids):
{{quotes}}

Write two things, using only what the quoted sentences say. Do not add facts, numbers, laws, periods or practices
that are not in the quotes, and do not judge the document (no "missing", "non-compliant", "should", "fails").
- "rationale": one sentence saying what peers typically state here, in general terms.
- "suggested_wording": one or two sentences an author could adapt, built only from the quoted sentences, with
  any specifics left as placeholders in square brackets (for example [period]).

Return only the JSON object {"rationale": "...", "suggested_wording": "..."}.
