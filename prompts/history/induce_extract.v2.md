---
id: induce_extract
version: 2
description: >
  Schema induction step 1 (plan §3.1): list the kinds of statements one section of a document makes, as general
  claims with a type guess and a verbatim supporting span. Genre enters only through the placeholders.
  v2 (after the v1 eye test): claims describe the move the document makes, abstracted from subject matter
  (v1 restated topic content for papers); same-kind list items are grouped; spans must be short exact copies.
---
You are reading one section of a {{document_kind}}.

Your job is to list the kinds of statements this section makes, so that the same kind of statement can later be
recognised across many documents of this genre. Treat each assertion as a move the document makes: disclosing a
practice, making a commitment, granting or refusing something, reporting what {{author_role}} did or found,
describing data or method, stating a limitation, or relating the work to other work. Statements that something is
NOT done are moves too.

For each assertion return:
- "claim": one sentence saying what the section asserts, about {{author_role}}, in general terms.
  KEEP what makes it this kind of statement: which kind of party, data, purpose, right, duration, condition,
  evidence, setting or limitation is involved.
  DROP subject-matter detail that would differ between two documents making the same move: names of
  organisations, products, people or places, specific technical topics, model or dataset names, exact numbers.
  When the section explains background knowledge or describes other people's work, write one claim per kind of
  move (for example "{{author_role}} review prior work on the problem and its limitations"), not one claim per fact.
- "type_guess": a snake_case label of 2 to 4 words naming the kind of move (for example "loan_period",
  "external_context"). Give the same label to the same kind of move.
- "span": a short exact copy of words from the section (at most 20 words) that shows the move. Copy character for
  character; do not paraphrase, shorten inside, or fix the text.

Rules:
- Only what the section states. Do not infer, do not add background knowledge.
- One claim per distinct kind of point. Several items of the same kind (for example a list of similar examples or
  of related settings) become one claim; items of different kinds stay separate.
- Ignore navigation text and headings on their own. A section that only gives contact details makes one move:
  it tells the reader how to contact {{author_role}}.
- At most {{max_assertions}} claims. Return an empty list only if the section really asserts nothing.

Examples (from an unrelated genre: a public library's rules and annual report).
Section: "Books may be borrowed for three weeks, DVDs for one week. We do not charge fees for late returns, but
borrowing is suspended until the item is back."
Output:
{"assertions": [
 {"claim": "The library lends items for fixed periods that depend on the type of item.", "type_guess": "loan_period", "span": "Books may be borrowed for three weeks, DVDs for one week."},
 {"claim": "The library does not charge fees for late returns.", "type_guess": "late_fee_policy", "span": "We do not charge fees for late returns"},
 {"claim": "The library suspends borrowing until overdue items are returned.", "type_guess": "overdue_sanction", "span": "borrowing is suspended until the item is back"}
]}
Section: "Like many city libraries across Europe, we abolished fines in 2019. Our survey of 1,200 members found that
returns became faster."
Output:
{"assertions": [
 {"claim": "The library relates its policy to a wider trend among other libraries.", "type_guess": "external_context", "span": "Like many city libraries across Europe"},
 {"claim": "The library reports a past change to its own fee policy.", "type_guess": "policy_change", "span": "we abolished fines in 2019"},
 {"claim": "The library reports survey evidence about the effect of the change.", "type_guess": "evaluation_evidence", "span": "Our survey of 1,200 members found that returns became faster."}
]}

Section heading: {{section_heading}}
Section text:
"""
{{section_text}}
"""

Return only the JSON object {"assertions": [...]}.
