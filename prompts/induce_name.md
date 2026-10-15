---
id: induce_name
version: 2
description: >
  Schema induction step 2 (plan §3.3): name one cluster of extracted claims as an assertion type of the genre,
  define it in one sentence, pick three representative claims, and say whether the cluster mixes kinds of
  statement (input to the human pass, §3.4). No few-shot examples, so no example labels can leak into the names.
---
You are helping to build a catalogue of the kinds of statements that documents of one genre make. The genre is:
{{document_kind}}. Each claim below was extracted from one section of such a document and is written about
{{author_role}}.

The claims below are {{n_shown}} of {{cluster_size}} similar claims that were grouped together automatically. The
group's claims come from {{n_documents}} different documents. The first claims are the most typical of the group;
the rest are a random sample of it.

Claims:
{{claims}}

Labels the extraction step gave to the group's claims, most frequent first (rough hints only, often too specific):
{{type_hints}}

Describe the kind of statement this group makes, as it would be recognised in any document of this genre:
- "name": a snake_case label of 2 to 5 words for the kind of statement. Name what the statement does or discloses,
  not the subject matter of one document. Other groups cover neighbouring kinds of statement, so make the name
  specific enough to tell this group apart: say which kind of party, data, purpose, right, condition, evidence or
  part of the work is involved, how something happens, or that something is NOT done, when the claims agree on it.
  Prefer the plain words that the claims themselves use.
- "definition": one sentence that lets a reader decide whether a new statement belongs to this kind. Start with a
  verb whose subject is the document (for example "States that ...", "Describes ...", "Reports ..."). Write it
  about {{author_role}} in general terms; say what is included and, if the group has a clear edge, what is not.
- "representative": the numbers of the three claims that best show this kind of statement.
- "single_kind": true if all or nearly all claims make the same kind of statement; false if the group mixes two
  or more different kinds that a reader would want to keep apart.
- "split_note": if single_kind is false, the kinds it mixes in a few words each; otherwise an empty string.

Return only the JSON object.
