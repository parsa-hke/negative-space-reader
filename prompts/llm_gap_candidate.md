---
id: llm_gap_candidate
version: 1
description: >
  Baseline (e) of plan §0.2 #17, the 7.3 frontier comparison (candidate-list setting): a chat model reads one
  document and the genre's frozen catalogue of statement types and ranks the types the document does not state,
  most conspicuous first. The same question the system answers from its cohort, asked of the model's own knowledge
  of the genre. The prompt never says that text was removed (the OmitBench-synthetic documents are edited). The
  catalogue comes before the document so that a batch of one genre shares a cached prefix.
---
You will read one {{document_kind}} and a catalogue of the kinds of statement that documents of this genre make.
Statements are about {{author_role}}.

The catalogue (type id: definition):
{{type_catalogue}}

Your task: find the catalogue types that this document does not state, and rank them so that the most
conspicuous absences come first. An absence is conspicuous when comparable documents of this genre usually make a
statement of that kind and this one does not. A type counts as stated when any passage of the document makes a
statement of that kind, even briefly, in other words, or as a narrower or negative case ("we do not ..."); do not
list it then. Return at most {{top_k}} type ids, copied exactly from the catalogue, most conspicuous first.

The document:
<document>
{{document}}
</document>

Return the ids of at most {{top_k}} catalogue types that the document does not state, most conspicuous first.
