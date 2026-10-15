# Prompt history

Superseded or rejected prompt versions, kept byte-for-byte so their hashes (sha256 of the whole file, first 12 hex)
still match the runs that used them. The registry never loads files from this folder.

## induce_extract (plan §3.1)

Eye test: the same 20 dev sections per genre (`prompts/eval/induce_extract_eye_test.json`), local teacher
`mlx-community/Qwen2.5-7B-Instruct-4bit`. Runs in `runs/eye_v<N>_s0/` (review page `eye_test.html`).

| Version | Hash | Run | Spans verbatim (privacy / papers) | Sentence coverage | Verdict |
|---|---|---|---|---|---|
| 1 | `e7457d8b7f02` | `eye_v1_s0` | 95.7% / 89.9% | 59% / 56% | Faithful, but paper claims restate subject matter ("MAML is used for cross-lingual NER"): would induce topic clusters, not genre types. One empty contact section. |
| 2 | `a756e28a41a1` | `eye_v2_s0` | 94.0% / 93.6% | 63% / 56% | **Candidate (current).** Claims are moves ("the authors define an evaluation metric"); contact sections handled. Residual: a list of recipient kinds merged into one claim; long related-work sections still listed fact by fact. |
| 3 | `9c3ede541a2a` | `eye_v3_s0` | 96.4% / 93.7% | 51% / 51% | **Rejected.** Fixed neither residual, and the 7B began copying the examples' labels (`external_context`, `evaluation_evidence`) onto unrelated claims: label leakage that would bias schema induction. |

v2 is confirmed or revised on the 72B teacher in the lab session (plan Appendix D).

## induce_name (task 3.3)

| Version | Hash | Tested on | Verdict |
|---|---|---|---|
| v1 | `ec100a0631e8` | 7B, 10 privacy clusters of `cluster_v1_s0` (run `name_7b_iter1_s0`) | names faithful but generic: two neighbouring collection clusters both `data_collection`; definition style drifts ("Discloses", "This statement discloses", "The statement assures") |
| **v2** | **`4f63706fe4d1`** | same 10 clusters on the 7B (run `name_7b_iter2_s0`) | **kept**: names must say what sets the group apart; definitions start with a verb with the document as subject ("States that ...", "Describes ..."). Definitions now uniform; the 7B still gave two collection clusters the same name (flagged `duplicate_name` for the human pass), left to the 72B |

## induce_coverage (task 3.5)

| Version | Hash | Runs | Verdict |
|---|---|---|---|
| v1 | `d637c0f37fd7` | 7B 10+10 sections (`coverage_7b_iter1_s0`: 40% / 50%, mostly missed mappings), 72B 10+10 (`coverage_72b_iter1_s0`: 10% / 0%), 72B 100+100 (`coverage_72b_s0`: privacy 18%, papers 3%), 72B fresh 100+100 after widening 11 definitions (`coverage_72b_v2_s1`: privacy 17%, papers 1%) | the 7B is too weak for the classification; with the 72B, privacy's remaining misses are a long tail in which explanations of technologies/laws and promotional filler made about half |
