# 7.2 — Walkthrough video plan (3–5 minutes, recorded on the Space)

Written 2026-10-25. The video shows the live demo at https://huggingface.co/spaces/parsa-hke/negative-space-reader
on one privacy policy, one paper, and ends with a benchmark reveal. Every example below is pinned in
`configs/base.yaml` (`serve.examples`, `serve.bench_examples`) and shipped as `app/examples/*.json`, so the Space shows
exactly these names in its Example dropdown, in this order: the corpus policy, two benchmark policies, the corpus paper,
two benchmark papers.

## Before pressing record

- The Space runs on `cpu-basic` and **sleeps after 48 h idle**: open it 5–10 minutes early, click *Restart this Space*
  if it is asleep, run one analysis so the models are loaded (the first call also warms the embedder).
- Timings on cpu-basic (CHANGELOG 2026-10-21): the Etilize policy ≈ 10 s, the FANS paper ≈ 36 s. Script the narration
  over these waits; do not cut them, they are honest.
- Browser at 1280 × 800 or 1440 × 900, zoom 110–125 % so the ghost paragraphs read on a phone; bookmarks bar hidden;
  one tab. QuickTime Player ▸ File ▸ New Screen Recording (built-in mic) is enough; export at 1080p.
- Nothing that calls an API is on the Space (`serve.allow_teacher` and `serve.allow_live_chat` are false): the chat-model
  panel shows the cached 7.3 answers only. Say so on camera when it appears.
- Rule of the voice-over: counts and quotes, never a verdict. "Peers like this one usually state X; this document does
  not", never "this policy is deficient".

## Shot list

| Time | On screen | Say (gist) |
|---|---|---|
| 0:00–0:25 | The empty state (sample ghost paragraph) and the five guide steps | A building inspector notices the missing smoke alarm without a checklist, from the memory of every other hallway. A document has the same property: an omission leaves no sentence to search for. Negative-Space Reader borrows the expectation from a document's 200 closest peers and shows what they state and this one does not, with their own sentences as evidence. |
| 0:25–1:25 | **Example: Corpus · privacy: Etilize – Privacy Policy** → Analyse (≈ 10 s) | While it runs: the document is split into sections, each section is scored for the 58 kinds of statement in the privacy catalogue, and its 200 nearest policies are found in an index of 22,300. Then read the status line aloud (sections read, peers, mean similarity, "states N of 58 kinds"). Scroll to the first ghost: "N of 200 peers state this; this document does not", placed under the heading where peers say it. |
| 1:25–1:50 | Click the first ghost → evidence panel | The count behind the number, the 95 % interval, and up to five peer sentences with their document ids. This is the whole ethical stance: every claim is a count of peers and a quote from them. |
| 1:50–2:45 | **Example: Corpus · papers: FANS: Fusing ASR and NLU for on-device SLU** → Analyse (≈ 36 s) | Same engine, different pack: 47 kinds of statement for cs.CL/cs.LG papers, 14,114 indexed. During the wait, open the cohort panel: k, the category filter, the date window; changing them re-ranks without re-embedding (instant). Read one ghost (a reproducibility statement peers make and this paper does not) and its evidence. |
| 2:45–4:15 | **Example: Benchmark · papers: MarIA: Spanish Language Models** → Analyse → read the *Challenge* line → turn on *Compare with a chat model* | The benchmark took a real paper and removed every sentence making five kinds of statement. The ranked ghosts are our answer. The reveal badges the ghosts that were in fact removed: three of the five are in our top five (software and libraries at rank 1, model architecture, compute resources). The panel sets our list beside Claude Opus 5.5 (two of the five in its top five, first hit at rank 2) and Qwen2.5-72B (none in its top five, first hit at rank 6), from recorded runs, not live calls. |
| 4:15–4:45 | About tab, results section | One document proves nothing, so the benchmark has 100 instances per genre: at recall@5 the peer model is at 0.655 for papers against 0.235 for Opus and 0.181 for Qwen; for policies 0.376 against 0.242 and 0.137. The report explains why peers help less for policies (the deletion moves a document towards peers that omit the same thing). |
| 4:45–5:00 | Back to the document, the first ghost | Close: prevalence with evidence, never a verdict; the code, the schemas and the benchmark are in the repository. |

Numbers in the narration are those of `reports/tables/llm_gap_candidate_<genre>.csv` and `bench_synth_<genre>.csv`,
as shown on the Space's About tab; read them from the screen rather than from memory, so the video and the tables
cannot disagree.

## Why these examples

- **Etilize policy (`privacy_pl:36309`) and FANS paper (`unarxive:2111.00400`):** the two documents of the 7.1 build
  and the cpu-basic timing runs (`space_cpu_timing_<genre>_s0`); mid-length (6–30 sections); the first randomly drawn
  privacy example was an adult site, which is why the corpus examples are pinned by id.
- **MarIA (`single_type:unarxive:2107.07253`) as the ending:** of the four pinned benchmark instances it is the one where
  the recorded ranks tell the paper's story on a single document: level-1 recall@5 0.6 (all five removed types within
  the top ten), Opus 0.4, Qwen 0.0 (`runs/bench_eval_v2_s0/papers/per_instance.csv`; `app/examples/papers.json`).
- **Fallback ending, Benchmark · privacy: RSsportscars.com (`single_type:privacy_pl:539904`):** three removed types
  (the policy-change family); level 1 puts two in the top five and all three in the top ten; Opus puts all three in its
  top three; Qwen none. Use it if the video should show a document where the chat model wins, which is also honest.
- Not used: Owens Community College (`privacy_pl:67783`, two removed, level 1 gets one at rank 1) and
  Difficulty-Aware MT Evaluation (`unarxive:2107.14402`, four removed, level 1 two in the top five, Opus one at rank 1).

## After recording

Export as `reports/demo/walkthrough_v0.1-class.mp4` (not committed; large), upload where the subject outline says and
link it from the README's demo section and the submission (see `reports/submission_v0.1-class.md`). Tick 7.2 in the plan
with the date, length and the examples used.
