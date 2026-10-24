# Negative-Space Reader: detecting what a document conspicuously omits by modelling what its peers assert

Parsa Hosseini · MAAI 902 (NLP Solution Design), Assessment 2 · University of Wollongong in Dubai · October 2026

Demo: https://huggingface.co/spaces/parsa-hke/negative-space-reader

Code: https://github.com/parsa-hke/negative-space-reader

<!-- Every result number in this report is filled from reports/tables/report_fragments/values.json by key (the HTML
     comment after each sentence that carries one names the keys); the tables are the fragments written by
     analysis/tables.py. Nothing numeric was typed by hand. Configuration facts (k = 200, 58 types, 300 reference
     sections, ...) come from the plan and the pack files. -->

## Abstract

We define conspicuous omission as a ranking task: given a document and a corpus of its genre peers, rank the kinds of statement it lacks by how strongly peers like it would be expected to make them, with the peers' own sentences as evidence. One pipeline builds it for privacy policies and computer-science papers; two label-free benchmarks, controlled deletion and the authors' own later revisions, score it. Peers beat genre-wide frequency for papers and not for policies, and we show why: a deletion moves a document towards the peers that omit the same thing. Two frontier chat models given the same catalogue fall well behind both corpus baselines. The demo reports prevalence with evidence, never a verdict.

## 1. Introduction: absence as a signal

A building inspector enters a house for the first time and, within seconds and without a checklist, notices that the hallway has no smoke alarm. Nothing on the ceiling told them so; what told them was the memory of every other hallway, in which an alarm hung in exactly that place. An absence becomes visible only against an expectation, and the expectation is borrowed from peers.

Documents have the same property, and our tools do not. A privacy policy that never says how long it keeps data, or a paper that never reports how its results vary across runs, contains no sentence to search for, embed or classify. The distance between what a document says and what documents like it usually say is not a property of the document alone. It needs the thousand hallways.

A large language model seems to carry the thousand hallways in its weights; Section 7 shows that, asked which statements a document lacks, two capable chat models fall well behind the simplest corpus statistic. That agrees with work showing that language models handle negation poorly [@truong; @hossain] and that retrieval models struggle to separate "states X" from "does not state X" [@weller]. A model trained to predict the next token learns what is present; an absence leaves no token to predict.

This report describes Negative-Space Reader, which makes absence a first-class signal by modelling what a document's genre peers assert. Its research question has three parts: can conspicuous omission be formalised as a ranking task with a label-free evaluation; does an expectation conditioned on the document beat a genre-wide prevalence; and how far behind are frontier chat models at the same task?

Where does the analogy break? The inspector has a building code behind the missing alarm; we have only peers, who encode what the dominant authors of a genre happen to do, including their habits of saying too little. That is why the system never calls a document deficient: it reports what share of similar documents state something, and shows their sentences.

## 2. The task, and what a score may mean

A report needs a scale. A genre pack holds a corpus *C* and a frozen catalogue *T* of assertion types ("states a retention period", "reports computational cost"); a presence detector gives a document *d* a probability per type, and every absent type *t* is scored

score(*t*) = P(*t* expected | *d*, *C*) × (1 − P(*t* present in *d*)) × *w*(*t*),

returning the top ten with their peer counts, up to five peer sentences and a template explanation.

The first factor is the whole idea. A house without a garage is unremarkable in a street of Victorian terraces and conspicuous in a new suburb: the same absence, two expectations, and the neighbourhood supplies the difference. Here the neighbourhood is the cohort, the *k* = 200 indexed documents nearest to *d* in embedding space, same category where the genre has one, never another version of *d*; the expectation, which we call level 1, is the share of the cohort that states *t*, with a Wilson interval [@wilson1927], and its baseline is the genre-wide prevalence of *t*. The second factor discounts types the detector half-sees; the third is an optional severity weight, 1 by default (1.5 for a few privacy types tied to GDPR or CCPA obligations, reported only as an ablation).

The picture has a limit. A neighbourhood is a description, not a standard: if most policies in a category say nothing about retention, the cohort will not expect it. The score ranks; it does not judge. Hence the plan's wording rule: "*N* per cent of peers like this one state *X*; this document does not", never "non-compliant", never "deficient". A second level, a masked-assertion model conditioned on the types the document already states rather than on an embedding of its text, is future work (Section 10).

## 3. Data

A neighbourhood needs a census. Three corpora supply it, each with the version history the real-omission benchmark needs [@amos; @wilson2016; @saier].

What a document looks like here: a privacy policy is one web page of a few thousand characters under a handful of headings (median {{v:corpus/privacy/sections_per_doc_median}} sections, {{v:corpus/privacy/chars_per_doc_median}} characters); a paper is the body text of an arXiv preprint, abstract to conclusion (median {{v:corpus/papers/sections_per_doc_median}} sections, {{v:corpus/papers/chars_per_doc_median}} characters). <!-- values: corpus/*/sections_per_doc_median, corpus/*/chars_per_doc_median --> The unit the system reads is the section; the unit it quotes is the sentence.

**Privacy policies** come from the Princeton–Leuven Longitudinal Corpus [@amos], about a million Wayback Machine snapshots of 130 thousand sites' policies, mostly the same policies repeated, so we keep one snapshot per site per year, stratified by year and category, with every site's full version chain. OPP-115 [@wilson2016], 115 policies hand-annotated with ten data-practice categories, checks our models and never trains them. **Research papers** come from the open subset of unarXive 2022 [@saier], primary category cs.CL or cs.LG with a first version from 2018 on; earlier versions of the test papers are fetched as arXiv e-prints and parsed with the same loader.

A heading-aware segmenter cuts policies into sections, with paragraph chunks as the fallback; unarXive's sections are split when they run to tens of thousands of characters. MinHash finds near-duplicates, collapsed into version chains; splits are disjoint by entity and sealed with a hash; every entity with a revision pair is in the test split, and no version of it is ever indexed (Table 1).

<!-- include: corpus -->
*Table 1. Corpus statistics (train is the indexed corpus; dev holds calibration and labelling samples; test is never indexed).*

Two findings shaped later phases. {{v:corpus/privacy/near_duplicate_rate}} of sampled policies are near-duplicates of another: many sites reuse one template, so some cohort neighbours are the same house painted twice, and {{v:corpus/privacy/cross_split_dropped}} near-copies of test documents were dropped from train and dev. <!-- values: corpus/privacy/near_duplicate_rate, corpus/privacy/cross_split_dropped --> And the open subset holds only {{v:corpus/papers/documents_total}} permissively licensed papers in scope, below the twenty to thirty thousand the plan asked for; we recorded this as a negative finding rather than widen the licence filter. <!-- values: corpus/papers/documents -->

The yearly snapshots suggest a shortcut, comparing a site's policy with its own earlier pages; but a site's earlier self says nothing about what the street expects, so cohort retrieval excludes the query's own entity, and in the revision benchmark the chain supplies the label and the street the expectation.

## 4. Architecture

The system is five steps, none of which asks a language model to guess at query time, in the lean tier that runs on a laptop and a free host; each component has a larger alternative behind one configuration key (decision record 001; Figure 1).

![Figure 1](figures/pipeline.png)
*Figure 1. The architecture: the offline pack build (top) and the query-time path (bottom), which calls no language model.*

**A genre is a pack.** The engine contains no genre-specific code. A pack bundles corpus, schema, presence model, prototypes, index and configuration, and adding a genre means running the pack builder, as a console plays whatever cartridge is inserted; a cartridge that needs the console modified is a bug in the console.

**Schema induction.** An open teacher model reads a stratified sample of development documents and extracts each section's assertions as short claims with verbatim spans, phrased as the *move* the document makes ("the authors report variance over seeds") rather than its subject, which would have induced a taxonomy of topics. Claims are embedded with Qwen3-Embedding-0.6B [@qwen3emb] under an instruction asking for the kind of statement, centred on their document's mean so that topic drops out, reduced with UMAP [@umap] and clustered with HDBSCAN [@campello; @mcinnes2017]. The larger teacher names each cluster; a single reviewer merges, splits and prunes against seeds (OPP-115's categories; the NeurIPS and ACL checklists [@neurips; @aclchecklist]); a coverage check on held-out sections found almost no statement without a type. Privacy has 58 types, papers 47.

**Presence detection** combines two cheap signals per section and type through a learned logistic layer: one calibrated logistic head per type over the frozen section embedding, and the maximum late-interaction similarity between the section's sentences and ten to thirty prototype sentences per type, computed ColBERT-style [@khattab] with a small Apache-licensed model [@colbertsmall; @clavie]. The plan added a cross-encoder reranker [@bgem3] for a borderline band, the way a clinic triages: a fast check for everyone, the specialist for the unsettled cases. The specialist did not pay, a few thousandths of macro-F1 at five to nine times the time, so it stays implemented, ablated and off. A small model fitted on whole labelled documents aggregates section probabilities to the document (Section 5).

**Cohort, ranking and evidence.** A document's vector is the mean of its section vectors blended with a title-and-abstract vector; the index is FAISS [@faiss]. Each indexed document carries a profile, a bit per type, and the expectation for an absent type is the cohort's mean bit. Evidence is each peer's best sentence per type, precomputed over the index. The Gradio demo [@gradio] renders each absence as a ghost paragraph where peers usually place that statement, with the evidence beside it. The stack is PyTorch, transformers and sentence-transformers [@wolf; @reimers], with every experiment run from one entry point and logged.

**What the user gets.** A document pasted or uploaded comes back in seconds with ghost paragraphs between its sections, one for each of the ten kinds of statement its peers state most often and it does not, each placed where peers usually say it and carrying the count, the interval and a peer's sentence (Figure 2); a click opens the evidence panel, and the cohort can be re-cut by k, category or date without re-embedding. The value is a second pair of eyes that has read the thousand hallways: for a policy, the practices sites of this kind usually disclose and this one does not; for a paper, the reporting its field expects.

![Figure 2](figures/demo_ghost.png)
*Figure 2. The demo's document panel on a test-split policy: the first ghost paragraph, placed where peers state what this document does not, with the count, the interval and a peer's sentence.*

## 5. Model selection and tuning

Why these models and not the plan's first choices? Three decisions, each recorded in the repository.

**The teacher.** The plan named Qwen2.5-72B-Instruct [@qwen25] on the university GPU server, which turned out to be a Windows workstation with four 32 GB consumer GPUs and a day-long session, where the 72B would have needed ten hours or more. I chose the plan's named fallback, the 32B of the same family at four-bit quantisation under llama.cpp [@llamacpp], confirmed the extraction prompt by eye on it, and used the 72B through a pinned, cached API route to name the clusters and label presence.

**The reference, and what a gate can measure.** The plan's hand-labelled reference set did not fit the schedule, so the reference became silver: the same 300 fixed sections per genre labelled by a second model of another family, Claude Haiku 5.5, with the teacher's own two-pass prompts (Table 2).

<!-- include: agreement -->
*Table 2. Agreement between labellers on the reference sections.*

Teacher and judge agree at micro-F1 {{v:agreement/privacy/teacher_vs_judge_privacy/micro_f1|f3}} on policies and {{v:agreement/papers/teacher_vs_judge_v2_papers/micro_f1|f3}} on papers, while either model agrees with OPP-115's human labels at about {{v:agreement/privacy/teacher_vs_opp115_reference_privacy/micro_f1|f3}}, whose scheme, mapped onto ours, is often narrower or broader than our definitions. <!-- values: agreement/* --> A student trained on the master's answers cannot agree with a third examiner more than the master does, so the plan's gate of macro-F1 0.80 was unreachable by construction and became relative, 80 per cent of the teacher's own macro-F1 against the same reference, with heads trained on teacher labels.

**The presence model.** Each head is a class-weighted logistic regression over the section embedding, regularisation and threshold chosen on the fitting half of the reference, with a temperature per type [@guo]; expected calibration error (ECE) [@naeini] is reported because later stages use P(present) as a probability (Table 3, evaluation half).

<!-- include: presence -->
*Table 3. Presence detection: section-level variants, the Phase 4 gate, and document-level aggregation.*

Three things stand out. Prototypes alone are weak, but next to the classifier they cut ECE from {{v:presence/privacy/classifier only/ece|f3}} to {{v:presence/privacy/classifier + prototypes/ece|f3}} on policies. <!-- values: presence/privacy/classifier only/ece, presence/privacy/classifier + prototypes/ece --> The gate passes for policies at {{v:presence/privacy/gate/share_of_ceiling|f3}} of the ceiling and is not met for papers at {{v:presence/papers/gate/share_of_ceiling|f3}}: a frozen whole-section embedding cannot separate 47 fine-grained types in long sections that carry about five each. <!-- values: presence/*/gate/share_of_ceiling --> The gate was recorded as not met; fine-tuning the embedder is the recorded remedy. And aggregating by "any section above its threshold" was miscalibrated in both directions; a logistic model over the best section probabilities and the section count, fitted on 150 whole judge-labelled documents per genre, raised document-level macro-F1 from {{v:doc_presence/privacy/v1/macro_f1|f3}} to {{v:doc_presence/privacy/v2/macro_f1|f3}} on policies and from {{v:doc_presence/papers/v1/macro_f1|f3}} to {{v:doc_presence/papers/v2/macro_f1|f3}} on papers, and cut ECE to {{v:doc_presence/privacy/v2/ece|f3}} and {{v:doc_presence/papers/v2/ece|f3}}. <!-- values: doc_presence/* -->

The cohort settings were fixed in the plan and selected on no benchmark; nothing was selected on the test split.

## 6. Evaluation I: OmitBench, and the privacy turn

**OmitBench-synthetic** is a jigsaw whose missing piece we removed ourselves. From each held-out test document we draw a type it states, delete the sentences that carry it, verify that it is now absent, and ask each system to rank the absences; three regimes delete one type, three types or every section carrying the type, 500 instances per regime and genre, except that long papers resist three-type deletion and gave {{v:synth/papers/three_types/n}}. <!-- values: synth/papers/three_types/n --> Metrics are recall at 1, 5 and 10, MRR and nDCG@10 with 95 per cent bootstrap intervals [@efron], plus the false-alarm rate (the share of the top five in fact stated) and the calibration of P(expected) on 150 untouched judge-labelled documents (Table 4, Figure 3; the other metrics and the untouched-document checks are in Appendix A, Table 11).

<!-- include: synth -->
*Table 4. OmitBench-synthetic, recall@5.*

![Figure 3](figures/bench_recall5.png)
*Figure 3. Recall@5 per regime and system, both genres, with 95 per cent bootstrap intervals; reference rows in grey.*

Both systems sit far above random order; whether the cohort beats genre-wide prevalence outside the paired interval is the plan's gate (Table 5, Figure 4).

<!-- include: gate66 -->
*Table 5. The gate quantity across benchmarks.*

![Figure 4](figures/level1_minus_uniform.png)
*Figure 4. Paired recall@5 difference, level 1 minus genre-uniform; blue where the interval excludes zero.*

**OmitBench-revision** treats the authors' own later addition as a confession: if version *n*+1 adds a type that at least half of the peers dated before version *n* already stated, version *n* conspicuously omitted it, with expectation computed against the time-windowed cohort (Table 6).

<!-- include: revision -->
*Table 6. OmitBench-revision.*

For papers the cohort wins everywhere the plan asked, by {{v:gate66/papers/single_type|s3}} on the one-type regime and {{v:gate66/papers/revision|s3}} on real revisions, intervals above zero. <!-- values: gate66/papers/single_type, gate66/papers/revision --> For policies it wins nowhere: {{v:gate66/privacy/single_type|s3}} {{v:gate66/privacy/single_type/ci|sci}} on the gate regime, and small negative differences elsewhere. <!-- values: gate66/privacy/single_type -->

**The diagnosis.** Per cohort variant we measured how well cohort prevalence predicts which types an untouched document states (a within-document AUC, 800 test documents), and recall@5 with the cohort retrieved from the edited document's vector, what the system sees, against the original's, what it would see if the deletion left no trace (Table 7, Figure 5; every variant in Appendix A, Table 12).

<!-- include: leak -->
*Table 7. Cohort diagnosis, the default cohort against genre-wide prevalence.*

![Figure 5](figures/cohort_diagnosis.png)
*Figure 5. Within-document AUC per cohort variant and, for privacy, the deletion leak: gain over genre-uniform with the cohort retrieved from the edited versus the original document.*

Privacy cohorts carry little information about practices: against genre prevalence the default cohort raises within-document AUC from {{v:leak/privacy/genre_uniform/within_doc_auc|f3}} to {{v:leak/privacy/default/within_doc_auc|f3}}, against {{v:leak/papers/genre_uniform/within_doc_auc|f3}} to {{v:leak/papers/default/within_doc_auc|f3}} for papers. <!-- values: leak/*/genre_uniform/within_doc_auc, leak/*/default/within_doc_auc --> Two settings fixed in the plan weaken them further, the site category, which does not track data practices, and the title part of the vector, which mostly encodes a brand; without both, AUC reaches {{v:leak/privacy/no_title_no_category/within_doc_auc|f3}}. <!-- values: leak/privacy/no_title_no_category/within_doc_auc --> The main cause: the deletion leaks into the cohort. Picture an essay with its methods section cut out, compared with the essays it now most resembles: they also lack a methods section, and they vouch for the omission. A policy that loses a topic moves towards the policies that never had it, so the deleted type's cohort prevalence drops exactly where it is needed. With the better cohort, the original vector would give {{v:leak/privacy/no_title_no_category/original_minus_uniform|s3}} {{v:leak/privacy/no_title_no_category/original_minus_uniform/ci|sci}} over genre-uniform; the edited vector gives {{v:leak/privacy/no_title_no_category/edited_minus_uniform|s3}}, and the leak, {{v:leak/privacy/no_title_no_category/leak_original_minus_edited|s3}}, takes most of the gain away. <!-- values: leak/privacy/no_title_no_category/* --> Papers are shielded by their title and abstract, which no deletion touches; their leak at the default is {{v:leak/papers/default/leak_original_minus_edited|s3}}. <!-- values: leak/papers/default/leak_original_minus_edited -->

One could object that the leak is an artefact of deletion; but a real omitter also resembles other omitters, and the revision benchmark, where nothing was deleted, shows the same sign for policies (Table 5). The leak is a property of the task, and it points at level 2: an expectation conditioned on what the document states cannot be moved by the absence it is asked to find.

A search over eleven cohort settings on the test bench finds a variant that would pass the privacy gate; I did not adopt it, since it was selected on test data, and it is recorded as post hoc and excluded from every table here.

## 7. Evaluation II: the frontier gap, speed and errors

Could a chat model have done this without a corpus? In the candidate-list setting the model reads one document and the genre's catalogue with definitions, and returns the ten types it judges most conspicuously absent; the prompt never mentions that text was removed. We ran Claude Opus 5.5 (batch mode, moderate reasoning) and the open teacher Qwen2.5-72B-Instruct (pinned provider), each once, on the same {{v:llm_gap/n}} one-type instances per genre, paired against the corpus systems (Table 8, Figure 6). <!-- values: llm_gap/n -->

<!-- include: llm_gap -->
*Table 8. The frontier comparison.*

![Figure 6](figures/llm_gap.png)
*Figure 6. Recall@1, 5 and 10 of the four systems on the same instances.*

An auditor with a checklist and no memory of the other thousand documents searches for silence 58 times over a long text; the surveyor who knows the neighbourhood does not search but counts. Opus reaches recall@5 of {{v:llm_gap/privacy/anthropic:claude-opus-5-5/recall@5|f3}} on policies and {{v:llm_gap/papers/anthropic:claude-opus-5-5/recall@5|f3}} on papers, against {{v:llm_gap/privacy/level1/recall@5|f3}} and {{v:llm_gap/papers/level1/recall@5|f3}} for the cohort and {{v:llm_gap/privacy/genre_uniform/recall@5|f3}} and {{v:llm_gap/papers/genre_uniform/recall@5|f3}} for genre-wide prevalence; every paired difference has an interval that excludes zero. <!-- values: llm_gap/*/recall@5 --> The open model is further behind: most of its lists follow the catalogue's own order, so it filters rather than ranks.

Two caveats. The yardstick was built with our own detector, so a chat model that reads a paraphrase we left behind as stated is marked wrong by a ruler it did not help make; and each model ran once, at one setting, so a better prompt or more thinking may close part of the gap.

**Speed.** The whole pipeline runs in seconds on a laptop (Apple M5, MPS; Table 9), embedding being most of the time; on the demo's free two-core host a policy takes roughly ten seconds and a paper about half a minute.

<!-- include: speed -->
*Table 9. End-to-end time per document on the laptop.*

**Errors.** The types the cohort never finds are the rare ones: {{v:errors/privacy/types_never_in_top5}} of {{v:errors/privacy/types_scored}} privacy types never reach the top five, with a median corpus prevalence of {{v:errors/privacy/median_prevalence_never|f3}} against {{v:errors/privacy/median_prevalence_all|f3}} for all scored types; a prevalence ranker cannot rank what few peers state. <!-- values: errors/privacy/* --> (For papers the never-found types are the opposite case, stated by every paper and so hardly ever deletable.) The recurring false alarms are types stated in unusual words, which the detector half-sees. The cohort's gain grows with length for papers, from {{v:errors/papers/by_length/q1|s3}} in the shortest quartile to {{v:errors/papers/by_length/q3|s3}} in the third, while for policies only the shortest quartile shows a gain with an interval above zero, {{v:errors/privacy/by_length/q1|s3}}. <!-- values: errors/*/by_length/* -->

## 8. Related work

Automated reading of privacy policies runs from the OPP-115 corpus [@wilson2016] to Polisis [@harkous], which classifies policy segments into data practices, and the longitudinal corpus we build on was created to study how policies changed around the GDPR [@amos]; all of it reads what a policy says, and the contribution here is the complement. For papers the nearest neighbours are the reporting checklists venues now require [@pineau; @neurips; @aclchecklist] and the studies of what they achieve [@magnusson]; a checklist is a hand-written catalogue of expected statements, and schema induction can be read as learning one from the genre. That language models are poor at negation and absence is established for understanding tasks [@truong; @hossain] and for retrieval [@weller]; Section 7 adds a document-level instance. The planned level-2 model is a set transformer over present types [@lee].

## 9. Limitations and responsible use

**Prevalence describes a community; it is not a standard.** A share of peers encodes what the dominant actors of a genre do, including their habits of saying too little: a dress code inferred from the people already in a room tells a newcomer what is usual and nothing about what is right. Every output is therefore a prevalence with evidence attached, never a verdict.

**The catalogue carries the teacher's eye, and the reference is silver.** A kind of statement the teacher does not notice cannot become a type, and every presence number measures agreement between two models; OPP-115 is the one human check, for privacy and a subset of types. Corpora, prompts and catalogues are English only.

**A deletion is not an omission, and the benchmark shares our detector.** The synthetic benchmark favours the corpus systems in the frontier comparison, and the deletion leaks into the cohort. Section 6 argued that real omissions leak the same way, but that rests on the revision benchmark, which people have not yet validated: the 50-document, two-annotator check is the next phase, and until then the revision labels carry two confounds, additions driven by regulation rather than by peers, and authors' own drift, neither quantified.

**Small models, one pass, two genres.** The papers detector is below its gate, so presence misses become omission false alarms; the chat models ran once each; the open-ended and retrieval-augmented baselines, the masked-assertion model and a third genre are future work.

**Dual use.** A detector of conspicuous absence is also a map of what a document can leave out unnoticed: an author could tune a text until the ghost paragraphs disappear while the substance stays missing. We see no clean technical answer: the choices that make the system honest also make it legible to such an author, and the mitigation is the reader, who sees the peers' sentences.

Table 10 sets each risk against its mitigation.

| risk | mitigation in place | still open |
|---|---|---|
| a count read as a standard | the wording rule ("N% of peers like this one state X"), evidence beside every number, no verdict word in the interface (tested) | a reader may still take a prevalence for a norm |
| norms that encode dominant actors | the cohort is shown with its k, category and date window and can be re-cut; prevalence is reported, never compliance | no measure of whose practices a cohort over-represents |
| the teacher's eye and a silver reference | OPP-115 as a human check for privacy; the gate relative to the teacher; the limits stated with every number | human validation of the revision labels, a human check for papers |
| dual use: tuning a text until the ghosts disappear | every claim is a quote from a peer, visible to the reader; the catalogue ships with the demo | none beyond the reader |
| the privacy of uploaded documents | processed in memory, never stored; no API call on the query path; teacher explanations off by default and flagged | none |
| third-party text in the corpora and the demo | licences recorded before use; short attributed quotes; the non-commercial OPP-115 is a check, not training data | the product layer needs a licence or a swap |

*Table 10. Risks, the mitigation in place, and what remains open.*

## 10. Conclusion

Conspicuous omission is now a ranking task with a frozen catalogue per genre, a score that multiplies expectation by absence, and two label-free benchmarks, built for two genres from one pipeline with no genre code. Conditioning expectation on the nearest peers beats genre-wide prevalence for research papers on every regime and on real revisions, and does not for privacy policies, where the cohort carries little information about practices and the absence itself moves the document towards peers that share it; we reported that failure as a finding and kept the settings the plan fixed. Two frontier chat models given the same catalogue fall well behind both corpus baselines, within the caveats of Section 7.

What comes next follows from what failed: the masked-assertion model conditions expectation on the statements a document makes, so the absence it is asked to find cannot move it; the revision labels need human validation; and a third genre would test the pack abstraction. The demo is live, and the repository holds the configurations, prompts and run log needed to produce every number here again.

## GenAI acknowledgement

<!-- Plan 7.5. The short form, as Parsa asked (2026-10-25): the practice is recorded in CLAUDE.md and the run log. -->

Generative AI was used as an assistant throughout this project, to work faster rather than to replace the work: Claude Code (Anthropic) helped write and test code, run the recorded experiments, keep the changelog, and edit this report, under rules kept in the repository. The research question, the design decisions, the review of every result, the human pass over both schemas and the final text are the author's own. Language models are also part of the system under study (Qwen2.5 for schema induction and presence labels, Claude Haiku 5.5 as the second judge, Claude Opus 5.5 and Qwen2.5-72B as the compared chat models); every prompt is versioned in the repository and every number in this report comes from the result tables.

## References

{{references}}

## Appendix A. Full benchmark tables

<!-- include: synth_more -->
*Table 11. OmitBench-synthetic, one-type regime: the other ranking metrics, and false alarms and calibration on untouched documents.*

<!-- include: leak_full -->
*Table 12. Cohort diagnosis, every variant; the variant selected on the test bench is not shown.*
