# Data sources

Plan §2.1–2.2. Everything here was checked against the raw files on **2026-10-12**; field names are copied from the
files, not from memory. Raw files live in `data/raw/<source>/` (gitignored) and are never modified.

| Corpus | Genre | Retrieved from | Licence / terms | Version, size | Citation |
|---|---|---|---|---|---|
| Princeton–Leuven Longitudinal Corpus of Privacy Policies | A (privacy) | https://privacypolicies.cs.princeton.edu/data-release/data/release_db.sqlite.xz (linked from https://privacypolicies.cs.princeton.edu) | Authors' contributions public domain; "no claims about the licensing of any other content" (the policies belong to their websites). Quote short spans with attribution. | Release of 2020-06-13; 3.29 GB `.xz`, **48.2 GB** SQLite; 1,071,487 snapshots, 333,986 distinct texts, 130,620 sites, 1997–2019 | Amos, Acar, Lucherini, Kshirsagar, Narayanan, Mayer. *Privacy Policies over Time: Curation and Analysis of a Million-Document Dataset.* WWW 2021. doi:10.1145/3442381.3450048 |
| OPP-115 | A (presence labels, sanity set) | https://usableprivacy.org/static/data/OPP-115_v1_0.zip (linked from https://usableprivacy.org/data) | © 2016 CMU, all rights reserved; research, teaching and scholarship only, "in the spirit of CC BY-NC". Separate commercial licence for the annotations. Publications must cite the paper and be emailed to Prof. Norman Sadeh. | v1.0 (2016-07-10); 99 MB zip; 115 policies, 3,792 segments, 19,720 consolidated practices (threshold 0.75) | Wilson et al. *The Creation and Analysis of a Website Privacy Policy Corpus.* ACL 2016 |
| unarXive, open subset | B (papers) | https://zenodo.org/records/7752615 (`unarXive_230324_open_subset.tar.xz`) | Dataset CC BY-SA 4.0; only permissively licensed papers are in the open subset (each paper's licence is in `metadata.license`) | Release 230324 (arXiv sources up to 2022-12-31, metadata snapshot 2023-01-01); 4.84 GB `.tar.xz` | Saier, Krause, Färber. *unarXive 2022: All arXiv Publications Pre-Processed for NLP, Including Structured Full-Text and Citation Network.* JCDL 2023 |
| arXiv e-prints (earlier versions) | B (revision pairs only) | `https://export.arxiv.org/e-print/<id><version>`, one request per 3 s, cached in `data/raw/arxiv_eprints/` | Each paper's own licence (only open-subset papers are fetched) | fetched on demand for multi-version test papers | arXiv |

**Schema seeds (plan §3.4 cross-check), read 2026-10-13, not downloaded; item lists paraphrased in
`packs/<genre>/seeds.yaml`:** the NeurIPS Paper Checklist, NeurIPS 2026 guidelines, 16 items,
https://neurips.cc/public/guides/PaperChecklist; the ACL Responsible NLP Research checklist, ARR revision of October
2024, 18 items (A1–A2, B1–B6, C1–C4, D1–D5, E1), https://aclrollingreview.org/responsibleNLPresearch/. Genre A uses
OPP-115's own scheme (`documentation/categories-july30.xml`, version 20151115-01, 10 categories).

Not used, recorded for later: **unarXive 2024** (Besrour & Färber, LREC 2026; https://huggingface.co/datasets/ines-besrour/unarxive_2024, MIT, 105 GB in three tar parts, covers post-2022 papers — the upgrade path for the 2022 ACL-limitations drift analysis in Phase 10); the OPP-115 GDPR extension (`JURIX_2020_OPP-115_GDPR_v1.0.zip` on usableprivacy.org).

**Licence consequences.** Lean-tier presence heads for genre A are trained on OPP-115 annotations, which are
non-commercial: fine for the class and the paper, a blocker for the Phase 12 product until replaced or licensed.
Evidence sentences shown in the demo come from public policies (short quotes, attributed) and from CC-licensed papers.

## Princeton–Leuven: `release_db.sqlite`

Real schema (from `sqlite_master`; matches `database-fields.pdf`):

```
sites(id INTEGER PK, domain TEXT, categories TEXT)
alexa_ranks(site_id INTEGER, year INTEGER, phase TEXT, rank INTEGER)
policy_texts(id INTEGER PK, policy_text TEXT, flesch_kincaid REAL, smog REAL, flesch_ease TEXT, length INTEGER,
             sha1 TEXT, simhash TEXT)
policy_htmls(id INTEGER PK, policy_html TEXT, policy_html_sha1 TEXT)
policy_reader_view_htmls(id INTEGER PK, policy_reader_view_html TEXT, policy_reader_view_html_sha1 TEXT)
policy_snapshots(id INTEGER PK, site_id INTEGER, homepage_snapshot_url TEXT, policy_snapshot_url TEXT, policy_url TEXT,
                 homepage_snapshot_redirected_url TEXT, year INTEGER, phase TEXT, policy_text_id INTEGER,
                 policy_html_id INTEGER, policy_reader_view_html_id INTEGER, file_type TEXT, policy_title TEXT,
                 link_text TEXT, pdf_filename TEXT, classifier_probability REAL, analysis_subcorpus INTEGER,
                 parked_domain INTEGER, cross_domain_homepage_redir INTEGER)
```

What the data actually looks like:

- **Documents and versions.** A document is a row of `policy_snapshots`; its text is `policy_texts.policy_text` via
  `policy_text_id`. There is at most one snapshot per `(site_id, year, phase)`; `phase` is `A` (Jan–Jun) or `B`
  (Jul–Dec). Snapshots whose policy did not change share a `policy_texts` row (1,071,487 snapshots → 333,986 texts).
  Versions of an entity = all snapshots of one `site_id`, ordered by date.
- **Dates.** `year` + `phase` only; the exact capture date is the Wayback timestamp inside `policy_snapshot_url`
  (`https://web.archive.org/web/20070627151050id_/...` → 2007-06-27). Used as `Document.date`.
- **Entity.** `sites.domain` ("may include subdomains"), lower-cased with `www.` stripped → `Document.entity_id`.
- **Categories.** `sites.categories`: Webshrinker categories, `;`-separated and **alphabetical** (e.g.
  `business;education;informationtech`), so there is no primary category. 41 distinct values, 121,718/130,620 sites
  categorised. We use the **rarest listed category** (catch-alls `business`, `uncategorized` only when nothing else is
  listed) as `Document.category`; `parked` and `illegalcontent` sites are excluded.
- **Quality flags.** `analysis_subcorpus` = 1 on 910,546 snapshots (the authors' cleaned subset; we require it),
  `parked_domain` = 1 on 24,289 (excluded), `file_type` `html` 1,066,349 / `pdf` 5,138 (PDFs excluded: their text is a
  separate extraction), `classifier_probability` 0.735–1.0.
- **Years.** 1997 (9 snapshots) to 2019 (74,755); peak 2016 (108,437). After filtering and one snapshot per site and
  year: 545,463 entity-years over 103,020 websites.
- **Text format.** Markdown: `#`/`##` headings, bold-line headings (`**1\. What ...**`), escaped numbering (`1\.`),
  numbered tables of contents, `[text](web.archive.org/...)` links, `  * ` bullets, `* * *` rules, empty `###` lines,
  and plain title lines with no markup ("Applicability"); some policies have no headings at all, or run-in headings
  inside paragraphs ("Acceptance of Terms. Welcome to ..."). The GitHub mirror adds a "The following text is
  extracted and transformed ..." preamble line; the database text does not have it.

## OPP-115: `OPP-115/`

Documented in `documentation/manual.txt`. Used files:

- `sanitized_policies/<policy UID>_<domain>.html`: the text annotators saw, segments separated by `|||`; markup is
  `br` (10,966), `strong` (2,413), `ul`/`li`, a handful of `p`/`ol`/`blockquote`. A leading `<strong>` run is the
  segment heading. Sections in our data are these segments, **never re-segmented** (`Section.idx` = segment ID).
- `consolidation/threshold-0.75-overlap-similarity/<UID>_<domain>.csv` (no header, Excel dialect): `annotation ID`
  (`C…` = consolidated from 2–3 annotators), `batch ID`, `annotator ID`, `policy ID`, `segment ID` (0-based),
  `category name`, `attribute-value pairs` (JSON: attribute → `{selectedText?, startIndexInSegment,
  endIndexInSegment, value}`; `"not-selected"` = no value), `date`, `policy URL`. Ten categories; counts at 0.75:
  First Party Collection/Use 8,149, Third Party Sharing/Collection 4,817, Other 2,306, User Choice/Control 1,677,
  Data Security 699, User Access, Edit and Deletion 690, International and Specific Audiences 529, Policy Change 463,
  Data Retention 337, Do Not Track 53.
- `documentation/policies_opp115.csv`: `Policy UID, Policy URL, Policy collection date, Policy last updated date`.
- `documentation/websites_opp115.csv`: `Site UID, Site URL, Site Human-Readable Name, Policy UID, Site Check Date,
  In 115 Set?, Comments, Sectoral Data, …` (DMOZ sectors spill over many unnamed columns; we keep the first five).

All 19,720 labels align with a segment; 3,791 of 3,792 segments carry at least one practice; 5 segments are empty.
Labels are written to `data/processed/privacy/opp115_labels.jsonl` (`doc_id, section_idx, category, attributes,
annotation_id`).

## unarXive open subset: `unarXive_230324_open_subset.tar.xz`

`<yy>/arXiv_src_<yymm>_<num>.jsonl`, one paper per line. Fields (README in the record, and checked on the file):
`paper_id`, `_pdf_hash` (always null), `_source_hash`, `_source_name`, `metadata` (the Kaggle arXiv snapshot record:
`id, submitter, authors, title, comments, journal-ref, doi, report-no, categories, license, abstract, versions,
update_date, authors_parsed`), `discipline`, `abstract`, `body_text` (list of `{section, sec_number, sec_type,
content_type, text, cite_spans, ref_spans}`), `bib_entries`, `ref_entries`. Placeholders in text:
`{{formula:<uuid>}}`, `{{cite:<hash>}}`, `{{ref:…}}`/`{{figure:…}}`/`{{table:…}}`.

- **Category.** `metadata.categories` is a space-separated string whose first entry is the primary category
  (`cs.CL cs.LG`). We keep primary `cs.CL` or `cs.LG`.
- **Versions and dates.** `metadata.versions` = `[{version: "v1", created: "Mon, 2 Apr 2007 19:18:42 GMT"}, …]`.
  The open subset's text is the latest version available on 2022-12-31; `Document.version`/`date` are that
  version's. Papers whose v1 predates 2018-01-01 are dropped.
- **Sections.** Consecutive `body_text` paragraphs with the same `section` name form one Section (subsections are
  their own `section` value); `content_type` `paragraph` and `listing` are kept; the abstract is section 0.

Extraction: 151,330 papers in the year folders read, **17,995 kept** (primary cs.CL/cs.LG, v1 on or after 2018-01-01); 1,561 files of earlier years skipped without parsing. Counts per split are in `reports/tables/corpus_stats.csv` and the CHANGELOG.

## arXiv e-prints (revision pairs, genre B)

`https://export.arxiv.org/e-print/<id>v<N>` returns that version's LaTeX source (gzipped tar, or a single gzipped
`.tex`). Checked on 1810.04805v1 (232 KB tar.gz). Both versions of a pair are parsed by
`data/loaders/arxiv_versions.py`, so differences are edits, not parser artefacts. Pair = (v1, last version ≤ 2022-12-31).
