"""Loaders (plan §2.4) and the build (plan §2.8) on small fixtures that use the corpora's real formats and field names
(data/README.md). Each loader round-trips three documents and writes an HTML preview with section boundaries."""

import csv
import gzip
import io
import json
import sqlite3
import tarfile

import pytest

from data.build import build_privacy, preview_html
from data.loaders import arxiv_versions, opp115, privacy_pl, unarxive
from data.schema import Document, Section, read_jsonl

POLICY_V1 = "## Information We Collect\n\nWe collect your name and email address when you register for an account.\n\n" \
            "## Sharing\n\nWe do not sell personal information to anyone, and we share it only with processors."
POLICY_V2 = POLICY_V1 + "\n\n## Data Retention\n\nWe keep account data for two years after your last login, then delete " \
            "it.\n\n## Your Rights\n\nYou can ask for a copy of your data or its deletion by writing to our office."


def check_round_trip(docs, tmp_path, name):
    assert len(docs) >= 3
    for d in docs[:3]:
        assert d.sections and all(s.doc_id == d.document.id for s in d.sections)
        assert [s.idx for s in d.sections] == list(range(len(d.sections)))
        for s in d.sections:
            assert d.text()[s.char_span[0]:s.char_span[1]] == s.text
    preview_html(docs[:3], tmp_path / f"loader_check_{name}.html", name)
    page = (tmp_path / f"loader_check_{name}.html").read_text()
    assert page.count("class='sec'") == sum(len(d.sections) for d in docs[:3])


# ------------------------------------------------------------------------------------ Princeton–Leuven
def make_privacy_db(path, n_sites=12):
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE sites (id INTEGER PRIMARY KEY AUTOINCREMENT, domain TEXT, categories TEXT);
    CREATE TABLE policy_texts (id INTEGER PRIMARY KEY AUTOINCREMENT, policy_text TEXT, flesch_kincaid REAL, smog REAL,
        flesch_ease TEXT, length INTEGER, sha1 TEXT, simhash TEXT);
    CREATE TABLE policy_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, site_id INTEGER, homepage_snapshot_url TEXT,
        policy_snapshot_url TEXT, policy_url TEXT, homepage_snapshot_redirected_url TEXT, year INTEGER, phase TEXT,
        policy_text_id INTEGER, policy_html_id INTEGER, policy_reader_view_html_id INTEGER, file_type TEXT,
        policy_title TEXT, link_text TEXT, pdf_filename TEXT, classifier_probability REAL, analysis_subcorpus INTEGER,
        parked_domain INTEGER, cross_domain_homepage_redir INTEGER);""")
    texts = {}
    for i in range(n_sites):
        cats = ["business;shopping", "business", "education;entertainment", "parked"][i % 4]
        con.execute("insert into sites (domain, categories) values (?, ?)", (f"www.site{i}.com", cats))
        body = POLICY_V1.replace("account", f"account at site {i} {'x' * i}")
        for version, text in (("v1", body), ("v2", body + POLICY_V2[len(POLICY_V1):])):
            cur = con.execute("insert into policy_texts (policy_text, length, sha1) values (?, ?, ?)",
                              (text, len(text), f"{i}{version}"))
            texts[(i, version)] = cur.lastrowid
        for year, phase, version in ((2016, "A", "v1"), (2016, "B", "v1"), (2018, "B", "v2")):
            con.execute("""insert into policy_snapshots (site_id, policy_snapshot_url, policy_url, year, phase,
                policy_text_id, file_type, policy_title, analysis_subcorpus, parked_domain) values (?,?,?,?,?,?,?,?,?,?)""",
                        (i + 1, f"https://web.archive.org/web/{year}{'03' if phase == 'A' else '09'}15120000id_/http%3A//site{i}.com/privacy",
                         f"site{i}.com/privacy", year, phase, texts[(i, version)], "html", f"Site {i} Privacy", 1, 0))
    con.execute("""insert into policy_snapshots (site_id, policy_snapshot_url, year, phase, policy_text_id, file_type,
        analysis_subcorpus, parked_domain) values (1, 'x', 2017, 'A', 1, 'pdf', 1, 0)""")
    con.commit()
    con.close()


def test_privacy_pl_index_and_round_trip(tmp_path):
    db = tmp_path / "release_db.sqlite"
    make_privacy_db(db)
    con = privacy_pl.connect(db)
    rows = privacy_pl.snapshot_index(con)
    assert {r.domain for r in rows} == {f"www.site{i}.com" for i in range(12) if i % 4 != 3}  # 'parked' excluded
    assert all(r.year != 2017 for r in rows)  # the pdf snapshot is filtered
    r0 = next(r for r in rows if r.domain == "www.site0.com" and r.phase == "A")
    assert (r0.entity_id, r0.date, r0.category) == ("site0.com", "2016-03-15", "shopping")  # rarest, not 'business'
    one = privacy_pl.one_per_entity_year(rows)
    assert len(one) == 2 * 9 and all(r.phase == "B" for r in one)
    docs = list(privacy_pl.iter_documents(con, sorted(one, key=lambda r: r.snapshot_id)))
    assert [s.heading for s in docs[0].sections] == ["Information We Collect", "Sharing"]
    assert docs[0].document.version == "2016B" and docs[0].document.meta["policy_url"] == "site0.com/privacy"
    check_round_trip(docs, tmp_path, "privacy_pl")


# ------------------------------------------------------------------------------------ OPP-115
def make_opp115(root, n=3):
    base = root / "OPP-115"
    for d in ("sanitized_policies", "consolidation/threshold-0.75-overlap-similarity", "documentation"):
        (base / d).mkdir(parents=True)
    with open(base / "documentation/policies_opp115.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Policy UID", "Policy URL", "Policy collection date", "Policy last updated date", ""])
        for i in range(n):
            w.writerow([10 + i, f"opp{i}.com/privacy", "2015-07-02", "2015-01-01", ""])
    with open(base / "documentation/websites_opp115.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Site UID", "Site URL", "Site Human-Readable Name", "Policy UID", "Site Check Date", "In 115 Set?",
                    "Comments", "Sectoral Data"])
        for i in range(n):
            w.writerow([i, f"www.opp{i}.com", f"Opp {i}", 10 + i, "2016-02-08", "Yes", "", "News", "Arts"])
    for i in range(n):
        segs = ["<strong> Privacy Policy </strong> <br> <br> <strong> Last updated: June 23, 2015 </strong> <br>",
                "We collect information you give us . <br> <br> <ul> <li> your name; </li> </ul> <ul> <li> your email </li> </ul>",
                "We keep data for 30 days &amp; then delete it ."]
        (base / f"sanitized_policies/{10 + i}_opp{i}.com.html").write_text("|||".join(segs))
        with open(base / f"consolidation/threshold-0.75-overlap-similarity/{10 + i}_opp{i}.com.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"C{i}", "batch", "1", 10 + i, 1, "First Party Collection/Use",
                        json.dumps({"Personal Information Type": {"startIndexInSegment": 3, "endIndexInSegment": 9,
                                                                  "selectedText": "name", "value": "Contact"},
                                    "Purpose": {"startIndexInSegment": -1, "endIndexInSegment": -1, "value": "not-selected"}}),
                        "7/2/15", f"http://opp{i}.com/privacy"])
            w.writerow([f"{i}b", "batch", "2", 10 + i, 2, "Data Retention",
                        json.dumps({"Retention Period": {"startIndexInSegment": 0, "endIndexInSegment": 5, "value": "Limited"}}),
                        "7/2/15", f"http://opp{i}.com/privacy"])
    return root


def test_opp115_segments_align_with_labels(tmp_path):
    raw = make_opp115(tmp_path / "opp115")
    docs = list(opp115.iter_documents(raw))
    d = docs[0]
    assert (d.document.id, d.document.entity_id, d.document.date) == ("opp115:10", "opp0.com", "2015-07-02")
    assert d.document.meta["dmoz_sectors"] == ["News", "Arts"]
    assert d.sections[0].heading == "Privacy Policy" and d.sections[0].text == "Last updated: June 23, 2015"
    assert d.sections[1].text.split("\n") == ["We collect information you give us.", "your name;", "your email"]
    assert d.sections[2].text == "We keep data for 30 days & then delete it."
    labels = list(opp115.iter_labels(raw))
    assert labels[0] == {"doc_id": "opp115:10", "section_idx": 1, "category": "First Party Collection/Use",
                         "attributes": {"Personal Information Type": "Contact"}, "annotation_id": "C0"}
    secs = {(x.document.id, s.idx) for x in docs for s in x.sections}
    assert all((lab["doc_id"], lab["section_idx"]) in secs for lab in labels)
    check_round_trip(docs, tmp_path, "opp115")


# ------------------------------------------------------------------------------------ unarXive
def make_unarxive(path):
    def paper(pid, cats, created):
        return {"paper_id": pid, "_pdf_hash": None, "_source_hash": "h", "_source_name": f"{pid}.gz",
                "metadata": {"id": pid, "categories": cats, "title": f"Paper {pid}", "license": "http://creativecommons.org/licenses/by/4.0/",
                             "versions": [{"version": "v1", "created": created}, {"version": "v2", "created": "Mon, 1 Jul 2019 10:00:00 GMT"}]},
                "discipline": "Computer Science", "abstract": {"section": "Abstract", "text": "We study {{formula:1}}."},
                "body_text": [{"section": "Introduction", "sec_number": "1", "sec_type": "section", "content_type": "paragraph",
                               "text": "Language models [cite] are useful {{cite:ab}}.", "cite_spans": [], "ref_spans": []},
                              {"section": "Limitations", "sec_number": "5", "sec_type": "section", "content_type": "paragraph",
                               "text": "Our study covers English only.", "cite_spans": [], "ref_spans": []}],
                "bib_entries": {"ab": {"bib_entry_raw": "X"}}, "ref_entries": {}}
    papers = [paper("1801.00001", "cs.CL cs.AI", "Tue, 2 Jan 2018 10:00:00 GMT"),
              paper("1801.00002", "cs.CV", "Tue, 2 Jan 2018 10:00:00 GMT"),
              paper("1712.00003", "cs.LG", "Fri, 1 Dec 2017 10:00:00 GMT"),
              paper("1903.00004", "cs.LG stat.ML", "Fri, 1 Mar 2019 10:00:00 GMT"),
              paper("1904.00005", "cs.CL", "Mon, 1 Apr 2019 10:00:00 GMT")]
    blob = "".join(json.dumps(p) + "\n" for p in papers).encode()
    old_blob = (json.dumps(paper("9912.00009", "cs.CL", "Wed, 1 Dec 1999 10:00:00 GMT")) + "\n").encode()
    with tarfile.open(path, "w:xz") as tar:
        for name, data in (("99/arXiv_src_9912_001.jsonl", old_blob), ("18/arXiv_src_1801_001.jsonl", blob)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_unarxive_extract_filters_and_round_trip(tmp_path):
    make_unarxive(tmp_path / "u.tar.xz")
    counts = unarxive.extract(tmp_path / "u.tar.xz", tmp_path / "x.jsonl.gz", categories={"cs.CL", "cs.LG"},
                              min_date="2018-01-01")
    assert counts == {"papers_seen": 5, "papers_kept": 3, "files_skipped_by_year": 1}
    docs = list(unarxive.iter_documents(tmp_path / "x.jsonl.gz"))
    assert [d.document.id for d in docs] == ["unarxive:1801.00001", "unarxive:1903.00004", "unarxive:1904.00005"]
    d = docs[0]
    assert (d.document.category, d.document.version, d.document.date) == ("cs.CL", "v2", "2019-07-01")
    assert [s.heading for s in d.sections] == ["Abstract", "Introduction", "Limitations"]
    assert d.sections[0].text == "We study [math]." and "bib_entries" not in gzip.open(tmp_path / "x.jsonl.gz", "rt").readline()
    check_round_trip(docs, tmp_path, "unarxive")


# ------------------------------------------------------------------------------------ arXiv e-prints
MAIN_TEX = r"""\documentclass{article}
\newcommand{\sys}{OmitNet}
\title{Learning What Is Missing}
\begin{document}
\maketitle
\begin{abstract}
We present \sys{}, a model of omissions.
\end{abstract}
\input{intro}
\section{Method}
We minimise $L = \sum_i x_i$ following \citet{smith}. % a comment
\begin{figure}\caption{drop me}\end{figure}
\section*{Limitations}
English only.
\bibliography{refs}
\end{document}
"""


def make_eprint(path, extra_section=""):
    with tarfile.open(path, "w:gz") as tar:
        for name, text in (("main.tex", MAIN_TEX.replace(r"\section*{Limitations}", extra_section + r"\section*{Limitations}")),
                           ("intro.tex", r"\section{Introduction} Omissions matter [see \cite{a}].")):
            blob = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))


def test_eprint_parsing(tmp_path):
    make_eprint(tmp_path / "2101.00001v1")
    d = arxiv_versions.eprint_to_document(tmp_path / "2101.00001v1", arxiv_id="2101.00001", version="v1",
                                          date="2021-01-01", category="cs.CL")
    assert d.document.title == "Learning What Is Missing"
    assert [s.heading for s in d.sections] == ["Abstract", "Introduction", "Method", "Limitations"]
    assert d.sections[0].text == "We present OmitNet, a model of omissions."
    assert "[cite]" in d.sections[1].text and "[math]" in d.sections[2].text
    assert "comment" not in d.text() and "drop me" not in d.text()
    (tmp_path / "single").write_bytes(gzip.compress(MAIN_TEX.replace(r"\input{intro}", "").encode()))
    single = arxiv_versions.eprint_to_document(tmp_path / "single", arxiv_id="x", version="v2", date=None, category=None)
    assert [s.heading for s in single.sections] == ["Abstract", "Method", "Limitations"]


def test_fetch_eprint_is_cached_and_anonymous(tmp_path, monkeypatch):
    import httpx

    calls = []

    def fake_get(url, headers, timeout, follow_redirects):
        calls.append((url, headers))
        return httpx.Response(200, content=b"\x1f\x8bdata", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    p1 = arxiv_versions.fetch_eprint("2101.00001", "v1", tmp_path, min_interval=0)
    p2 = arxiv_versions.fetch_eprint("2101.00001", "v1", tmp_path, min_interval=0)
    assert p1 == p2 and p1.read_bytes() == b"\x1f\x8bdata" and len(calls) == 1
    assert calls[0][0] == "https://export.arxiv.org/e-print/2101.00001v1"
    assert "@" not in calls[0][1]["User-Agent"]  # no personal data sent

    def not_found(url, headers, timeout, follow_redirects):
        calls.append((url, headers))
        return httpx.Response(404, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", not_found)
    for _ in range(2):
        with pytest.raises((httpx.HTTPStatusError, FileNotFoundError)):
            arxiv_versions.fetch_eprint("2101.00002", "v3", tmp_path, min_interval=0)
    assert len(calls) == 2  # the second attempt is answered from the cached 404


# ------------------------------------------------------------------------------------ build
def test_build_privacy_end_to_end(tmp_path):
    from experiments.config import load_config

    make_privacy_db(tmp_path / "db.sqlite", n_sites=40)
    make_opp115(tmp_path / "opp")
    cfg = load_config("configs/data.yaml", ["smoke"])
    cfg["corpus"]["max_documents_per_genre"] = 16
    cfg["paths"].update({k: str(tmp_path / k) for k in ("data_processed", "splits", "tables", "figures")})
    cfg["data_sources"]["privacy"]["privacy_pl"]["db"] = str(tmp_path / "db.sqlite")
    cfg["data_sources"]["privacy"]["opp115"]["dir"] = str(tmp_path / "opp")
    metrics = build_privacy(cfg)
    out = tmp_path / "data_processed" / "privacy"
    docs = list(read_jsonl(out / "documents.jsonl", Document))
    manifest = json.loads((tmp_path / "splits" / "privacy.json").read_text())
    split = {d: s for s, ids in manifest["doc_ids"].items() for d in ids}
    entities = {s: {d.entity_id for d in docs if split[d.id] == s} for s in ("train", "dev", "test")}
    assert not entities["test"] & (entities["train"] | entities["dev"])
    assert {d.entity_id for d in docs if d.source == "opp115"} <= entities["dev"]
    assert sum(d.source == "opp115" for d in docs) == 3  # human-labelled documents are never deduplicated away
    assert metrics["privacy/documents"] == len(docs) and metrics["privacy/test_sha256"] == manifest["test_sha256"]
    assert len(list(read_jsonl(out / "sections.jsonl", Section))) == metrics["privacy/sections"]
    pairs = [json.loads(l) for l in open(out / "revisions.jsonl")]
    assert pairs and all(p["old_date"] < p["new_date"] for p in pairs)
    rev_docs = {d.id for d in read_jsonl(out / "revisions" / "documents.jsonl", Document)}
    assert all(p["old_doc_id"] in rev_docs and p["new_doc_id"] in rev_docs for p in pairs)
    assert {p["entity_id"] for p in pairs} <= entities["test"]
    for table in ("corpus_stats.csv", "dedup.csv", "corpus_sample.csv", "revisions.csv"):
        assert (tmp_path / "tables" / table).is_file()
    assert (tmp_path / "figures" / "loader_check_privacy_pl.html").is_file()


def test_build_papers_end_to_end(tmp_path, monkeypatch):
    from data.build import build_papers
    from experiments.config import load_config

    def paper(i):
        body = [{"section": sec, "sec_number": str(j), "sec_type": "section", "content_type": "paragraph",
                 "text": f"Paper {i} section {sec} paragraph {k} discusses topic {i * 7 + k} in detail."}
                for j, sec in enumerate(("Introduction", "Method", "Results")) for k in range(3)]
        return {"paper_id": f"2001.{i:05d}", "metadata": {"categories": "cs.CL" if i % 2 else "cs.LG cs.AI",
                "title": f"P{i}", "license": "http://creativecommons.org/licenses/by/4.0/",
                "versions": [{"version": "v1", "created": "Wed, 1 Jan 2020 10:00:00 GMT"},
                             {"version": "v2", "created": "Mon, 1 Jun 2020 10:00:00 GMT"}]},
                "abstract": {"text": f"Abstract of paper {i}."}, "body_text": body}

    blob = "".join(json.dumps(paper(i)) + "\n" for i in range(40)).encode()
    with tarfile.open(tmp_path / "u.tar.xz", "w:xz") as tar:
        info = tarfile.TarInfo("20/arXiv_src_2001_001.jsonl")
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))

    def fake_fetch(arxiv_id, version, cache_dir, **kw):
        path = tmp_path / "eprints" / f"{arxiv_id}{version}"
        path.parent.mkdir(exist_ok=True)
        make_eprint(path, extra_section="" if version == "v1" else
                    r"\section{Ethics Statement} We release code and data under an open licence, and we discuss risks "
                    r"of misuse at length, including deployment harms, dual use and the limits of our evaluation. ")
        return path

    monkeypatch.setattr(arxiv_versions, "fetch_eprint", fake_fetch)
    cfg = load_config("configs/data.yaml", ["smoke"])
    cfg["corpus"]["max_documents_per_genre"] = 24
    cfg["paths"].update({k: str(tmp_path / k) for k in ("data_processed", "splits", "tables", "figures")})
    src = cfg["data_sources"]["papers"]
    src["unarxive"].update(archive=str(tmp_path / "u.tar.xz"), extract=str(tmp_path / "x.jsonl.gz"))
    src["arxiv_eprints"]["max_pairs"] = None
    metrics = build_papers(cfg)
    out = tmp_path / "data_processed" / "papers"
    docs = list(read_jsonl(out / "documents.jsonl", Document))
    assert metrics["papers/documents"] == len(docs) and len(docs) >= 30
    assert {d.category for d in docs} == {"cs.CL", "cs.LG"} and all(d.version == "v2" for d in docs)
    manifest = json.loads((tmp_path / "splits" / "papers.json").read_text())
    test_ids = {d.split(":")[1] for d in manifest["doc_ids"]["test"]}
    pairs = [json.loads(l) for l in open(out / "revisions.jsonl")]
    assert pairs and {p["entity_id"] for p in pairs} <= test_ids
    assert metrics["papers/revision_candidates"] == len(test_ids)
    rev = {d.id: d for d in read_jsonl(out / "revisions" / "documents.jsonl", Document)}
    assert all(rev[p["old_doc_id"]].version == "v1" and rev[p["new_doc_id"]].version == "v2" for p in pairs)
