"""Loader for unarXive (open subset, release 230324; Saier, Krause & Färber 2023) for genre B (plan §2.4).

Raw layout (data/README.md): unarXive_230324_open_subset.tar.xz -> <yy>/arXiv_src_<yymm>_<num>.jsonl, one paper per
line with fields paper_id, metadata (the Kaggle arXiv snapshot record: categories, versions, license, title, ...),
abstract, body_text (paragraphs with section, sec_number, sec_type, content_type, text), bib_entries, ref_entries.

Two steps, because the archive is 4.8 GB of xz:
  extract(tar_path, out_path, ...)   one streaming pass; keeps papers whose primary category is in `categories` and
                                     whose v1 is on/after `min_date`; drops bib/ref entries; gzip jsonl
  iter_documents(extract_path)       Document + Sections (consecutive paragraphs of one section form a Section)
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
import tarfile
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterator

from data.schema import Document, ParsedDocument, Section, with_char_spans
from data.segment import clean_inline

SOURCE = "unarxive"
KEEP_CONTENT = {"paragraph", "listing"}
# unarXive renders \\ref{} as "REF": "Figure REF", "Eq. (REF )", and glued "TableREF", "AREF" (from Table~\\ref{..})
_BARE_REF = re.compile(r"(?:(?<![A-Za-z])|(?<=[a-z])|(?<=(?<![A-Za-z])[A-Z]))REF\b(\s(?=[)\],.;:]))?")


def _clean(text: str) -> str:
    text = clean_inline(text)
    return _BARE_REF.sub(lambda m: (" [ref]" if m.start() and text[m.start() - 1].isalpha() else "[ref]"), text)


def _date(created: str | None) -> str | None:
    if not created:
        return None
    try:
        return parsedate_to_datetime(created).date().isoformat()
    except (TypeError, ValueError):
        try:
            return datetime.strptime(created[:10], "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None


def primary_category(meta: dict[str, Any]) -> str | None:
    cats = (meta.get("categories") or "").split()
    return cats[0] if cats else None


def versions(meta: dict[str, Any]) -> list[tuple[str, str | None]]:
    """[(label, ISO date)] in order, e.g. [("v1", "2019-03-02"), ("v2", "2019-06-11")]."""
    return [(v.get("version"), _date(v.get("created"))) for v in meta.get("versions") or []]


def extract(tar_path: str | Path, out_path: str | Path, *, categories: set[str], min_date: str | None = None,
            max_date: str | None = None, limit: int | None = None) -> dict[str, int]:
    """Stream the archive once and write the filtered papers. Returns counts."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = kept = skipped_files = 0
    # cheap byte-level prefilter (primary category = first token of metadata.categories); json only for candidates
    prefilter = re.compile(rb'"categories":\s*"(?:' + b"|".join(re.escape(c.encode()) for c in sorted(categories)) + rb')[ "]')
    min_year = int(min_date[:4]) if min_date else None
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    proc = None
    if shutil.which("xz"):  # the archive is multi-block xz: system xz decompresses on all cores
        proc = subprocess.Popen(["xz", "-T0", "-dc", str(tar_path)], stdout=subprocess.PIPE)
        tar_cm = tarfile.open(fileobj=proc.stdout, mode="r|")
    else:
        tar_cm = tarfile.open(tar_path, "r|xz")
    with tar_cm as tar, gzip.open(tmp, "wt", encoding="utf-8") as out:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".jsonl"):
                continue
            year_dir = Path(member.name).parts[-2] if len(Path(member.name).parts) > 1 else ""
            if min_year is not None and len(year_dir) == 2 and year_dir.isdigit():
                full_year = (1900 if int(year_dir) >= 90 else 2000) + int(year_dir)  # <yy> = year of the paper's v1
                if full_year < min_year:
                    skipped_files += 1
                    continue
            f = tar.extractfile(member)
            if f is None:
                continue
            for raw in f:
                seen += 1
                if not prefilter.search(raw):
                    continue
                paper = json.loads(raw)
                meta = paper.get("metadata") or {}
                if primary_category(meta) not in categories:
                    continue
                vs = versions(meta)
                first = vs[0][1] if vs else None
                if min_date and (first is None or first < min_date):
                    continue
                if max_date and first and first > max_date:
                    continue
                paper.pop("bib_entries", None)
                paper.pop("ref_entries", None)
                for p in paper.get("body_text") or []:
                    p.pop("cite_spans", None)
                    p.pop("ref_spans", None)
                out.write(json.dumps(paper, ensure_ascii=False) + "\n")
                kept += 1
                if limit is not None and kept >= limit:
                    break
            if limit is not None and kept >= limit:
                break
    if proc is not None:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    tmp.rename(out_path)
    return {"papers_seen": seen, "papers_kept": kept, "files_skipped_by_year": skipped_files}


def _abstract_text(abstract: Any) -> str:
    if isinstance(abstract, dict):
        return abstract.get("text") or ""
    return abstract or ""


def paper_to_document(paper: dict[str, Any], *, genre: str = "papers", cutoff: str = "2022-12-31") -> ParsedDocument:
    meta = paper.get("metadata") or {}
    pid = paper["paper_id"]
    vs = [(label, d) for label, d in versions(meta) if d is None or d <= cutoff]
    label, date = vs[-1] if vs else (None, None)
    doc_id = f"{SOURCE}:{pid}"
    sections: list[Section] = []
    abstract = _clean(_abstract_text(paper.get("abstract")))
    if abstract:
        sections.append(Section(doc_id=doc_id, idx=0, heading="Abstract", text=abstract))
    current_heading, buf = None, []
    for p in paper.get("body_text") or []:
        if p.get("content_type", "paragraph") not in KEEP_CONTENT:
            continue
        text = _clean(p.get("text") or "")
        if not text:
            continue
        heading = clean_inline(p.get("section") or "") or None
        if buf and heading != current_heading:
            sections.append(Section(doc_id=doc_id, idx=0, heading=current_heading, text="\n".join(buf)))
            buf = []
        current_heading = heading
        buf.append(text)
    if buf:
        sections.append(Section(doc_id=doc_id, idx=0, heading=current_heading, text="\n".join(buf)))
    doc = Document(
        id=doc_id, genre=genre, source=SOURCE, entity_id=pid, version=label, date=date,
        category=primary_category(meta), title=(meta.get("title") or "").strip() or None,
        meta={"categories": meta.get("categories"), "license": meta.get("license"), "versions": vs,
              "all_versions": [v for v, _ in versions(meta)], "discipline": paper.get("discipline"),
              "source_hash": paper.get("_source_hash")},
    )
    return ParsedDocument(doc, with_char_spans(sections))


def iter_documents(extract_path: str | Path, *, limit: int | None = None, genre: str = "papers",
                   ids: set[str] | None = None, cutoff: str = "2022-12-31") -> Iterator[ParsedDocument]:
    with gzip.open(extract_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            paper = json.loads(line)
            if ids is not None and paper["paper_id"] not in ids:
                continue
            yield paper_to_document(paper, genre=genre, cutoff=cutoff)


def iter_metadata(extract_path: str | Path, *, cutoff: str = "2022-12-31") -> Iterator[dict[str, Any]]:
    """Light first pass for sampling: id, date and version of the text, primary category, all versions, body size."""
    with gzip.open(extract_path, "rt", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            meta = paper.get("metadata") or {}
            vs = [(label, d) for label, d in versions(meta) if d is None or d <= cutoff]
            yield {"paper_id": paper["paper_id"], "category": primary_category(meta),
                   "version": vs[-1][0] if vs else None, "date": vs[-1][1] if vs else None, "versions": vs,
                   "license": meta.get("license"),
                   "n_paragraphs": sum(1 for p in paper.get("body_text") or []
                                       if p.get("content_type", "paragraph") in KEEP_CONTENT and p.get("text"))}
