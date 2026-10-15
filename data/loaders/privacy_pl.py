"""Loader for the Princeton–Leuven Longitudinal Corpus of Privacy Policies (Amos et al., 2021), genre A (plan §2.4).

Raw file: release_db.sqlite (from release_db.sqlite.xz). Real schema, checked 2026-10-12 (data/README.md):
  sites(id, domain, categories)                        categories: Webshrinker, ';'-separated, alphabetical
  policy_snapshots(id, site_id, homepage_snapshot_url, policy_snapshot_url, policy_url,
                   homepage_snapshot_redirected_url, year, phase, policy_text_id, policy_html_id,
                   policy_reader_view_html_id, file_type, policy_title, link_text, pdf_filename,
                   classifier_probability, analysis_subcorpus, parked_domain, cross_domain_homepage_redir)
                   at most one snapshot per (site_id, year, phase); phase A = Jan-Jun, B = Jul-Dec
  policy_texts(id, policy_text, flesch_kincaid, smog, flesch_ease, length, sha1, simhash)
                   Markdown text; snapshots with an unchanged policy share one policy_texts row

Access pattern: `snapshot_index` reads metadata only (no text) so sampling never touches the 48 GB file's content;
`load_texts` fetches the chosen texts in batches; `to_parsed` segments one text into Sections.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from data.loaders.opp115 import normalise_domain
from data.schema import Document, ParsedDocument, Section, with_char_spans
from data.segment import normalise_sections, segment_markdown

SOURCE = "privacy_pl"
GENERIC_CATEGORIES = {"business", "uncategorized"}
EXCLUDED_CATEGORIES = {"parked", "illegalcontent"}
_WAYBACK_TS = re.compile(r"/web/(\d{4})(\d{2})(\d{2})\d*")


def connect(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db_path)}?mode=ro&immutable=1", uri=True)


def snapshot_date(policy_snapshot_url: str | None, year: int | None = None, phase: str | None = None) -> str | None:
    m = _WAYBACK_TS.search(policy_snapshot_url or "")
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    if year:
        return f"{year}-{'01' if phase == 'A' else '07'}-01"
    return None


@dataclass
class SnapshotRow:
    snapshot_id: int
    site_id: int
    domain: str
    categories: list[str]
    year: int
    phase: str
    date: str | None
    policy_url: str | None
    snapshot_url: str | None
    title: str | None
    text_id: int
    length: int
    sha1: str
    category: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def entity_id(self) -> str:
        return normalise_domain(self.domain)


def category_frequencies(con: sqlite3.Connection) -> Counter:
    freq: Counter = Counter()
    for (cats,) in con.execute("select categories from sites"):
        freq.update(c for c in (cats or "").split(";") if c)
    return freq


def primary_category(categories: list[str], freq: Counter) -> str | None:
    """Webshrinker lists are alphabetical, so there is no stated primary. Use the most specific (rarest) category,
    ignoring the catch-alls; fall back to a catch-all if that is all there is."""
    specific = [c for c in categories if c not in GENERIC_CATEGORIES]
    if specific:
        return min(specific, key=lambda c: (freq.get(c, 0), c))
    return categories[0] if categories else None


def snapshot_index(con: sqlite3.Connection, *, analysis_subcorpus_only: bool = True,
                   file_types: Iterable[str] = ("html",), min_year: int | None = None,
                   max_year: int | None = None) -> list[SnapshotRow]:
    freq = category_frequencies(con)
    where = ["s.parked_domain = 0"]
    if analysis_subcorpus_only:
        where.append("s.analysis_subcorpus = 1")
    types = [t.lower() for t in file_types]
    where.append(f"lower(s.file_type) in ({','.join('?' * len(types))})")
    params: list[Any] = list(types)
    if min_year:
        where.append("s.year >= ?")
        params.append(min_year)
    if max_year:
        where.append("s.year <= ?")
        params.append(max_year)
    sql = f"""select s.id, s.site_id, st.domain, st.categories, s.year, s.phase, s.policy_snapshot_url, s.policy_url,
                     s.policy_title, s.policy_text_id, t.length, t.sha1
              from policy_snapshots s join sites st on st.id = s.site_id join policy_texts t on t.id = s.policy_text_id
              where {' and '.join(where)}"""
    rows = []
    for r in con.execute(sql, params):
        cats = [c for c in (r[3] or "").split(";") if c]
        if EXCLUDED_CATEGORIES & set(cats):
            continue
        row = SnapshotRow(snapshot_id=r[0], site_id=r[1], domain=r[2], categories=cats, year=r[4], phase=r[5],
                          date=snapshot_date(r[6], r[4], r[5]), snapshot_url=r[6], policy_url=r[7], title=r[8],
                          text_id=r[9], length=r[10] or 0, sha1=r[11] or "")
        row.category = primary_category(cats, freq)
        rows.append(row)
    return rows


def one_per_entity_year(rows: Iterable[SnapshotRow]) -> list[SnapshotRow]:
    """Keep the latest snapshot of each (entity, year): phase B over A."""
    best: dict[tuple[str, int], SnapshotRow] = {}
    for r in rows:
        key = (r.entity_id, r.year)
        if key not in best or (r.phase, r.date or "") > (best[key].phase, best[key].date or ""):
            best[key] = r
    return list(best.values())


def load_texts(con: sqlite3.Connection, text_ids: Iterable[int], *, batch: int = 900) -> dict[int, str]:
    ids = sorted(set(text_ids))
    out: dict[int, str] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        q = f"select id, policy_text from policy_texts where id in ({','.join('?' * len(chunk))})"
        out.update({tid: text or "" for tid, text in con.execute(q, chunk)})
    return out


def to_parsed(row: SnapshotRow, text: str, *, genre: str = "privacy", min_section_chars: int = 80,
              max_section_chars: int = 8000, fallback_section_chars: int = 1500) -> ParsedDocument:
    doc_id = f"{SOURCE}:{row.snapshot_id}"
    raw = normalise_sections(segment_markdown(text, fallback_chars=fallback_section_chars),
                             min_chars=min_section_chars, max_chars=max_section_chars)
    sections = [Section(doc_id=doc_id, idx=0, heading=s.heading, text=s.text) for s in raw]
    doc = Document(
        id=doc_id, genre=genre, source=SOURCE, entity_id=row.entity_id, version=f"{row.year}{row.phase}",
        date=row.date, category=row.category, title=(row.title or "").strip() or None,
        meta={"categories": row.categories, "policy_url": row.policy_url, "snapshot_url": row.snapshot_url,
              "policy_text_id": row.text_id, "text_sha1": row.sha1, "length": row.length, **row.meta},
    )
    return ParsedDocument(doc, with_char_spans(sections))


def iter_documents(con: sqlite3.Connection, rows: list[SnapshotRow], *, genre: str = "privacy",
                   batch: int = 500, **segment_kw: Any) -> Iterator[ParsedDocument]:
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        texts = load_texts(con, (r.text_id for r in chunk))
        for r in chunk:
            yield to_parsed(r, texts.get(r.text_id, ""), genre=genre, **segment_kw)
