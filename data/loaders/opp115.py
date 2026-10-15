"""Loader for OPP-115 (Wilson et al., 2016), the human-annotated set for genre A (plan §2.4, §4.1).

Raw layout (OPP-115_v1_0.zip, documented in documentation/manual.txt and data/README.md):
  sanitized_policies/<policy UID>_<domain>.html   segments separated by "|||" (HTML: br, strong, ul/li)
  consolidation/threshold-<t>-overlap-similarity/<UID>_<domain>.csv
      columns: annotation ID, batch ID, annotator ID, policy ID, segment ID, category name,
               attribute-value JSON, date, policy URL (Excel dialect, no header)
  documentation/policies_opp115.csv   Policy UID, Policy URL, collection date, last updated date
  documentation/websites_opp115.csv   Site UID, Site URL, name, Policy UID, ..., DMOZ sectors

Sections are the native annotation segments, never re-segmented: Section.idx == OPP-115 segment ID, so the
labels from `iter_labels` line up with sections exactly.
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any, Iterator

from data.schema import Document, ParsedDocument, Section, with_char_spans

SOURCE = "opp115"
ROOT = "OPP-115"


def _root(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    return raw_dir / ROOT if (raw_dir / ROOT).is_dir() else raw_dir


def normalise_domain(url_or_domain: str) -> str:
    d = re.sub(r"^[a-z]+://", "", url_or_domain.strip().lower()).split("/")[0]
    return d[4:] if d.startswith("www.") else d


_BLOCK_END = re.compile(r"</(p|li|ul|ol|h\d|div|blockquote)>|<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")
_STRONG = re.compile(r"^\s*<strong>(.*?)</strong>", re.I | re.S)


def segment_to_text(segment_html: str) -> tuple[str | None, str]:
    """(heading, text) for one sanitized segment. A leading <strong> run of <= 15 words is the heading."""
    heading = None
    m = _STRONG.match(segment_html)
    if m:
        candidate = html.unescape(_TAG.sub(" ", m.group(1)))
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate and len(candidate.split()) <= 15:
            heading = candidate
            segment_html = segment_html[m.end():]
    text = _BLOCK_END.sub("\n", segment_html)
    text = html.unescape(_TAG.sub(" ", text))
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.split("\n")]
    lines = [re.sub(r"\s+([.,;:])", r"\1", ln) for ln in lines if ln]
    return heading, "\n".join(lines)


def _policy_index(root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with open(root / "documentation" / "policies_opp115.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            uid = (row.get("Policy UID") or "").strip()
            if uid:
                index[uid] = {"url": row.get("Policy URL", "").strip(),
                              "collection_date": row.get("Policy collection date", "").strip() or None,
                              "last_updated": row.get("Policy last updated date", "").strip() or None}
    with open(root / "documentation" / "websites_opp115.csv", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) > 3 and row[3].strip() in index:
                index[row[3].strip()].update(site_url=row[1].strip(), site_name=row[2].strip(),
                                             sectors=[s for s in row[7:] if s.strip()])
    return index


def iter_documents(raw_dir: str | Path, *, limit: int | None = None, genre: str = "privacy") -> Iterator[ParsedDocument]:
    root = _root(raw_dir)
    index = _policy_index(root)
    for i, path in enumerate(sorted((root / "sanitized_policies").glob("*.html"), key=lambda p: int(p.stem.split("_")[0]))):
        if limit is not None and i >= limit:
            break
        uid, domain = path.stem.split("_", 1)
        info = index.get(uid, {})
        doc_id = f"{SOURCE}:{uid}"
        segments = path.read_text(encoding="utf-8").split("|||")
        sections = []
        for seg_id, seg in enumerate(segments):
            heading, text = segment_to_text(seg)
            sections.append(Section(doc_id=doc_id, idx=seg_id, heading=heading, text=text))
        doc = Document(
            id=doc_id, genre=genre, source=SOURCE, entity_id=normalise_domain(info.get("site_url") or domain),
            version=None, date=info.get("collection_date"), category=None, title=None,
            meta={"policy_uid": uid, "policy_url": info.get("url"), "last_updated": info.get("last_updated"),
                  "site_name": info.get("site_name"), "dmoz_sectors": info.get("sectors", [])[:5]},
        )
        yield ParsedDocument(doc, with_char_spans(sections))


def iter_labels(raw_dir: str | Path, *, threshold: str = "0.75") -> Iterator[dict[str, Any]]:
    """One record per consolidated data practice: doc_id, section_idx (= segment ID), category, and the
    attribute values (spans dropped; "not-selected" values omitted)."""
    root = _root(raw_dir)
    folder = root / "consolidation" / f"threshold-{threshold}-overlap-similarity"
    for path in sorted(folder.glob("*.csv"), key=lambda p: int(p.stem.split("_")[0])):
        uid = path.stem.split("_", 1)[0]
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 7:
                    continue
                attrs = json.loads(row[6]) if row[6].strip() else {}
                values = {k: v.get("value") for k, v in attrs.items()
                          if isinstance(v, dict) and v.get("value") not in (None, "not-selected")}
                yield {"doc_id": f"{SOURCE}:{uid}", "section_idx": int(row[4]), "category": row[5],
                       "attributes": values, "annotation_id": row[0]}
