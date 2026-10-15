"""Document / Section / Sentence / Assertion records and jsonl I/O (plan §2.3).

Every corpus is converted into these records by its loader (data/loaders/), so nothing downstream knows
which corpus a document came from. Field meanings:

  Document  id           globally unique: "<source>:<native id>[:<version>]"
            genre        pack name ("privacy" | "papers"); the engine only ever compares within a genre
            source       corpus name ("privacy_pl" | "opp115" | "unarxive")
            entity_id    what versions belong to: website domain for policies, arXiv id for papers
            version      position in the entity's chain (policies: snapshot label; papers: "v1", "v2", ...)
            date         ISO date (YYYY-MM-DD) of this version
            category     website / app category or arXiv primary category; None when unknown
            title        document title if the corpus has one
            meta         corpus-specific extras, kept verbatim for provenance (never read by the engine)
  Section   heading is None when the source has no heading; char_span is (start, end) into the
            document's concatenated section text (sections joined by "\\n\\n").
  Assertion produced from Phase 3 on; prob is the calibrated P(present).
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

SECTION_JOINER = "\n\n"


@dataclass
class Document:
    id: str
    genre: str
    source: str
    entity_id: str
    version: str | None = None
    date: str | None = None
    category: str | None = None
    title: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    doc_id: str
    idx: int
    heading: str | None
    text: str
    char_span: tuple[int, int] = (0, 0)


@dataclass
class Sentence:
    doc_id: str
    section_idx: int
    idx: int
    text: str


@dataclass
class Assertion:
    doc_id: str
    section_idx: int
    type_id: str
    prob: float
    evidence_sentence_ids: list[int] = field(default_factory=list)


@dataclass
class RevisionPair:
    """Two consecutive, materially different versions of one entity (plan §2.6b, OmitBench-revision)."""

    entity_id: str
    genre: str
    old_doc_id: str
    new_doc_id: str
    old_date: str | None
    new_date: str | None
    change_ratio: float  # 1 - similarity of the two versions' normalised text


@dataclass
class ParsedDocument:
    """What a loader yields: one document with its sections, before sentence splitting."""

    document: Document
    sections: list[Section]

    def text(self) -> str:
        return SECTION_JOINER.join(s.text for s in self.sections)


def with_char_spans(sections: list[Section]) -> list[Section]:
    """Fill char_span offsets for sections joined with SECTION_JOINER; renumbers idx from 0."""
    out, pos = [], 0
    for i, s in enumerate(sections):
        out.append(Section(doc_id=s.doc_id, idx=i, heading=s.heading, text=s.text,
                           char_span=(pos, pos + len(s.text))))
        pos += len(s.text) + len(SECTION_JOINER)
    return out


# ------------------------------------------------------------------------------------ jsonl
R = TypeVar("R", Document, Section, Sentence, Assertion, RevisionPair)


def _open(path: Path, mode: str):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r) if hasattr(r, "__dataclass_fields__") else r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, cls: type[R] | None = None) -> Iterator[R | dict[str, Any]]:
    with _open(Path(path), "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if cls is Section and "char_span" in row:
                row["char_span"] = tuple(row["char_span"])
            yield cls(**row) if cls is not None else row
