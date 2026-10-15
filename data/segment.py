"""Section and sentence segmentation, boilerplate removal (plan §2.5).

Three entry points, all genre-agnostic (they look at text format, not at what the text is about):

  segment_markdown(text)        Markdown or plain text -> [(heading, body)]: ATX headings, bold-line headings,
                                numbered and ALL-CAPS heading lines; runs of headings with no body (tables of
                                contents) are dropped; no headings at all -> paragraph chunks.
  normalise_sections(...)       merge sections below min_section_chars, split those above max_section_chars
                                at paragraph breaks (split_oversized, the split step alone, is used by the
                                loaders whose sources already mark sections: unarXive, arXiv e-prints).
  split_sentences(text)         rule-based splitter with an abbreviation list. Chosen over pysbd (accurate but
                                333 sentences/s on the M5, ~4 h for the corpus) and spaCy's sentencizer
                                (50k/s but splits after "Sec.", "Fig.", "et al.,"); see CHANGELOG 2026-10-12.

clean_inline() removes Markdown link/emphasis syntax, escapes, images and unarXive placeholders
({{formula:..}} -> [math], {{cite:..}} -> [cite], {{ref:..}}/{{figure:..}}/{{table:..}} -> [ref]).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ------------------------------------------------------------------------------------ inline cleaning
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
_AUTOLINK = re.compile(r"<(https?://[^>]+)>")
_EMPH = re.compile(r"(\*\*|__|(?<![\w*])\*(?=\S)|(?<=\S)\*(?![\w*])|(?<!\w)_(?=\S)|(?<=\S)_(?!\w))")
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])")
_UNARXIVE = [
    (re.compile(r"\{\{formula:[^}]*\}\}"), "[math]"),
    (re.compile(r"\{\{cite:[^}]*\}\}"), "[cite]"),
    (re.compile(r"\{\{(?:ref|figure|table):[^}]*\}\}"), "[ref]"),
]
_WS = re.compile(r"[ \t ]+")


_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(\s[^<>]{0,300})?/?>")


def clean_inline(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _AUTOLINK.sub(r"\1", text)
    for pattern, repl in _UNARXIVE:
        text = pattern.sub(repl, text)
    text = _ESCAPE.sub(r"\1", text)
    text = _EMPH.sub("", text)
    text = re.sub(r"^\|\s*|\s*\|$", "", text.strip())
    return _WS.sub(" ", text).strip()


_NOISE_LINE = re.compile(r"^\s*([#*_\-=|]\s*)+$")  # empty headings, horizontal rules, table borders


# ------------------------------------------------------------------------------------ boilerplate
_PREAMBLE = re.compile(r"^>\s*\*?The following text is extracted and transformed from", re.I)
_BOILER_LINE = re.compile(
    r"^(skip to (main )?content|back to top|print( this page)?|share( this)?( on)?\b.*|"
    r"(©|\(c\)|copyright)\s*(\d{4})?.*|all rights reserved\.?|"
    r"(home|about( us)?|contact( us)?|login|log in|sign in|sign up|search|menu)(\s*[|·•/]\s*\w+.*)?)$",
    re.I,
)
_COOKIE_BANNER = re.compile(
    r"(accept( all)? cookies|by (continuing|using) (to (use|browse) )?(this|our) (site|website).{0,80}cookies|"
    r"this (site|website) uses cookies.{0,120}(accept|agree|continue|ok\b))",
    re.I,
)


def remove_boilerplate(lines: list[str]) -> list[str]:
    """Drop navigation, copyright, mirror-preamble lines anywhere, and cookie-banner lines near either end
    of the document (cookie sentences in the body are content and are kept)."""
    n = len(lines)
    edge = max(3, n // 10)
    out = []
    for i, line in enumerate(lines):
        stripped = clean_inline(line.strip().lstrip(">").strip())
        if _PREAMBLE.search(line.strip()):
            continue
        if stripped and len(stripped) < 120 and _BOILER_LINE.match(stripped):
            continue
        if (i < edge or i >= n - edge) and len(stripped) < 400 and _COOKIE_BANNER.search(stripped):
            continue
        out.append(line)
    return out


# ------------------------------------------------------------------------------------ headings
_ATX = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_BOLD_LINE = re.compile(r"^\s*(\*\*|__)(.+?)(\*\*|__)\s*:?\s*$")
_NUMBERED = re.compile(r"^\s*((\d{1,2}(\\?\.\d{1,2})*\\?[.)])|([IVXivx]{1,5}\\?[.)])|([A-H]\\?[.)]))\s+(\S.{0,110})$")
_LIST_ITEM = re.compile(r"^\s*([*+\-•]|\d{1,3}\\?[.)])\s+")


def _title_line(line: str) -> str | None:
    """A plain short line that reads like a title ("Applicability", "Information We Collect"). Only accepted as a
    heading by segment_markdown when a body paragraph follows it."""
    text = clean_inline(line)
    words = text.split()
    if not (1 <= len(words) <= 8) or len(text) > 70 or not text[0].isupper() or text.endswith((".", ",", ";", "!")):
        return None
    if re.search(r"\b(19|20)\d{2}\b", text):  # "Last updated May 24, 2018" is a date line, not a title
        return None
    long_words = [w for w in words if len(w) > 3 and w[0].isalpha()]
    capitalised = sum(w[0].isupper() for w in long_words) / len(long_words) if long_words else 1.0
    if text.endswith(":") and capitalised < 0.6:
        return None
    if capitalised < 0.5 and not text.endswith("?"):
        return None
    return text.rstrip(":").strip()


def _heading_text(line: str) -> str | None:
    """Return the heading text if the line is a heading, else None."""
    raw = line.strip()
    if not raw or len(raw) > 160:
        return None
    m = _ATX.match(raw)
    if m:
        return clean_inline(m.group(2)) or None
    m = _BOLD_LINE.match(raw)
    if m:
        inner = clean_inline(m.group(2))
        if 1 <= len(inner.split()) <= 15 and not inner.endswith((".", ",", ";")):
            return inner
        if re.match(r"^\d{1,2}(\.\d{1,2})*\.?\s", inner) and len(inner.split()) <= 15:
            return inner
        return None
    text = clean_inline(raw)
    if not text or len(text.split()) > 14 or text.endswith((",", ";")):
        return None
    if _NUMBERED.match(raw) and not text.endswith(".") and text[0:1].isalnum():
        body = _NUMBERED.match(raw).group(6)
        long_words = [w for w in clean_inline(body).split() if len(w) > 3 and w[0].isalpha()]
        capitalised = sum(w[0].isupper() for w in long_words) / len(long_words) if long_words else 1.0
        if body[:1].isupper() and (capitalised >= 0.5 or body.rstrip().endswith("?")):  # not a numbered list item
            return text
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters) and not text.endswith("."):
        return text
    return None


@dataclass
class RawSection:
    heading: str | None
    paragraphs: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


def segment_markdown(text: str, *, fallback_chars: int = 1500) -> list[RawSection]:
    """Split a Markdown/plain-text document into heading-delimited sections (see module doc)."""
    lines = remove_boilerplate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    lines = [ln for ln in lines if ln.strip() and not _NOISE_LINE.match(ln)]
    sections: list[RawSection] = []
    current = RawSection(heading=None, paragraphs=[])
    n_headings = 0
    for i, line in enumerate(lines):
        heading = _heading_text(line) if not _LIST_ITEM.match(line) or _NUMBERED.match(line) else None
        if heading is None and not _LIST_ITEM.match(line) and i + 1 < len(lines):
            following = clean_inline(lines[i + 1])
            if len(following) >= 60 and _heading_text(lines[i + 1]) is None:
                heading = _title_line(line)
        if heading is not None:
            n_headings += 1
            if current.paragraphs or current.heading is not None:
                sections.append(current)
            current = RawSection(heading=heading, paragraphs=[])
            continue
        para = clean_inline(_LIST_ITEM.sub("", line, count=1) if _LIST_ITEM.match(line) and not _NUMBERED.match(line) else line)
        if para:
            current.paragraphs.append(para)
    sections.append(current)
    # headings with no body: a table of contents, or a heading directly followed by a sub-heading.
    # Keep the heading text only when it is the parent of the next section (joined as "A / B").
    merged: list[RawSection] = []
    pending: list[str] = []
    for s in sections:
        if not s.paragraphs:
            if s.heading:
                pending.append(s.heading)
            continue
        if pending and s.heading is not None and len(pending) == 1:
            s = RawSection(heading=f"{pending[0]} / {s.heading}", paragraphs=s.paragraphs)
        elif pending and s.heading is None:
            s = RawSection(heading=pending[-1], paragraphs=s.paragraphs)
        pending = []
        merged.append(s)
    if n_headings == 0 and merged:
        return chunk_paragraphs(merged[0].paragraphs, target_chars=fallback_chars)
    return merged


def chunk_paragraphs(paragraphs: list[str], *, target_chars: int, heading: str | None = None) -> list[RawSection]:
    """Group paragraphs into chunks of about target_chars (never splitting a paragraph)."""
    out, buf, size = [], [], 0
    for p in paragraphs:
        if buf and size + len(p) > target_chars:
            out.append(RawSection(heading=heading, paragraphs=buf))
            buf, size = [], 0
        buf.append(p)
        size += len(p) + 1
    if buf:
        out.append(RawSection(heading=heading, paragraphs=buf))
    return out


def normalise_sections(sections: list[RawSection], *, min_chars: int = 80, max_chars: int = 8000) -> list[RawSection]:
    """Merge tiny sections into the following one (or the previous, at the end); split oversized ones."""
    out: list[RawSection] = []
    carry: RawSection | None = None
    for s in sections:
        if carry is not None:
            s = RawSection(heading=carry.heading or s.heading if carry.heading and not s.heading else s.heading,
                           paragraphs=carry.paragraphs + s.paragraphs)
            carry = None
        if len(s.text) < min_chars:
            carry = s
            continue
        out.append(s)
    if carry is not None:
        if out:
            out[-1] = RawSection(heading=out[-1].heading, paragraphs=out[-1].paragraphs + carry.paragraphs)
        else:
            out.append(carry)
    return split_oversized(out, max_chars=max_chars)


def _wrap(text: str, width: int) -> list[str]:
    """Cut a string with no usable sentence boundary (a table, a listing) at spaces into pieces of at most width."""
    pieces, buf = [], ""
    for word in text.split(" "):
        while len(word) > width:  # a single token longer than width: cut it hard
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(word[:width])
            word = word[width:]
        if buf and len(buf) + 1 + len(word) > width:
            pieces.append(buf)
            buf = ""
        buf = f"{buf} {word}" if buf else word
    if buf:
        pieces.append(buf)
    return pieces


def split_oversized(sections: list[RawSection], *, max_chars: int | None) -> list[RawSection]:
    """Split every section longer than max_chars at paragraph breaks into chunks of about max_chars / 2, keeping its
    heading; a paragraph longer than max_chars is split at sentence boundaries first (and a sentence longer than
    max_chars at spaces). Shorter sections are returned unchanged; max_chars None disables the step. Shared by every
    loader, so long documents become more rows in every genre (plan §0.2 #4)."""
    if not max_chars:
        return sections
    target = max_chars // 2
    final: list[RawSection] = []
    for s in sections:
        if len(s.text) <= max_chars:
            final.append(s)
            continue
        paragraphs = []
        for p in s.paragraphs:
            if len(p) <= max_chars:
                paragraphs.append(p)
                continue
            sentences = [piece for x in split_sentences(p)
                         for piece in (_wrap(x, target) if len(x) > max_chars else [x])]
            paragraphs += [c.text for c in chunk_paragraphs(sentences, target_chars=target)]
        final += chunk_paragraphs(paragraphs, target_chars=target, heading=s.heading)
    return final


# ------------------------------------------------------------------------------------ sentences
ABBREVIATIONS = frozenset("""
e.g i.e etc al et.al cf vs viz approx resp ca fig figs sec secs sect eq eqs eqn tab tbl ref refs ch chap art para
no nos vol pp p inc ltd corp co llc plc dept est mr mrs ms dr prof st jr sr u.s u.k e.u u.s.a jan feb mar apr jun
jul aug sep sept oct nov dec min max avg std approx w.r.t
""".split())
_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])")
_TOKEN_BEFORE = re.compile(r"(\S+)$")


def split_sentences(text: str, *, min_chars: int = 15) -> list[str]:
    """Split on . ! ? followed by space + capital/digit, except after abbreviations and single-letter initials.
    Line breaks are always boundaries (list items, headings). Fragments (shorter than min_chars and with fewer than
    two words, e.g. "1." or "[math]") are merged into the previous sentence."""
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        start = 0
        for m in _BOUNDARY.finditer(line):
            prev = _TOKEN_BEFORE.search(line[start:m.start() + 1])
            token = prev.group(1).lower().rstrip(".!?…").lstrip("(\"'“‘[") if prev else ""
            if token in ABBREVIATIONS or (len(token) == 1 and token.isalpha()) or re.fullmatch(r"(\w\.)+\w", token):
                continue
            sentences.append(line[start:m.end()].strip())
            start = m.end()
        tail = line[start:].strip()
        if tail:
            sentences.append(tail)
    def fragment(x: str) -> bool:
        return len(x) < min_chars and len(re.findall(r"[A-Za-z]{2,}", x)) < 2

    merged: list[str] = []
    for s in sentences:
        if merged and (fragment(s) or fragment(merged[-1])):
            merged[-1] = f"{merged[-1]} {s}"
        else:
            merged.append(s)
    return merged
