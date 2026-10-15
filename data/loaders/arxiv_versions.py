"""Version-specific arXiv full texts for OmitBench-revision, genre B (plan §2.1, §2.6b, §0.2 #16).

unarXive holds one version per paper; a revision pair needs two. Both versions are therefore fetched as LaTeX
e-prints from arXiv and parsed by the same function, so differences between them are real edits, not parser
differences.

  fetch_eprint(arxiv_id, version, cache_dir)  GET https://export.arxiv.org/e-print/<id><version>, cached on disk,
                                              at most one request every `min_interval` seconds (arXiv's guidance
                                              for programmatic access); no personal data in the User-Agent
  eprint_to_document(path, ...)               tar.gz or single gzipped .tex -> Document + Sections

Parsing: main file = the .tex with \\documentclass; \\input/\\include resolved from the archive; comments and the
bibliography removed; abstract environment -> "Abstract" section; split at \\section / \\subsection; LaTeX -> text
with pylatexenc (math -> [math], citations -> [cite]).
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
import time
from pathlib import Path

from data.schema import Document, ParsedDocument, Section, with_char_spans
from data.segment import RawSection, split_oversized

SOURCE = "arxiv_eprint"
USER_AGENT = "negspace-research/0.1 (academic project; github.com)"
EPRINT_URL = "https://export.arxiv.org/e-print/{id}{version}"
_last_request = [0.0]


def fetch_eprint(arxiv_id: str, version: str, cache_dir: str | Path, *, min_interval: float = 3.0,
                 timeout: float = 60.0) -> Path:
    import httpx

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{arxiv_id.replace('/', '_')}{version}"
    if path.exists() and path.stat().st_size > 0:
        return path
    missing = path.with_name(path.name + ".missing")
    if missing.exists():  # arXiv answered 404 before: do not ask again
        raise FileNotFoundError(f"{arxiv_id}{version}: not available on arXiv (cached 404)")
    wait = _last_request[0] + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()
    r = httpx.get(EPRINT_URL.format(id=arxiv_id, version=version), headers={"User-Agent": USER_AGENT},
                  timeout=timeout, follow_redirects=True)
    if r.status_code == 404:
        missing.write_text("404\n")
    r.raise_for_status()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.rename(path)
    return path


# ------------------------------------------------------------------------------------ LaTeX
_COMMENT = re.compile(r"(?<!\\)%.*")
_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
_SECTION = re.compile(r"\\(section|subsection)\*?\s*(?:\[[^\]]*\])?\s*\{")
_ABSTRACT = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.S)
_BIB = re.compile(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}|\\bibliography\{[^}]*\}|\\bibliographystyle\{[^}]*\}", re.S)
_DROP_ENVS = re.compile(r"\\begin\{(figure\*?|table\*?|algorithm\*?|tabular\*?)\}.*?\\end\{\1\}", re.S)


def _read_sources(path: Path) -> dict[str, str]:
    """name -> text for every .tex file in an e-print (tar.gz, or a single gzipped file)."""
    raw = path.read_bytes()
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            out = {}
            for m in tar.getmembers():
                if m.isfile() and m.name.endswith(".tex"):
                    f = tar.extractfile(m)
                    if f is not None:
                        out[m.name] = f.read().decode("utf-8", errors="replace")
            return out
    except tarfile.ReadError:
        data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        return {"main.tex": data.decode("utf-8", errors="replace")}


def _resolve_inputs(text: str, files: dict[str, str], depth: int = 0) -> str:
    if depth > 5:
        return text

    def repl(m: re.Match) -> str:
        name = m.group(1).strip()
        for cand in (name, f"{name}.tex"):
            for key in files:
                if key == cand or key.endswith("/" + cand):
                    return _resolve_inputs(_COMMENT.sub("", files[key]), files, depth + 1)
        return ""

    return _INPUT.sub(repl, text)


def _braced(text: str, start: int) -> tuple[str, int]:
    """Content of the brace group opening just before `start`, and the index after its closing brace."""
    depth, i = 1, start
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[start:i - 1], i


def _to_text(latex: str) -> str:
    from pylatexenc.latex2text import LatexNodes2Text, MacroTextSpec

    l2t = LatexNodes2Text(math_mode="remove", strict_latex_spaces=True)
    l2t.latex_context.add_context_category("negspace", prepend=True, macros=[
        MacroTextSpec("cite", simplify_repl="[cite]"), MacroTextSpec("citep", simplify_repl="[cite]"),
        MacroTextSpec("citet", simplify_repl="[cite]"), MacroTextSpec("ref", simplify_repl="[ref]"),
        MacroTextSpec("cref", simplify_repl="[ref]"), MacroTextSpec("label", simplify_repl=""),
        MacroTextSpec("footnote", simplify_repl=""), MacroTextSpec("url", simplify_repl="%s"),
        # pylatexenc renders these as "[NO \\title GIVEN] ... <today's date>": the build date would leak into the text
        MacroTextSpec("maketitle", simplify_repl=""), MacroTextSpec("today", simplify_repl=""),
    ])
    latex = re.sub(r"\$[^$]+\$|\\\(.*?\\\)", " [math] ", latex, flags=re.S)
    latex = re.sub(r"\\begin\{(equation|align|eqnarray|gather|multline)\*?\}.*?\\end\{\1\*?\}", " [math] ", latex, flags=re.S)
    try:
        text = l2t.latex_to_text(latex)
    except Exception:  # pylatexenc is lenient; anything left is residue we strip below
        text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", latex)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n".join(p for p in paragraphs if p)


_NEWCOMMAND = re.compile(r"\\(?:re)?newcommand\*?\s*\{?\\([A-Za-z]+)\}?\s*\{")
_DEF = re.compile(r"\\def\s*\\([A-Za-z]+)\s*\{")


def _expand_simple_macros(body: str) -> str:
    """Expand zero-argument \\newcommand / \\def macros (e.g. \\newcommand{\\bert}{BERT}); others are left."""
    macros = {}
    for pattern in (_NEWCOMMAND, _DEF):
        for m in pattern.finditer(body):
            value, _ = _braced(body, m.end())
            if "#" not in value and len(value) < 200:
                macros[m.group(1)] = value
    for name in sorted(macros, key=len, reverse=True):
        body = re.sub(r"\\" + name + r"(?![A-Za-z])(\{\})?", lambda _m, v=macros[name]: v, body)
    return body


def latex_sections(files: dict[str, str]) -> tuple[str | None, list[tuple[str | None, str]]]:
    """(title, [(heading, text)]) from an e-print's .tex files."""
    main = next((n for n, t in files.items() if "\\documentclass" in t and "\\begin{document}" in t), None)
    if main is None:
        main = max(files, key=lambda n: len(files[n]))
    body = _expand_simple_macros(_resolve_inputs(_COMMENT.sub("", files[main]), files))
    title_m = re.search(r"\\title\s*(?:\[[^\]]*\])?\s*\{", body)
    title = _to_text(_braced(body, title_m.end())[0]) if title_m else None
    if "\\begin{document}" in body:
        body = body.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
    body = _DROP_ENVS.sub(" ", _BIB.sub(" ", body))
    sections: list[tuple[str | None, str]] = []
    abstract = _ABSTRACT.search(body)
    if abstract:
        sections.append(("Abstract", _to_text(abstract.group(1))))
        body = body[:abstract.start()] + body[abstract.end():]
    marks = []
    for m in _SECTION.finditer(body):
        heading, end = _braced(body, m.end())
        marks.append((m.start(), end, _to_text(heading)))
    for i, (start, end, heading) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(body)
        text = _to_text(body[end:stop])
        if text:
            sections.append((heading or None, text))
    return title, sections


def eprint_to_document(path: str | Path, *, arxiv_id: str, version: str, date: str | None, category: str | None,
                       genre: str = "papers", max_section_chars: int | None = None) -> ParsedDocument:
    title, secs = latex_sections(_read_sources(Path(path)))
    doc_id = f"{SOURCE}:{arxiv_id}{version}"
    raw = split_oversized([RawSection(heading=h, paragraphs=t.split("\n")) for h, t in secs], max_chars=max_section_chars)
    sections = [Section(doc_id=doc_id, idx=0, heading=r.heading, text=r.text) for r in raw]
    doc = Document(id=doc_id, genre=genre, source=SOURCE, entity_id=arxiv_id, version=version, date=date,
                   category=category, title=title, meta={"eprint": Path(path).name})
    return ParsedDocument(doc, with_char_spans(sections))
