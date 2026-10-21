"""document_markdown + segment reproduce a corpus document's stored sections (CHANGELOG 2026-10-20).

analyse() segments the Markdown it is given, while the indexed peers' profiles were built on the corpus's stored
sections; a corpus document rebuilt as Markdown (gate 5.9, the `analyse` task, demo examples) must come back with
the same sections, or the query is segmented differently from its peers. Fixtures reproduce the three failures
found on test documents: a headingless policy chunked by fallback_section_chars, a paper whose title spans two
lines, and paper sections whose lines look like headings (plus blank lines, short sections, '#'/'\\' line starts).
Text without the section marker still goes through the heuristic segmenter unchanged."""

from data.schema import Document, Section
from data.segment import SECTIONS_MARKER, normalise_sections, read_sections_markdown, segment_markdown
from engine.pipeline import document_markdown, guess_title, segment

SCFG = {"min_section_chars": 80, "max_section_chars": 8000, "fallback_section_chars": 1500}

PARA = ("We collect the information you give us when you register, place an order or write to us, and we use it "
        "only to provide the service you asked for.")


def _doc(doc_id, title, sections):
    d = Document(id=doc_id, genre="g", source="s", entity_id=doc_id, title=title)
    return d, [Section(doc_id=doc_id, idx=i, heading=h, text=t) for i, (h, t) in enumerate(sections)]


def _corpus_policy(raw):
    """Sections as the privacy loader stores them (data/loaders/privacy_pl.to_parsed)."""
    return [(s.heading, s.text) for s in normalise_sections(segment_markdown(raw, fallback_chars=1500),
                                                            min_chars=80, max_chars=8000)]


def _assert_round_trip(doc, secs):
    got = segment(document_markdown(doc, secs), SCFG)
    assert [t for _, t in got] == [s.text for s in secs]
    assert [h for h, _ in got] == [s.heading for s in secs]


def test_headingless_policy_keeps_its_fallback_chunks():
    raw = "\n\n".join(f"{PARA} Paragraph {i} adds a little more about how long we keep it." for i in range(30))
    stored = _corpus_policy(raw)
    assert len(stored) > 1 and all(h is None for h, _ in stored)
    doc, secs = _doc("privacy_pl:1", "Privacy Policy", stored)
    _assert_round_trip(doc, secs)


def test_headed_policy_with_lists_pipes_and_heading_like_lines():
    secs = [(None, f"| Privacy Policy\nHome PageBack\n{PARA}"),
            ("Information We Collect", f"{PARA}\nCookies\n{PARA}\n# not a heading\n\\ a backslash line"),
            ("How Can I Opt-Out?", f"{PARA}  with  double spaces |")]
    doc, secs = _doc("privacy_pl:2", "Plantronics - Privacy Policy", secs)
    _assert_round_trip(doc, secs)


def test_paper_with_abstract_multiline_title_and_inner_pseudo_headings():
    secs = [("Abstract", f"{PARA} " * 3),
            ("Introduction", f"{PARA}\n\n{PARA}"),                      # a blank line inside a section
            ("Training algorithms", f"{PARA}\nParameterparameters\n{PARA}\nDefinition 2 (Sectional curvature [cite])\n{PARA}"),
            ("Training algorithms", "Short tail of a split section."),  # under min_section_chars
            (None, f"Theorem 3.1 (Equidistribution in [math] [cite], [cite])\n{PARA}"),
            ("METEOR", f"## {PARA}\n<!-- section -->\n{PARA}")]
    doc, secs = _doc("unarxive:1801.00632", "Character-level Recurrent Neural Networks in Practice: Comparing\n"
                     "Training and Sampling Schemes", secs)
    _assert_round_trip(doc, secs)
    md = document_markdown(doc, secs)
    assert guess_title(md) == "Character-level Recurrent Neural Networks in Practice: Comparing Training and Sampling Schemes"


def test_document_without_title():
    doc, secs = _doc("x:1", None, [("Abstract", PARA), ("Method", PARA)])
    _assert_round_trip(doc, secs)
    doc, secs = _doc("x:2", None, [(None, PARA), (None, PARA)])
    _assert_round_trip(doc, secs)
    assert guess_title(document_markdown(doc, secs)) == PARA


def test_marked_markdown_edited_by_hand_still_parses():
    """A user who edits an exported document keeps sensible sections: a new heading starts a section, text before
    the first heading is a headingless section, and an oversized section is still split at max_section_chars."""
    md = f"{SECTIONS_MARKER}\n# A title\n\nIntro text {PARA}\n\n### Added heading\n{PARA}\n\n## Long\n" + "\n".join([PARA] * 120)
    title, raw = read_sections_markdown(md)
    assert title == "A title" and [r.heading for r in raw] == [None, "Added heading", "Long"]
    got = segment(md, SCFG)
    assert [h for h, _ in got[:2]] == [None, "Added heading"] and len(got) > 3
    assert all(len(t) <= 8000 for _, t in got)


def test_unmarked_text_uses_the_heuristic_segmenter():
    raw = f"# Privacy\n\n## Sharing\n{PARA}\n\nCookies\n{PARA}"
    assert read_sections_markdown(raw) is None
    assert segment(raw, SCFG) == [(r.heading, r.text) for r in normalise_sections(segment_markdown(raw), min_chars=80)]
