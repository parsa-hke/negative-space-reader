"""data/segment.py: section and sentence segmentation, boilerplate removal (plan §2.5).

Five documents per genre. They are synthetic but reproduce, case by case, the formats found when inspecting the
real corpora (CHANGELOG 2026-10-12): Markdown policies with ATX headings and a table of contents, bold-line
headings with escaped numbering, ALL-CAPS headings, policies with no headings, mirror preambles, navigation,
cookie banners and copyright footers; unarXive papers with formula/citation placeholders, dict abstracts,
listings, repeated section names and versions past the corpus cutoff.
"""

import pytest

from data.loaders.unarxive import paper_to_document
from data.segment import RawSection, clean_inline, normalise_sections, segment_markdown, split_oversized, split_sentences


def headings(text, **kw):
    return [s.heading for s in normalise_sections(segment_markdown(text), **kw)]


# ------------------------------------------------------------------------------------ genre A: policies
POLICY_ATX_TOC = """> *The following text is extracted and transformed from the example.com privacy policy that was archived on 2019-06-30.*

# Privacy policy – Help

_Last Updated on May 24, 2018_

Example Corp ("we") cares about your privacy. This policy describes how we collect and use information.

1\\. What Personal Information Do We Gather?

2\\. How Do We Use It?

**1\\. What Personal Information Do We Gather?**

We collect your name, email address and payment details when you subscribe to our services.

**2\\. How Do We Use It?**

We use your information to provide the service, for billing, and to send you newsletters you asked for.
"""

POLICY_BOLD = """**Information We Collect**

We collect information you give us directly, such as your name and [email address](https://web.archive.org/x).

  * Account details you provide.
  * Messages you send to support.

**Sharing**

We share data with service providers that process it on our behalf under contract.
"""

POLICY_CAPS = """PRIVACY NOTICE

This notice explains our practices for visitors of this website and customers of our shops.

DATA RETENTION

We keep order records for seven years to comply with tax law, then delete them securely.

YOUR RIGHTS

You may ask for a copy of your data or for its deletion by writing to our privacy office.
"""

POLICY_NO_HEADINGS = "\n\n".join(
    f"Paragraph {i} explains one more practice of the company in plain prose without any heading at all." for i in range(40))

POLICY_BOILERPLATE = """Skip to main content
Home | About | Contact
This website uses cookies to improve your experience. Click OK to accept.

## Cookies

We use cookies to remember your preferences and to measure traffic. You can disable cookies in your browser.

## Contact

Write to privacy@example.com with any question about this policy and we will answer within 30 days.

© 2019 Example Corp. All rights reserved.
"""


def test_policy_atx_headings_and_toc_dropped():
    secs = normalise_sections(segment_markdown(POLICY_ATX_TOC), min_chars=40)
    assert [s.heading for s in secs] == ["Privacy policy – Help", "1. What Personal Information Do We Gather?",
                                         "2. How Do We Use It?"]
    assert "Last Updated on May 24, 2018" in secs[0].text
    assert not any("extracted and transformed" in s.text for s in secs)  # mirror preamble removed
    assert all("1. What" not in s.text for s in secs)  # table of contents is not body text


def test_policy_bold_headings_lists_and_links():
    secs = normalise_sections(segment_markdown(POLICY_BOLD), min_chars=40)
    assert [s.heading for s in secs] == ["Information We Collect", "Sharing"]
    assert "email address" in secs[0].text and "web.archive.org" not in secs[0].text
    assert "Account details you provide." in secs[0].text.split("\n")  # list marker stripped


def test_policy_all_caps_headings():
    assert headings(POLICY_CAPS, min_chars=40) == ["PRIVACY NOTICE", "DATA RETENTION", "YOUR RIGHTS"]


def test_policy_without_headings_is_chunked():
    secs = normalise_sections(segment_markdown(POLICY_NO_HEADINGS, fallback_chars=600))
    assert len(secs) >= 4 and all(s.heading is None for s in secs)
    assert all(len(s.text) <= 700 for s in secs)
    assert sum(s.text.count("Paragraph") for s in secs) == 40  # nothing lost


def test_policy_boilerplate_removed_but_cookie_content_kept():
    secs = normalise_sections(segment_markdown(POLICY_BOILERPLATE), min_chars=40)
    text = "\n".join(s.text for s in secs)
    assert [s.heading for s in secs] == ["Cookies", "Contact"]
    assert "Skip to" not in text and "Home | About" not in text and "Click OK" not in text and "©" not in text
    assert "We use cookies to remember your preferences" in text


# ------------------------------------------------------------------------------------ genre B: papers
def paper(body, *, abstract="We study X.", versions=(("v1", "Mon, 2 Apr 2018 19:18:42 GMT"),)):
    return {"paper_id": "1804.00001", "discipline": "Computer Science", "abstract": abstract,
            "metadata": {"categories": "cs.CL cs.LG", "title": "A Paper", "license": "http://creativecommons.org/licenses/by/4.0/",
                         "versions": [{"version": v, "created": c} for v, c in versions]},
            "body_text": body}


def para(section, text, content_type="paragraph"):
    return {"section": section, "sec_number": "1", "sec_type": "section", "content_type": content_type, "text": text}


def test_paper_placeholders_cleaned():
    pd = paper_to_document(paper([para("Method", "We minimise {{formula:ab-12}} as in {{cite:9f}} (see {{ref:fig1}}). "
                                                 "Figure REF and Eq. (REF ) agree with TableREF .")]))
    assert pd.sections[1].text == ("We minimise [math] as in [cite] (see [ref]). Figure [ref] and Eq. ([ref]) agree "
                                   "with Table [ref].")


def test_paper_dict_abstract_becomes_first_section():
    pd = paper_to_document(paper([para("Intro", "Text.")], abstract={"section": "Abstract", "text": "We show Y."}))
    assert (pd.sections[0].heading, pd.sections[0].text) == ("Abstract", "We show Y.")


def test_paper_keeps_listings_drops_other_content():
    pd = paper_to_document(paper([para("Code", "for x in y:", "listing"), para("Code", "| a | b |", "table")]))
    assert pd.sections[1].text == "for x in y:"


def test_paper_paragraphs_grouped_by_section():
    pd = paper_to_document(paper([para("Intro", "A."), para("Intro", "B."), para("Method", "C."), para("Intro", "D.")]))
    assert [(s.heading, s.text) for s in pd.sections[1:]] == [("Intro", "A.\nB."), ("Method", "C."), ("Intro", "D.")]
    assert [s.idx for s in pd.sections] == [0, 1, 2, 3]
    start, end = pd.sections[2].char_span
    assert pd.text()[start:end] == "C."


def test_paper_version_after_cutoff_ignored():
    pd = paper_to_document(paper([para("A", "x")], versions=(("v1", "Mon, 2 Apr 2018 19:18:42 GMT"),
                                                             ("v2", "Tue, 3 Jan 2023 10:00:00 GMT"))))
    assert (pd.document.version, pd.document.date) == ("v1", "2018-04-02")
    assert pd.document.category == "cs.CL" and pd.document.entity_id == "1804.00001"


# ------------------------------------------------------------------------------------ shared
@pytest.mark.parametrize("text,expected", [
    ("We collect e.g. your name. We keep it.", ["We collect e.g. your name.", "We keep it."]),
    ("See Sec. 3.2 of the U.S. policy for details. Fig. 2 shows it.", ["See Sec. 3.2 of the U.S. policy for details.", "Fig. 2 shows it."]),
    ("Prior work (Smith et al., 2019) reports gains. We agree.", ["Prior work (Smith et al., 2019) reports gains.", "We agree."]),
    ("J. Smith wrote the policy for the company. It changed.", ["J. Smith wrote the policy for the company.", "It changed."]),
    ("Our address is below. 1. Done", ["Our address is below. 1. Done"]),
    ("First line without a stop\nSecond line ends here.", ["First line without a stop", "Second line ends here."]),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


def test_oversized_section_is_split_without_losing_text():
    long = "\n".join(f"Sentence number {i} is part of a very long section about data sharing." for i in range(300))
    secs = normalise_sections(segment_markdown(f"## Sharing\n\n{long}"), max_chars=2000)
    assert len(secs) > 1 and all(len(s.text) <= 2000 for s in secs) and all(s.heading == "Sharing" for s in secs)
    assert sum(s.text.count("Sentence number") for s in secs) == 300



def test_split_oversized_only_splits_long_sections():
    """The split step on its own (used by every loader): short sections, including tiny ones, are left as they are;
    a long one is split at paragraph breaks, and a paragraph longer than the limit at sentence boundaries."""
    short = RawSection(heading="Tiny", paragraphs=["Short."])
    long = RawSection(heading="Results", paragraphs=[f"Paragraph {i} reports one more result in detail." for i in range(60)]
                      + [" ".join(f"Sentence {j} of a single giant paragraph." for j in range(80))])
    out = split_oversized([short, long], max_chars=500)
    assert out[0] == short
    rest = out[1:]
    assert len(rest) > 2 and all(len(s.text) <= 500 for s in rest) and all(s.heading == "Results" for s in rest)
    assert " ".join(" ".join(s.paragraphs) for s in rest).split() == " ".join(long.paragraphs).split()
    assert split_oversized([long], max_chars=None) == [long]

def test_clean_inline():
    assert clean_inline("**_Contact Us_**") == "Contact Us"
    assert clean_inline("keep snake_case and x * y") == "keep snake_case and x * y"
    assert clean_inline("![logo](a.png) [here](http://x.org/a_(b)) 1\\. done") == "here 1. done"
