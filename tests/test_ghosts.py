"""engine/explain.py ghost placement (plan 7.1): where the app inserts a ghost paragraph for an absence, from the
type's section_hint only (generic: no genre vocabulary). A heading named in the hint wins (token overlap, words
common to the document's headings count less); otherwise the hinted part of the document (near the start, first
half, second half, near the end), after its last section."""

from engine.explain import ghost_position, place_ghosts

PAPER = ["Abstract", "1 Introduction", "2 Related Work", "3 Method", "4 Results", "5 Conclusion"]
POLICY = ["Privacy Policy", "Information we collect", "Your privacy choices", "Security measures", "Children", "Contact us"]


def test_a_quoted_heading_wins():
    p = ghost_position("under a heading like 'security', in the second half of the document", POLICY)
    assert p == {"after": 3, "how": "heading", "heading": "Security measures"}


def test_spelling_variants_and_unquoted_section_names_match():
    acks = PAPER + ["Acknowledgments", "References"]
    assert ghost_position("under a heading like 'acknowledgements', near the end of the document", acks)["after"] == 6
    assert ghost_position("in the introduction or related work, in the first half of the document", PAPER)["after"] in (1, 2)
    assert ghost_position("in or next to the cookies and tracking section, in the second half of the document",
                          ["Intro", "Cookie Policy", "Rights"])["after"] == 1


def test_words_common_to_the_headings_count_less():
    hint = "under a heading like 'children s privacy', in the second half of the document"
    assert ghost_position(hint, POLICY)["heading"] == "Children"


def test_without_a_matching_heading_the_hinted_part_is_used():
    six = ["A", "B", "C", "D", "E", "F"]
    assert ghost_position("in the first half of the document", six) == {"after": 2, "how": "position", "heading": None}
    assert ghost_position("in the second half of the document", six)["after"] == 5
    assert ghost_position("near the start of the document", six)["after"] == 0
    assert ghost_position("near the end, under a heading like 'Your California Privacy Rights'", six)["after"] == 5
    assert ghost_position(None, six)["after"] == 5
    assert ghost_position("under a heading like 'security', in the first half", [None, None, None, None])["after"] == 1
    assert ghost_position("in the first half of the document", ["Only"])["after"] == 0


def test_place_ghosts_stacks_absences_at_one_position_in_rank_order():
    absences = [{"type": "a", "section_hint": "in the second half of the document"},
                {"type": "b", "section_hint": "under a heading like 'security'"},
                {"type": "c", "section_hint": "near the end of the document"}]
    placed = place_ghosts(absences, POLICY)
    assert {k: [a["type"] for a in v] for k, v in placed.items()} == {5: ["a", "c"], 3: ["b"]}
    assert placed[3][0]["ghost"]["how"] == "heading"
