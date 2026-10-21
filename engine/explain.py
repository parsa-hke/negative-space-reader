"""Explanations for the reported absences (plan §0.2 #14, #21; §5.8).

explain.mode: template (default, lean): the sentence is filled from the evidence only, "N% of peers like this one
state X; this document does not. Peers typically say: ...", with the quotes verbatim. Never a verdict ("missing",
"non-compliant"), only prevalence among peers. explain.mode: teacher (opt-in, flagged in the UI because it sends the
peer quotes off the machine, never the document): a rationale and a suggested wording for the top-N, written from
the peer quotes only (prompts/explain.md). Both modes fill the same Report fields (EXPLAIN_FIELDS).
"""

from __future__ import annotations

from typing import Any


def label(card: dict[str, Any]) -> str:
    name = card.get("name") or card["id"]
    return name.replace("_", " ")


def template(absence: dict[str, Any], card: dict[str, Any], evidence: list[dict[str, Any]], *, quotes: int = 2) -> str:
    n, count = absence["n"], absence["count"]
    pct = round(100 * count / n) if n else 0
    text = (f"{pct}% of peers like this one ({count:g} of {n}) make a statement of the kind "
            f"“{label(card)}” ({card.get('definition', '').rstrip('.')}); this document does not.")
    said = [e for e in evidence if e["source"] == "cohort"][:quotes]
    if said:
        text += " Peers typically say: " + " ".join(f"“{e['text']}” ({e['doc_id']})" for e in said)
    return text


EXPLAIN_FIELDS = ("explanation", "rationale", "suggested_wording", "explanation_mode")

_STOP = set("""a an the and or but nor of to in on at by for from with as into onto about over under than then that this
these those there here it its they them their we us our you your he she his her i me my is are was were be been being
do does did have has had can could may might will would shall should must not no any all some each such other more
most also only if when where which who whom whose what how why so too very own same just""".split())
# The prompt asks for "what peers typically state" in general terms: these framing words are expected, not content.
_FRAMING = set("""peer peers typically usually often commonly generally state states stated statement statements say
says said comparable similar document documents mention mentions mentioned describe describes explain explains note
notes indicate indicates provide provides include includes""".split())


def unsupported_words(text: str, sources: list[str]) -> list[str]:
    """The content words of text (teacher-written) that no source contains, in order of appearance: the check of
    plan 5.8 that a teacher explanation stays within the quoted peer sentences. Lenient by design (it flags, it does
    not judge): words are compared on their first five letters (so "deletes" matches "deleted"), numbers exactly;
    [placeholders], stop words and the prompt's framing words are ignored."""
    import re

    def words(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", re.sub(r"\[[^\]]*\]", " ", s.lower()))

    key = lambda w: w if w.isdigit() else w[:5]  # noqa: E731
    have = {key(w) for s in sources for w in words(s)}
    out: list[str] = []
    for w in words(text):
        if w in _STOP or w in _FRAMING or (len(w) < 3 and not w.isdigit()) or key(w) in have or w in out:
            continue
        out.append(w)
    return out


def render_explain_prompt(absence: dict[str, Any], card: dict[str, Any], genre: dict[str, Any], quotes: list[dict[str, Any]]) -> str:
    from models.teacher import PROMPTS

    n, count = absence["n"], absence["count"]
    return PROMPTS.get("explain").render(
        document_kind=genre["document_kind"], author_role=genre["author_role"], n=n, count=f"{count:g}",
        pct=round(100 * count / n) if n else 0, type_label=label(card), definition=card.get("definition", ""),
        quotes="\n".join(f"- {q['text']} ({q['doc_id']})" for q in quotes))


def explain_absences(absences: list[dict[str, Any]], cards: dict[str, dict[str, Any]], *, mode: str = "template",
                     teacher: Any = None, genre: dict[str, Any] | None = None, top_n: int = 5,
                     max_tokens: int = 256) -> list[dict[str, Any]]:
    """Fill EXPLAIN_FIELDS on every absence. template: the evidence sentence only. teacher (opt-in; sends the peer
    quotes, not the document, to the teacher): adds a rationale and a suggested wording for the top_n absences that
    have cohort quotes, written from those quotes only (prompts/explain.md)."""
    from pydantic import BaseModel

    for a in absences:
        a.update(explanation=template(a, cards[a["type"]], a.get("evidence", [])), rationale=None,
                 suggested_wording=None, explanation_mode="template")
    if mode != "teacher":
        return absences
    if teacher is None or genre is None:
        raise ValueError("explain.mode teacher needs a teacher and the genre's document_kind / author_role")

    class Explanation(BaseModel):
        rationale: str
        suggested_wording: str

    todo = [(a, [e for e in a.get("evidence", []) if e["source"] == "cohort"]) for a in absences[:top_n]]
    todo = [(a, q) for a, q in todo if q]
    if todo:
        answers = teacher.generate_structured_batch([render_explain_prompt(a, cards[a["type"]], genre, q) for a, q in todo],
                                                    Explanation, max_tokens=max_tokens)
        for (a, _), ans in zip(todo, answers):
            a.update(rationale=ans.rationale, suggested_wording=ans.suggested_wording, explanation_mode="teacher")
    return absences


# ---------------------------------------------------------------- ghost placement (plan 7.1)
# Words of a section_hint that say where, not what: the rest names the heading the type usually sits under.
_WHERE = set("""under heading headings like near start beginning end first second half part document section sections
next close before after towards toward""".split())


def _tokens(s: str | None) -> list[str]:
    import re

    return [w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) >= 3 and w not in _STOP]


def ghost_position(hint: str | None, headings: list[str | None]) -> dict[str, Any]:
    """Where a ghost paragraph for an absence goes, from its type's section_hint (schema text, any genre): after the
    section whose heading best matches the words the hint names (compared on their first five letters, each weighted
    by how few of the document's headings contain it, so a word in every heading counts little); without a match,
    after the last section of the hinted part (near the start: the first section; first half; second half or near the
    end: the last section; no hint: the last section). Returns {after: section index, how: heading | position,
    heading: the matched heading or None}."""
    import math

    n = len(headings)
    if n == 0:
        return {"after": 0, "how": "position", "heading": None}
    key = lambda w: w[:5]  # noqa: E731
    wanted = {key(w) for w in _tokens(hint) if w not in _WHERE}
    head_keys = [{key(w) for w in _tokens(h)} for h in headings]
    df: dict[str, int] = {}
    for ks in head_keys:
        for w in ks:
            df[w] = df.get(w, 0) + 1
    best, best_w = None, 0.0
    for i, ks in enumerate(head_keys):
        w = sum(math.log(1 + n / df[x]) for x in ks & wanted)
        if w > best_w:
            best, best_w = i, w
    if best is not None:
        return {"after": best, "how": "heading", "heading": headings[best]}
    h = (hint or "").lower()
    if "start" in h or "beginning" in h:
        after = 0
    elif "first half" in h:
        after = max(0, math.ceil(n / 2) - 1)
    else:  # second half, near the end, or no hint
        after = n - 1
    return {"after": after, "how": "position", "heading": None}


def place_ghosts(absences: list[dict[str, Any]], headings: list[str | None]) -> dict[int, list[dict[str, Any]]]:
    """Section index -> the absences whose ghost goes after it, in rank order; each absence gets its `ghost` position."""
    out: dict[int, list[dict[str, Any]]] = {}
    for a in absences:
        a["ghost"] = ghost_position(a.get("section_hint"), headings)
        out.setdefault(a["ghost"]["after"], []).append(a)
    return out
