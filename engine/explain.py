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
