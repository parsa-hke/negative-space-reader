"""Explanations for the reported absences (plan §0.2 #14, #21; §5.8).

explain.mode: template (default, lean): the sentence is filled from the evidence only, "N% of peers like this one
state X; this document does not. Peers typically say: ...", with the quotes verbatim. Never a verdict ("missing",
"non-compliant"), only prevalence among peers. (explain.mode: teacher, a teacher-written rationale, is task 5.8.)
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
