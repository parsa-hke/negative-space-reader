"""Gradio class demo: ghost paragraphs, cohort panel, evidence panel, chat-model comparison (plan §7.1). Target of
`run.py serve`.

    python -m experiments.run --config configs/serve.yaml          # http://127.0.0.1:7860

Layout: a hero header; an input card (paste, upload .txt / .md / .html, pack or "auto", the pinned examples of
app/examples.py, the "compare with a chat model" switch); the document as a reading pane with a dashed ghost
paragraph after the section where each top absence's type usually sits (engine/explain.py ghost_position; a click
on a ghost opens its evidence); on the right, Cohort (k, category filter, date window, re-rank without embedding
again: engine/pipeline.py prepare / score) and Evidence (the selected absence: peer count and Wilson interval, up to
five peer sentences with document ids, the template explanation); below, when the switch is on, the comparison panel:
this system's ranking next to the lists the 7.3 runs recorded for a benchmark example (Claude Opus 5.5, the open
teacher), with the types the benchmark deleted marked. Cached answers only: a live call is behind
serve.allow_live_chat and not implemented in the lean tier.

Hosting: `serve.zerogpu: true` (configs/space_zerogpu.yaml, a Pro account's ZeroGPU Space) wraps the analyse handler
in `spaces.GPU`, so models load to CUDA and an H200 slice is attached per analysis; otherwise everything runs where
`run.device` says (the CPU bundle, the Mac).

Look and motion: app/static/nsr.css and nsr.js (cream page, teal / light teal / orange, a lens cursor with a particle trail, entrance
and breathing animations; all off under prefers-reduced-motion). Plan §0.2 #14, #21: the wording is prevalence among
peers, never a verdict; uploads are held in memory for the session only (an uploaded file is deleted as soon as it is
read; serve.store_uploads must stay false). Teacher rationales send the peer quotes (not the document) off the
machine: hidden unless serve.allow_teacher is true, then off by default and flagged.
"""

from __future__ import annotations

import html
import os
import statistics
import time
from pathlib import Path
from typing import Any

STATIC = Path(__file__).parent / "static"
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;'
         '0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">')
FORBIDDEN = ("non-compliant", "noncompliant", "deficient", "missing", "violation")  # plan §0.2 #21: never a verdict
# a reload with ?__theme=light, once, so Gradio's own controls render light whatever the system setting
FORCE_THEME_JS = """() => { const u = new URL(window.location); if (u.searchParams.get('__theme') !== 'light') {
  u.searchParams.set('__theme', 'light'); window.location.replace(u.toString()); } }"""


def css() -> str:
    return (STATIC / "nsr.css").read_text(encoding="utf-8")


def head() -> str:
    return FONTS + "<script>" + (STATIC / "nsr.js").read_text(encoding="utf-8") + "</script>"


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _label(t: str) -> str:
    return t.replace("_", " ")


def _pct(a: dict[str, Any]) -> int:
    return round(100 * a["count"] / a["n"]) if a["n"] else 0


def _paragraphs(text: str) -> str:
    parts = [p.strip() for p in text.replace("\r", "").split("\n\n") if p.strip()]
    return "".join(f"<p>{_esc(p).replace(chr(10), '<br>')}</p>" for p in parts)


# ---------------------------------------------------------------- rendering
def ghost_html(rank: int, a: dict[str, Any]) -> str:
    g = a.get("ghost") or {}
    where = (f"placed after the section headed “{_esc(g['heading'])}”" if g.get("how") == "heading"
             else f"placed by position ({_esc(a.get('section_hint') or 'no hint')})")
    sev = f"<span class='nsr-badge sev'>severity ×{a['weight']:g}</span>" if a.get("weight", 1.0) != 1.0 else ""
    quote = next((e for e in a.get("evidence", []) if e["source"] == "cohort"), None)
    said = f"<div class='nsr-said'>“{_esc(quote['text'])}” <span>{_esc(quote['doc_id'])}</span></div>" if quote else ""
    p = _pct(a)
    return (f"<div class='nsr-ghost' data-type='{_esc(a['type'])}' style='--i:{rank - 1}'>"
            f"<div class='nsr-head'><span class='nsr-rank'>#{rank}</span>"
            f"<span class='nsr-pct' data-count='{p}' data-suffix='%'>{p}%</span>"
            f"<span class='nsr-claim'>of peers like this one state <b>“{_esc(_label(a['type']))}”</b>; this document does not.{sev}</span></div>"
            f"<div class='nsr-bar'><i style='--p:{p / 100:.3f}'></i></div>"
            f"<div class='nsr-meta'>{a['count']:g} of {a['n']} peers · 95% interval {a['ci'][0]:.2f}–{a['ci'][1]:.2f} · "
            f"{_esc(a.get('definition'))} · {where}</div>{said}</div>")


def document_html(sections: list[tuple[str | None, str]], absences: list[dict[str, Any]], title: str | None = None,
                  found: set[str] | None = None) -> str:
    """The document, section by section, with the ghost paragraphs of the ranked absences after their sections; the
    ghosts of `found` types (the ones a benchmark removed, once the comparison is revealed) carry a badge."""
    from engine.explain import place_ghosts

    placed = place_ghosts(absences, [h for h, _ in sections])
    rank = {a["type"]: i + 1 for i, a in enumerate(absences)}
    out = ["<div class='nsr-doc'>"]
    if title:
        out.append(f"<h2 class='nsr-title'>{_esc(title)}</h2>")
    for i, (heading, text) in enumerate(sections):
        if heading:
            out.append(f"<h3>{_esc(heading)}</h3>")
        out.append(_paragraphs(text))
        for a in placed.get(i, []):
            h = ghost_html(rank[a["type"]], a)
            if found and a["type"] in found:
                h = h.replace("this document does not.", "this document does not.<span class='nsr-found'>removed by the benchmark · found</span>", 1)
            out.append(h)
    out.append("</div>")
    return "".join(out)


def guide_html() -> str:
    steps = [("1 · Pick", "Paste a policy or a paper, upload a file, or choose an example. “Benchmark” examples are real documents with some statements secretly removed."),
             ("2 · Analyse", "The document is split, its statements detected, and its 200 closest peers found. A second or two."),
             ("3 · Read the ghosts", "Dashed paragraphs appear where peers usually say something this document does not, with how many of them do."),
             ("4 · Check the evidence", "Click a ghost: the peers' own sentences, with document ids, and the exact count behind the number."),
             ("5 · Change the peers, or compare", "Re-rank with a different k, category or date window, or see what two chat models answered for the same benchmark document.")]
    return "<div class='nsr-guide'>" + "".join(f"<div style='--i:{i}'><b>{_esc(t)}</b><span>{d}</span></div>" for i, (t, d) in enumerate(steps)) + "</div>"


def legend_html() -> str:
    return ("<div class='nsr-legend'><span><i style='background:var(--nsr-lime)'></i>absent kinds of statement</span>"
            "<span><i style='background:var(--nsr-cyan)'></i>peers' evidence</span>"
            "<span><i style='background:var(--nsr-magenta)'></i>severity and benchmark reveals</span></div>")


def empty_html() -> str:
    sample = {"type": "states_how_long_data_is_kept", "count": 140, "n": 200, "ci": [0.63, 0.76], "weight": 1.0, "section_hint": "under a heading like 'retention'",
              "definition": "States how long personal data is kept, or the rule that decides it.", "ghost": {"how": "position", "heading": None},
              "evidence": [{"source": "cohort", "text": "We retain your account information for as long as your account is active and for 24 months after.", "doc_id": "example"}]}
    return ("<div class='nsr-empty'><h3>Your document will appear here.</h3><p>Between its sections you will see <b>ghost paragraphs</b>: one for "
            "each kind of statement that this document's closest peers usually make and it does not, ranked by how many of them make it. "
            "Nothing here is a verdict; every number is a count of peers, and every quote is theirs.</p>" + ghost_html(1, sample) + "</div>")


def narrate(rep: dict[str, Any], meta: dict[str, Any] | None, *, detection: dict[str, Any] | None, compare: bool, note: str = "") -> str:
    """The status line as a short narration of what was done, in numbers the reader can check on the panels."""
    c, s, t = rep["cohort"], rep["cohort"]["stats"], rep["timings"]
    kind = {"privacy": "privacy policies", "papers": "research papers"}.get(rep["genre"], rep["genre"])
    parts = [note] if note else []
    if detection is not None:
        parts.append(f"Auto-detect chose the <b>{_esc(rep['genre'])}</b> pack (nearest-neighbour similarity {max(detection['scores'].values()):.2f}).")
    indexed = rep["pack"].get("indexed_documents")
    among = f"<b>{indexed:,}</b> {kind}" if isinstance(indexed, int) else f"the indexed {kind}"
    same_cat = f" of the same category ({_esc(c['filters'].get('category'))})" if c["filters"].get("category") else ""
    parts.append(f"Read <b>{rep['sections']}</b> sections and found this document's <b>{c['n']}</b> closest peers among {among}{same_cat}, "
                 f"mean similarity {s.get('mean_similarity') or 0:.2f}, in {t.get('total', 0):.2f} s.")
    parts.append(f"It states <b>{len(rep['present'])}</b> of the catalogue's <b>{rep['pack'].get('types', '?')}</b> kinds of statement; below are the "
                 f"<b>{len(rep['absences'])}</b> kinds its peers state most often and it does not, each placed where peers usually say it.")
    if c["fallback"]:
        parts.append("The category filter left fewer than k peers, so it was dropped.")
    meta = meta or {}
    if meta.get("kind") == "benchmark":
        n = len(meta.get("removed") or [])
        if compare:
            parts.append(f"<span class='nsr-challenge'><b>Reveal.</b> The benchmark removed every sentence making <b>{n}</b> kind{'s' if n != 1 else ''} of statement "
                         "from this document; the ghosts it found are badged, and the panel at the bottom sets our ranking beside two chat models'.</span>")
        else:
            parts.append(f"<span class='nsr-challenge'><b>Challenge.</b> This is a benchmark copy of a real document: we removed every sentence making "
                         f"<b>{n}</b> kind{'s' if n != 1 else ''} of statement. Did the ghosts catch them? Turn on <i>Compare with a chat model</i> to reveal "
                         "the answer next to what Claude Opus 5.5 and Qwen2.5-72B said.</span>")
    return "<div class='nsr-status'>" + " ".join(parts) + "</div>"


def evidence_html(a: dict[str, Any] | None) -> str:
    if a is None:
        return "<div class='nsr-panel'><p class='nsr-note'>Analyse a document, then pick one of its ranked absences, or click a ghost paragraph.</p></div>"
    p = _pct(a)
    weight = f"<span class='nsr-badge sev'>severity ×{a['weight']:g}</span>" if a.get("weight", 1.0) != 1.0 else ""
    out = [f"<div class='nsr-panel'><h3>{_esc(_label(a['type']))}{weight}</h3><p class='nsr-def'>{_esc(a.get('definition'))}</p>",
           f"<div class='nsr-meter'><div class='nsr-ring' style='--p:{p / 100:.3f}'><b data-count='{p}' data-suffix='%'>{p}%</b></div>"
           f"<p><b>{a['count']:g} of {a['n']} peers</b> state it (95% Wilson interval {a['ci'][0]:.2f}–{a['ci'][1]:.2f}). "
           f"This document: P(present) {a['p_present']:.2f}; omission score {a['score']:.3f}.</p></div>"]
    quotes = a.get("evidence", [])
    out.append(f"<p class='nsr-note'>{'What peers say:' if quotes else 'No peer sentence could be quoted for this type.'}</p>")
    for i, e in enumerate(quotes):
        src = "" if e["source"] == "cohort" else " <span class='nsr-badge proto'>prototype from the labelled set, not this cohort</span>"
        out.append(f"<div class='nsr-quote' style='--i:{i}'>{_esc(e['text'])}<span class='nsr-src'>{_esc(e['doc_id'])}{src}</span></div>")
    out.append(f"<div class='nsr-explain'>{_esc(a.get('explanation'))}</div>")
    if a.get("explanation_mode") == "teacher":
        out.append(f"<div class='nsr-explain'><span class='nsr-badge proto'>teacher-written, from the quotes above</span><br>{_esc(a.get('rationale'))}"
                   f"<br><br>Suggested wording: {_esc(a.get('suggested_wording'))}</div>")
    out.append("</div>")
    return "".join(out)


def cohort_html(report: dict[str, Any], detection: dict[str, Any] | None) -> str:
    c, s, f = report["cohort"], report["cohort"]["stats"], report["cohort"]["filters"]
    t = report["timings"]
    kv = [("peers", f"{c['n']}<small>/{c['k']}</small>", "lime"), ("entities", s.get("entities", "?"), ""),
          ("mean similarity", f"{s.get('mean_similarity') or 0:.3f}", "cyan"), ("seconds", f"{t.get('total', 0):.2f}", "")]
    out = ["<div class='nsr-panel'><div class='nsr-kv'>" + "".join(f"<div><span>{k}</span><b class='{cls}'>{v}</b></div>" for k, v, cls in kv) + "</div>"]
    out.append(f"<p class='nsr-note'><b>Pack</b> {_esc(report['genre'])} · {report['pack'].get('indexed_documents', '?')} indexed documents · "
               f"{report['pack'].get('types', '?')} types · {report['sections']} sections · {len(report['present'])} types stated · "
               f"embed {t.get('embed', 0):.2f} s, cohort {t.get('cohort', 0):.2f} s</p>")
    if detection is not None:
        sc = ", ".join(f"{_esc(k)} {v:.3f}" for k, v in sorted(detection["scores"].items(), key=lambda kv_: -kv_[1]))
        margin = f"; margin {detection['margin']:.3f}" if detection.get("margin") is not None else ""
        out.append(f"<p class='nsr-note'><b>Auto-detect</b> mean similarity of the {detection['neighbours']} nearest documents: {sc}{margin}</p>")
    dw = f.get("date_window") or (None, None)
    out.append(f"<p class='nsr-note'><b>Filters</b> category {_esc(f.get('category') or 'none')} · peers dated {_esc(dw[0] or 'any')} to {_esc(dw[1] or 'any')}</p>")
    if c["fallback"]:
        out.append("<p class='nsr-note warn'>The category filter left fewer than k peers, so it was dropped (fallback).</p>")
    if c["n"] < c["k"]:
        out.append(f"<p class='nsr-note warn'>Only {c['n']} peers match these filters: prevalences rest on a small cohort (see the intervals).</p>")
    out.append("<table class='nsr-peers'><tr><th>nearest peers</th><th>similarity</th><th>category</th><th>date</th></tr>")
    for n in c["nearest"]:
        out.append(f"<tr><td>{_esc(n['doc_id'])}</td><td class='sim'>{n['similarity']:.3f}<i style='--s:{n['similarity']:.3f}'></i></td>"
                   f"<td>{_esc(n.get('category') or '')}</td><td>{_esc(n.get('date') or '')}</td></tr>")
    out.append("</table></div>")
    return "".join(out)


def present_html(report: dict[str, Any]) -> str:
    if not report["present"]:
        return "<p class='nsr-note'>No type of the schema is stated with P ≥ 0.5.</p>"
    return "<div class='nsr-chips'>" + "".join(
        f"<span class='nsr-chip' style='--i:{i}'>{_esc(_label(p['type']))}<span>{p['p_present']:.2f}</span></span>"
        for i, p in enumerate(report["present"])) + "</div>"


def compare_html(report: dict[str, Any] | None, meta: dict[str, Any] | None, *, allow_live: bool = False) -> str:
    """This system's ranking beside the chat models' cached lists for a benchmark example; the deleted types marked."""
    if report is None:
        return "<p class='nsr-note'>Analyse a benchmark example to compare.</p>"
    meta = meta or {}
    if meta.get("kind") != "benchmark" or not meta.get("chat"):
        why = ("A live call would send the document to a chat-model API; it is off on this host (serve.allow_live_chat)."
               if not allow_live else "A live call is not implemented in the lean tier.")
        return (f"<p class='nsr-note'>Cached chat-model answers exist only for the <b>benchmark examples</b> (OmitBench-synthetic "
                f"instances the 7.3 runs answered). {why}</p>")
    removed = list(meta.get("removed") or [])
    hit = set(removed)
    cols = [("Negative-Space Reader", "level 1 · cohort prevalence", [a["type"] for a in report["absences"]], True)]
    cols += [(label, "candidate list · cached 7.3 answer", d.get("ranked") or [], False) for label, d in meta["chat"].items()]
    out = [f"<div class='nsr-banner'><b>Benchmark instance {_esc(meta.get('instance_id'))}.</b> The benchmark deleted every sentence that "
           f"stated {len(removed)} type{'s' if len(removed) != 1 else ''} from this document; each list below is what a system ranked as "
           f"absent. Marked: a deleted type found.<br>" + "".join(f"<span class='nsr-removed'>{_esc(_label(t))}</span>" for t in removed) + "</div>",
           "<div class='nsr-compare'>"]
    for i, (name, sub, ranked, ours) in enumerate(cols):
        first = next((k + 1 for k, t in enumerate(ranked) if t in hit), None)
        found = sum(t in hit for t in ranked[:10])
        items = "".join(f"<li class='{'hit' if t in hit else ''}'>{_esc(_label(t))}</li>" for t in ranked[:10]) or "<li>no answer</li>"
        foot = (f"first deleted type at rank {first}; {found} of {len(removed)} in the top 10" if first else f"none of the {len(removed)} deleted types in the top 10")
        out.append(f"<div class='nsr-col{' ours' if ours else ''}' style='--i:{i}'><h4>{_esc(name)}</h4><div class='nsr-sub'>{sub}</div><ol>{items}</ol><div class='nsr-foot'>{foot}</div></div>")
    out.append("</div><p class='nsr-note'>Ranks are per instance; the recorded comparison over 100 instances per genre is in reports/tables/llm_gap_candidate_&lt;genre&gt;.csv.</p>")
    return "".join(out)


def hero_html(packs: dict[str, Any]) -> str:
    indexed = sum(len(p.index) for p in packs.values())
    types = " · ".join(f"{_esc(g)} <b>{len(p.types)}</b> types" for g, p in packs.items())
    return (f"<div class='nsr-hero'><div><h1>Negative-<em>Space</em><br>Reader</h1>"
            f"<p>Reads what a document <b>conspicuously doesn't say</b>: for every kind of statement its genre peers usually make, "
            f"how many of its nearest peers make it, with their own sentences as evidence. Prevalence, never a verdict.</p></div>"
            f"<div class='nsr-pills'><span class='nsr-pill'><b>{indexed:,}</b> indexed peers</span><span class='nsr-pill'>{types}</span>"
            f"<span class='nsr-pill'>no LLM on the hot path</span><span class='nsr-pill'>uploads never stored</span></div></div>")


def pack_choices(pack: Any) -> tuple[list[str], list[str]]:
    """Categories and years of a pack's indexed documents (for the cohort filters)."""
    payloads = getattr(pack.index, "payloads", None) or []
    if isinstance(payloads, dict):
        payloads = list(payloads.values())
    cats = sorted({p.get("category") for p in payloads if p.get("category")})
    years = sorted({str(p.get("date"))[:4] for p in payloads if p.get("date")})
    return cats, years


# ---------------------------------------------------------------- the app
def build_app(packs: dict[str, Any], cfg: dict[str, Any], *, examples: list[dict[str, Any]] | None = None,
              teacher_factory: Any = None, record: list[dict[str, Any]] | None = None,
              results: dict[str, Any] | None = None) -> Any:
    """The Blocks app (not launched). packs: name -> GenrePack; examples: app/examples.py records; teacher_factory:
    () -> teacher, used only when serve.allow_teacher is on and the box is ticked; record: if given, one
    {pack, seconds, sections} per analysis is appended (no text); results: app/content.py numbers for the About pages."""
    import gradio as gr

    from app.about import about_html
    from engine.onthefly import detect_pack
    from engine.pipeline import embed_upload, prepare, score

    scfg = cfg.get("serve") or {}
    if scfg.get("store_uploads", False):
        raise ValueError("serve.store_uploads must be false (plan §0.2 #21)")
    examples = examples or []
    by_name = {e["name"]: e for e in examples}
    choices = {name: pack_choices(p) for name, p in packs.items()}
    k_default = (cfg.get("cohort") or {}).get("k", 200)
    oth = cfg.get("onthefly") or {}
    allow_teacher, allow_live = bool(scfg.get("allow_teacher", False)), bool(scfg.get("allow_live_chat", False))
    NONE, ANY = "none", "any"

    def window(y0: str, y1: str) -> tuple[str | None, str | None] | None:
        if y0 in (None, ANY) and y1 in (None, ANY):
            return None
        return (None if y0 in (None, ANY) else f"{y0}-01-01", None if y1 in (None, ANY) else f"{y1}-12-31")

    def rescore(st: dict[str, Any], k: float, cat_on: bool, category: str, y0: str, y1: str, teach: bool) -> dict[str, Any]:
        pack = packs[st["pack"]]
        teacher = teacher_factory() if (teach and allow_teacher and teacher_factory) else None
        r = score(st["prepared"], pack, k=int(k), category_filter=bool(cat_on),
                  category=None if category in (None, NONE) else category, entity_id=st.get("entity_id"),
                  date_window=window(y0, y1), explain_mode="teacher" if teacher is not None else "template",
                  teacher=teacher)
        st["report"] = r.to_dict()
        return st

    def labels_of(rep: dict[str, Any]) -> list[str]:
        return [f"#{i + 1} {_label(a['type'])} ({_pct(a)}% of peers)" for i, a in enumerate(rep["absences"])]

    def outputs(st: dict[str, Any], compare: bool, note: str = "") -> tuple[Any, ...]:
        rep, meta = st["report"], st.get("meta") or {}
        found = set(meta.get("removed") or []) if (compare and meta.get("kind") == "benchmark") else None
        doc = document_html(st["prepared"].sections, rep["absences"], st["prepared"].title, found=found)
        labels = labels_of(rep)
        cats, years = choices[st["pack"]]
        status = narrate(rep, meta, detection=st.get("detection"), compare=compare, note=note)
        return (st, doc, status, cohort_html(rep, st.get("detection")), present_html(rep),
                gr.update(choices=labels, value=labels[0] if labels else None),
                evidence_html(rep["absences"][0] if rep["absences"] else None),
                gr.update(choices=[NONE] + cats), gr.update(choices=[ANY] + years), gr.update(choices=[ANY] + years),
                gr.update(value=compare_html(rep, st.get("meta"), allow_live=allow_live), visible=bool(compare)))

    def analyse_click(text, pack_name, meta, compare, k, cat_on, category, y0, y1, teach, progress=gr.Progress()):
        if not (text or "").strip():
            raise gr.Error("Paste a document, upload a file or pick an example first.")

        def tell(p: float, what: str) -> None:
            try:
                progress(p, desc=what)
            except Exception:  # called outside a request (tests): no progress bar to drive
                pass

        t0 = time.perf_counter()
        meta = meta or {}
        detection = None
        if pack_name == "auto":
            tell(0.1, "Reading and embedding the document, choosing the pack")
            detection = detect_pack(text, packs, neighbours=int(oth.get("detect_neighbours", 10)),
                                    threshold=oth.get("pack_similarity_threshold"))
            if detection["pack"] is None:
                raise gr.Error("No pack matches this document closely enough (on-the-fly mode is Phase 10).")
            pack_name = detection["pack"]
            embedded = detection["embedded"][pack_name]
            detection = {k_: v for k_, v in detection.items() if k_ != "embedded"}
        else:
            tell(0.1, "Reading and embedding the document")
            embedded = embed_upload(text, packs[pack_name])
        tell(0.5, f"Detecting which of the catalogue's kinds of statement its {len(embedded.sections)} sections make")
        prepared = prepare(text, packs[pack_name], embedded=embedded)
        tell(0.75, "Finding the 200 closest peers and counting what they state")
        same = meta.get("pack") == pack_name
        if category not in choices[pack_name][0]:  # a category of another pack or another document does not carry over
            category = meta.get("category") if same else None
        category = category or NONE
        st = {"pack": pack_name, "prepared": prepared, "detection": detection, "entity_id": meta.get("entity_id") if same else None,
              "meta": meta if same else {}}
        st = rescore(st, k, cat_on, category, y0, y1, teach)
        tell(0.95, "Placing the ghost paragraphs and quoting the peers")
        if record is not None:
            record.append({"pack": pack_name, "seconds": round(time.perf_counter() - t0, 3), "sections": len(prepared.sections)})
        out = outputs(st, compare)
        return out[:7] + (gr.update(choices=[NONE] + choices[pack_name][0], value=category),) + out[8:]

    def rerank_click(st, compare, k, cat_on, category, y0, y1, teach):
        if not st:
            raise gr.Error("Analyse a document first.")
        return outputs(rescore(st, k, cat_on, category, y0, y1, teach), compare, note="Re-ranked with the new cohort settings.")

    def pick_absence(st, label):
        if not st or not label:
            return evidence_html(None)
        i = int(label.split(" ", 1)[0].lstrip("#")) - 1
        return evidence_html(st["report"]["absences"][i])

    def pick_ghost(st, value):
        """A click on a ghost paragraph (nsr.js writes 'type|stamp' into the hidden box and presses the hidden button:
        a synthetic input event alone does not reach Gradio): select it, open Evidence."""
        if not st or not value:
            return gr.update(), gr.update(), gr.update()
        t = value.split("|", 1)[0]
        rep = st["report"]
        i = next((k for k, a in enumerate(rep["absences"]) if a["type"] == t), None)
        if i is None:
            return gr.update(), gr.update(), gr.update()
        return gr.update(value=labels_of(rep)[i]), evidence_html(rep["absences"][i]), gr.update(selected="evidence")

    def toggle_compare(st, compare):
        """The switch redraws the document (benchmark badges), the narration (challenge or reveal) and the panel."""
        if not st:
            return gr.update(), gr.update(), gr.update(value=compare_html(None, None), visible=bool(compare))
        out = outputs(st, compare)
        return out[1], out[2], out[10]

    def pick_example(name):
        e = by_name.get(name)
        if e is None:
            return gr.update(), None, gr.update(), gr.update()
        cat = e.get("category") if e.get("category") in choices[e["pack"]][0] else NONE
        meta = {k_: v for k_, v in e.items() if k_ != "markdown"}
        return e["markdown"], meta, e["pack"], gr.update(choices=[NONE] + choices[e["pack"]][0], value=cat)

    def upload(path):
        if not path:
            return gr.update(), None
        from engine.pipeline import read_document

        p = Path(path)
        if p.suffix.lower() not in (".txt", ".md", ".markdown", ".html", ".htm"):
            raise gr.Error("Upload a .txt, .md or .html file.")
        try:
            text = read_document(p)
        finally:
            if not scfg.get("store_uploads", False):
                try:
                    os.remove(p)  # plan §0.2 #21: the upload is kept in memory for this session only
                except OSError:
                    pass
        return text, None

    with gr.Blocks(title="Negative-Space Reader", delete_cache=(60, 120)) as demo:
        gr.HTML(hero_html(packs))
        meta = gr.State(None)
        state = gr.State(None)
        with gr.Tabs(elem_id="nsr-top"):
          with gr.Tab("Demo", id="demo"):
            gr.HTML(guide_html())
            with gr.Row(elem_classes=["nsr-card"]):
                with gr.Column(scale=3):
                    text = gr.Textbox(label="Document", placeholder="Paste a privacy policy or a paper (text, Markdown or HTML), upload a file, or pick an example →",
                                      lines=9, max_lines=14)
                    with gr.Row():
                        example = gr.Dropdown([e["name"] for e in examples], value=None, label="Example", visible=bool(examples), scale=3)
                        pack = gr.Dropdown(["auto"] + list(packs), value="auto", label="Pack", scale=1)
                with gr.Column(scale=2):
                    file = gr.File(label="Upload .txt / .md / .html", file_types=[".txt", ".md", ".html", ".htm"], type="filepath", height=150)
                    compare = gr.Checkbox(value=False, label="Compare with a chat model (cached 7.3 answers for benchmark examples)")
                    teach = gr.Checkbox(value=False, visible=allow_teacher,
                                        label="Teacher rationale and suggested wording (off by default: sends the peer quotes, not your document, to the teacher API)")
                    go = gr.Button("Analyse", variant="primary", elem_id="nsr-go")
            status = gr.HTML()
            with gr.Row():
                with gr.Column(scale=7, elem_classes=["nsr-card"]):
                    gr.HTML(legend_html())
                    doc = gr.HTML(empty_html(), max_height=960)
                    with gr.Accordion("Types this document states", open=False):
                        present = gr.HTML()
                with gr.Column(scale=5, elem_classes=["nsr-card"]):
                    with gr.Tabs(elem_id="nsr-tabs") as tabs:
                        with gr.Tab("Cohort", id="cohort"):
                            k = gr.Slider(10, 1000, value=k_default, step=10, label="k (peers)")
                            with gr.Row():
                                cat_on = gr.Checkbox(value=(cfg.get("cohort") or {}).get("category_filter", True), label="Category filter", scale=1)
                                category = gr.Dropdown([NONE], value=NONE, label="Category of this document", scale=2)
                            with gr.Row():
                                y0 = gr.Dropdown([ANY], value=ANY, label="Peers dated from")
                                y1 = gr.Dropdown([ANY], value=ANY, label="to")
                            rerank = gr.Button("Re-rank with these settings")
                            cohort = gr.HTML()
                        with gr.Tab("Evidence", id="evidence"):
                            absence = gr.Dropdown([], label="Absence")
                            evidence = gr.HTML(evidence_html(None))
            with gr.Column(visible=False, elem_classes=["nsr-card"]) as compare_box:
                gr.HTML("<div class='nsr-panel'><h3>Compare with a chat model</h3></div>")
                comparison = gr.HTML()
            with gr.Row(elem_id="nsr-pick-row"):  # the ghost-click bridge: nsr.js fills the box and presses the button
                pick = gr.Textbox(elem_id="nsr-pick", label="pick", visible=True)
                pick_go = gr.Button("pick", elem_id="nsr-pick-go")
          with gr.Tab("About the project", id="about"):
            gr.HTML(about_html(results))
        gr.HTML("<p class='nsr-small' style='margin-top:28px'>Every statement is a share of comparable documents that state a kind of "
                "thing, with the peers' own sentences as evidence. The system does not judge whether a document should state it. "
                "Documents are analysed in memory and never stored.</p>")
        controls = [k, cat_on, category, y0, y1, teach]
        outs = [state, doc, status, cohort, present, absence, evidence, category, y0, y1, comparison]
        if scfg.get("zerogpu"):  # a Hugging Face ZeroGPU Space: the GPU is attached only while this handler runs
            try:
                import spaces
            except ImportError as e:
                raise RuntimeError("serve.zerogpu needs the `spaces` package (Hugging Face ZeroGPU Spaces install it)") from e
            analyse_click = spaces.GPU(duration=int(scfg.get("zerogpu_seconds", 60)))(analyse_click)
        go.click(analyse_click, [text, pack, meta, compare] + controls, outs).then(lambda c: gr.update(visible=bool(c)), compare, compare_box)
        rerank.click(rerank_click, [state, compare] + controls, outs)
        absence.change(pick_absence, [state, absence], evidence)
        pick_go.click(pick_ghost, [state, pick], [absence, evidence, tabs])
        compare.change(toggle_compare, [state, compare], [doc, status, comparison]).then(lambda c: gr.update(visible=bool(c)), compare, compare_box)
        example.change(pick_example, example, [text, meta, pack, category])
        text.input(lambda: (None, NONE), None, [meta, category])  # typed or pasted text is not the example any more
        file.upload(upload, file, [text, meta])
    demo.nsr_handlers = {"analyse": analyse_click, "rerank": rerank_click, "absence": pick_absence, "ghost": pick_ghost,
                         "compare": toggle_compare, "example": pick_example, "upload": upload}
    return demo


def build_from_config(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Load the packs and examples named by the config and build the app: (demo, launch kwargs, the per-analysis
    record, load info). Shared by run.py's `serve` task and a Space's app.py."""
    import gradio as gr

    from app.content import read_results
    from app.examples import load_examples
    from engine.embed import Embedder
    from experiments.distil import load_colbert
    from experiments.run import resolve_device
    from packs.spec import GenrePack

    scfg = cfg.get("serve") or {}
    genres = list(scfg.get("packs") or cfg["genres"])
    t0 = time.perf_counter()
    device = resolve_device(cfg["run"].get("device", "auto"))
    ecfg = cfg["models"]["embedder"]
    embedder = Embedder(ecfg["id"], devices=[device], dtype=ecfg.get("dtype", "float16"),
                        max_length=ecfg.get("max_length", 1024),
                        batch_size=ecfg.get("query_batch_size", ecfg.get("batch_size", 32)), normalize=True,
                        revision=ecfg.get("revision"))
    colbert = load_colbert(cfg)
    packs = {g: GenrePack.load(cfg, g, embedder=embedder, colbert=colbert, device=device) for g in genres}
    examples = load_examples(cfg, genres, per_genre=scfg.get("examples_per_genre", 3), seed=cfg["run"].get("seed", 0))
    load_s = round(time.perf_counter() - t0, 1)

    def teacher_factory() -> Any:
        from models.teacher import load_teacher

        return load_teacher(cfg["models"]["teacher"])

    record: list[dict[str, Any]] = []
    results = read_results(scfg.get("examples_dir", "app/examples"))
    demo = build_app(packs, cfg, examples=examples, teacher_factory=teacher_factory, record=record, results=results)
    theme = gr.themes.Base(primary_hue="teal", secondary_hue="orange", neutral_hue="stone",  # the fonts come from head()
                           font=[gr.themes.Font("DM Sans"), gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")],
                           font_mono=[gr.themes.Font("JetBrains Mono"), gr.themes.Font("ui-monospace"), gr.themes.Font("monospace")])
    launch = dict(server_name=scfg.get("host", "127.0.0.1"), server_port=int(scfg.get("port", 7860)), theme=theme, css=css(),
                  head=head(), js=FORCE_THEME_JS, max_file_size=f"{scfg.get('max_upload_mb', 20)}mb", footer_links=[])
    info = {"packs": genres, "device": device, "load_seconds": load_s, "examples": len(examples), "store_uploads": False}
    return demo, launch, record, info


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """run.py task `serve`: launch, run until Ctrl-C or SIGTERM, then report the analyses as the run's metrics."""
    import signal
    import threading

    demo, launch, record, info = build_from_config(cfg)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    demo.launch(prevent_thread_lock=True, **launch)
    print(f"[serve] {', '.join(info['packs'])} loaded in {info['load_seconds']} s on {info['device']}; {info['examples']} examples; "
          f"http://{launch['server_name']}:{launch['server_port']}", flush=True)
    try:
        while not stop.wait(1):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        demo.close()
    m: dict[str, Any] = {**info, "analyses": len(record)}
    for g in info["packs"]:
        s = [r["seconds"] for r in record if r["pack"] == g]
        if s:
            m[f"{g}_median_seconds"], m[f"{g}_max_seconds"] = round(statistics.median(s), 3), max(s)
    return m
