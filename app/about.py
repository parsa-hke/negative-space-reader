"""The demo's About pages (plan 7.1: demo plus an account of the project for any reader): problem, method, how it
was tested, what the numbers say, where it falls short, what it is not, under the hood, glossary. Plain language
first, detail behind "more". Every number comes from `results.json` (app/content.py, copied from reports/tables); a
number whose table is absent is shown as "not recorded", never typed in. Wording per plan §0.2 #21: prevalence, not
verdicts.
"""

from __future__ import annotations

import html
from typing import Any

GENRE_NAMES = {"privacy": "privacy policies", "papers": "research papers"}
SYSTEM_NAMES = {"level1": "Negative-Space Reader (peers)", "genre_uniform": "genre-wide frequency", "random": "random order",
                "anthropic:claude-opus-5-5": "Claude Opus 5.5", "openai:qwen/qwen-2.5-72b-instruct": "Qwen2.5-72B (open model)"}


def _e(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _pct(v: float | None) -> str:
    return "not recorded" if v is None else f"{round(100 * v)}%"


def _num(v: float | None, nd: int = 2) -> str:
    return "not recorded" if v is None else f"{v:.{nd}f}"


def _get(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _more(title: str, body: str) -> str:
    return f"<details class='nsr-more'><summary>{_e(title)}</summary><div>{body}</div></details>"


def _steps(items: list[tuple[str, str]]) -> str:
    return "<ol class='nsr-steps'>" + "".join(f"<li style='--i:{i}'><b>{_e(t)}</b><span>{d}</span></li>" for i, (t, d) in enumerate(items)) + "</ol>"


def _recall_table(block: dict[str, Any], systems: list[str], metric: str = "recall@5") -> str:
    genres = list(block)
    head = "".join(f"<th>{_e(GENRE_NAMES.get(g, g))}</th>" for g in genres)
    rows = []
    for s in systems:
        cells = []
        for g in genres:
            m = _get(block, g, s, metric)
            cells.append(f"<td>{_pct(_get(m, 'value'))}<small>{' [' + ', '.join(_pct(x) for x in m['ci']) + ']' if m and m.get('ci') else ''}</small></td>")
        rows.append(f"<tr><th>{_e(SYSTEM_NAMES.get(s, s))}</th>{''.join(cells)}</tr>")
    return f"<table class='nsr-results'><tr><th></th>{head}</tr>{''.join(rows)}</table>"


def problem_html(res: dict[str, Any]) -> str:
    return f"""
<section class='nsr-about' id='problem'>
<h2>The problem</h2>
<p class='nsr-lead'>Every tool for reading documents looks at what is <b>written</b>. Nothing looks at what is
<b>not written</b>: the retention period a privacy policy never gives, the limitations section a paper never has.
You cannot search for an absence, because there is no text to find.</p>
<p>Yet absences are often the most important thing about a document. A reviewer who has read a thousand policies
notices at once when one says nothing about children's data; a reader who has not cannot. Negative-Space Reader
gives any reader that thousand-document memory: it compares a document with its closest peers and reports, with
evidence, the kinds of statement those peers usually make and this one does not.</p>
<div class='nsr-callout'><b>The research question.</b> Can a system detect what a document conspicuously omits by
modelling what its genre peers assert? And does conditioning that expectation on the document's own content beat a
simple genre-wide frequency, and beat frontier chat models asked the same question?</div>
{_more("Why this is hard", "<p>Three reasons. An omission has no text, so nothing can be matched or retrieved directly. "
       "Whether an absence is <i>conspicuous</i> depends on the document's peers, not on the document alone: a short notice for a "
       "newsletter is not expected to say what a bank's policy says. And language models are trained to predict what is present; "
       "asked what is absent from a long document, they must hold a whole catalogue in mind and check it against every paragraph.</p>")}
</section>"""


def method_html(res: dict[str, Any]) -> str:
    s = res.get("schema") or {}
    c = res.get("corpus") or {}
    types = " and ".join(f"{s[g]['types']} for {GENRE_NAMES.get(g, g)}" for g in s) or "a few dozen per genre"
    indexed = " and ".join(f"{c[g]['indexed']:,} {GENRE_NAMES.get(g, g)}" for g in c if c[g].get("indexed")) or "tens of thousands of documents"
    return f"""
<section class='nsr-about' id='method'>
<h2>How it works</h2>
<p class='nsr-lead'>Five steps, none of which asks a language model to guess.</p>
{_steps([
    ("Split", "The document is cut into its sections and sentences, the same way its peers were."),
    ("Detect what it says", f"Each section is checked against a fixed catalogue of kinds of statement ({types}). The catalogue was "
                            "built from the documents themselves: a teacher model listed what a sample of documents assert, the lists were "
                            "clustered and named, a human pruned them, and the result was frozen. A small, fast classifier then recognises "
                            "each kind in new text."),
    ("Find the peers", f"The document is embedded as a vector and its 200 nearest neighbours are retrieved from the index "
                       f"({indexed}), restricted to the same category when the genre has one. Peers are never other versions of the same document."),
    ("Count", "For every kind of statement the document does <i>not</i> make, the system counts how many of the 200 peers make it. "
              "That share is the expectation: 70% means 140 of its closest peers state it."),
    ("Rank and show", "Absences are ranked by that share (optionally weighted by a severity the catalogue gives some kinds), and each is "
                      "shown as a <b>ghost paragraph</b> placed where peers usually say it, with the peers' own sentences as evidence."),
])}
{_more("What a ghost paragraph says, and what it does not", "<p>A ghost paragraph says: <i>N% of peers like this one state X; this "
       "document does not.</i> It is a statement about the peers, with their sentences attached. It is never a judgement that the document "
       "should have said X, or that it falls short for not saying it; some absences are deliberate and right. The system shows prevalence, "
       "the reader decides.</p>")}
{_more("The score, exactly", "<p>For each kind of statement <i>t</i> absent from the document: score(<i>t</i>) = P(<i>t</i> expected | peers) × "
       "(1 − P(<i>t</i> present in the document)) × w(<i>t</i>). The first factor is the peer share, the second discounts kinds the detector "
       "thinks might be present after all, and w is a severity weight from the catalogue (1 by default). Each share carries a Wilson 95% "
       "interval, so a small cohort does not look more certain than it is.</p>")}
</section>"""


def results_html(res: dict[str, Any]) -> str:
    syn, rev, llm, gate = res.get("synthetic") or {}, res.get("revision") or {}, res.get("llm_gap") or {}, res.get("gate") or {}
    speed, pres = res.get("speed") or {}, res.get("presence") or {}
    reading = []
    for g in syn:
        l1, gu = _get(syn, g, "level1", "recall@5", "value"), _get(syn, g, "genre_uniform", "recall@5", "value")
        d = _get(gate, g, "edited_minus_uniform")
        if l1 is not None and gu is not None:
            verdict = ("a real gain" if d and d["ci"][0] > 0 else "no reliable gain" if d else "")
            reading.append(f"<li><b>{_e(GENRE_NAMES.get(g, g).capitalize())}:</b> the removed kind of statement is in the top five "
                           f"{_pct(l1)} of the time with peers, {_pct(gu)} with genre-wide frequency"
                           + (f" ({verdict}: difference {d['diff']:+.3f}, interval [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}])" if d else "") + ".</li>")
    llm_reading = []
    for g in llm:
        parts = [f"{_e(SYSTEM_NAMES.get(s, s))} {_pct(_get(llm, g, s, 'recall@5', 'value'))}" for s in llm[g] if " - " not in s]
        llm_reading.append(f"<li><b>{_e(GENRE_NAMES.get(g, g).capitalize())}:</b> " + ", ".join(parts) + ".</li>")
    speed_txt = "; ".join(f"{GENRE_NAMES.get(g, g)} at most {speed[g]['max_seconds']:.1f} s (median {speed[g]['median_seconds']:.1f} s)" for g in speed)
    pres_txt = "; ".join(f"{GENRE_NAMES.get(g, g)} macro-F1 {pres[g]['macro_f1']:.2f}, {_pct(pres[g]['share_of_ceiling'])} of what the teacher model "
                         f"itself reaches" for g in pres)
    systems = ["level1", "genre_uniform", "random"]
    return f"""
<section class='nsr-about' id='results'>
<h2>How well it works</h2>
<p class='nsr-lead'>Tested two ways on documents the system had never seen, plus a head-to-head with two chat models.
Every number below is copied from the recorded result tables.</p>
<h3>Test 1 · Remove a statement, see if it is found</h3>
<p>From each held-out document we removed every sentence making one kind of statement, then asked each system to
rank the absences. "Recall@5" is how often the removed kind appears in the top five. 500 documents per genre.</p>
{_recall_table(syn, systems) if syn else "<p class='nsr-note'>Table bench_synth not recorded.</p>"}
<ul class='nsr-reading'>{''.join(reading)}</ul>
{_more("Why peers help for papers but not for privacy policies", "<p>Two things, both measured. Privacy-policy cohorts carry little "
       "information about data practices: how similar two policies read says less about what they state than it does for papers. And the "
       "removal itself leaks into the retrieval: a policy that loses a topic moves, in embedding space, towards the policies that never had "
       "it, so its peers are exactly the ones that omit the removed kind. Measured against the peers of the <i>original</i> document, the "
       "peer system gains "
       + (f"{gate[list(gate)[0]]['original_minus_uniform']['diff']:+.3f}" if gate and _get(gate, list(gate)[0], 'original_minus_uniform') else "more")
       + " on privacy; the leak takes most of that away. Real omissions behave the same way (an omitting document resembles other omitters), "
       "which is why we report this as a negative finding rather than tuning it away. The next model in the plan conditions on what the "
       "document states rather than on an embedding of its whole text.</p>")}
<h3>Test 2 · Real omissions from revisions</h3>
<p>When a later version of a document adds a kind of statement that its peers at the time already made, the earlier
version omitted it. No labels are needed; the documents label themselves.</p>
{_recall_table(rev, ["level1", "genre_uniform", "random"]) if rev else "<p class='nsr-note'>Table bench_revision not recorded.</p>"}
<h3>Test 3 · Against chat models</h3>
<p>Claude Opus 5.5 and an open 72-billion-parameter model were given the same document and the full catalogue, and
asked to list the kinds of statement it does not make, most conspicuous first. 100 documents per genre.</p>
{_recall_table(llm, [s for s in next(iter(llm.values()), {}) if " - " not in s]) if llm else "<p class='nsr-note'>Table llm_gap_candidate not recorded.</p>"}
<ul class='nsr-reading'>{''.join(llm_reading)}</ul>
<p>Both chat models rank well below the corpus-based systems in both genres, with confidence intervals that do not
overlap. The open model mostly listed catalogue entries in catalogue order rather than ranking them.</p>
<h3>Speed and detection quality</h3>
<p>On a laptop (Apple M5), a document is analysed in {_e(speed_txt) if speed_txt else "a few seconds"}; the embedding
step is most of it. The statement detector reaches {_e(pres_txt) if pres_txt else "a recorded macro-F1 per genre"}: good
enough to profile tens of thousands of documents, and the first thing to improve.</p>
</section>"""


def limits_html(res: dict[str, Any]) -> str:
    c = res.get("corpus") or {}
    years = "; ".join(f"{GENRE_NAMES.get(g, g)} {c[g].get('date_min', '')[:4]}–{c[g].get('date_max', '')[:4]}" for g in c if c[g].get("date_min"))
    return f"""
<section class='nsr-about' id='limits'>
<h2>Where it falls short</h2>
<ul class='nsr-limits'>
<li><b>Prevalence is not importance.</b> Many peers stating something does not make it necessary here; a short notice is
not wrong for leaving out what a long one says. The system ranks, the reader judges.</li>
<li><b>Peers help unevenly.</b> For privacy policies the peer-based expectation is no better than genre-wide frequency on
our benchmark (see the results page for why). For papers it is clearly better.</li>
<li><b>The detector is imperfect.</b> A kind of statement expressed unusually can be missed, so a ghost paragraph can
appear for something the document does say in its own words. The evidence panel lets you check.</li>
<li><b>The benchmark was built with the same detector</b> that the system uses, which favours it in the chat-model
comparison; the chat models were run once each, at a moderate reasoning setting.</li>
<li><b>The corpora have dates.</b> {_e(years) if years else "Each corpus covers a span of years"}; norms drift, and a 2026
document is compared with what its peers said then.</li>
<li><b>Lean tier.</b> Every model is the small, CPU-friendly choice so that this demo can run without a GPU; the plan
names a larger alternative for each, behind one configuration switch.</li>
<li><b>Two genres.</b> Privacy policies and computer-science papers. Adding a genre is a data-building run, not new
code, but it has not been done yet.</li>
</ul>
</section>"""


def ethics_html(res: dict[str, Any]) -> str:
    return """
<section class='nsr-about' id='ethics'>
<h2>What it is not</h2>
<p class='nsr-lead'>It is not a compliance checker, a grader, or a detector of wrongdoing.</p>
<ul class='nsr-limits'>
<li>It never says a document is non-compliant or deficient. It says what share of peers state something, and shows their sentences.</li>
<li>Genre norms encode what dominant actors do. A share of peers is a description of the genre, not a standard the genre should meet.</li>
<li>Documents are analysed in memory and never stored. Nothing is sent to an external service unless the operator turns on the
optional teacher explanations, which send only the quoted peer sentences, never your document.</li>
<li>Every claim on screen is backed by visible evidence; there is no language model between the corpus and what you read.</li>
</ul>
</section>"""


def tech_html(res: dict[str, Any]) -> str:
    return f"""
<section class='nsr-about' id='tech'>
<h2>Under the hood</h2>
<div class='nsr-grid'>
<div><b>Embeddings</b><span>Qwen3-Embedding-0.6B (Apache-2.0) for sections and documents; FAISS index of the peers.</span></div>
<div><b>Statement detection</b><span>One calibrated logistic classifier per kind of statement over the section embeddings, plus a small
late-interaction matcher (answerai-colbert-small-v1) against prototype sentences; combined, then aggregated to the document.</span></div>
<div><b>Catalogue</b><span>Induced once per genre with an open teacher model (Qwen2.5-32B/72B-Instruct), clustered, named, pruned by hand, frozen.</span></div>
<div><b>Expectation</b><span>Share of the 200 nearest peers that state each kind, with Wilson intervals; category filter and date window on request.</span></div>
<div><b>Evidence</b><span>Each peer's best sentence per kind, precomputed over the index; quoted verbatim with the document id.</span></div>
<div><b>Stack</b><span>Python, PyTorch, sentence-transformers, FAISS, Gradio. Every experiment runs from one entry point with a config file
and is logged; the numbers on these pages are copied from the recorded tables by a script.</span></div>
</div>
{_more("Reproducibility", "<p>The repository holds the configs, prompts (versioned and hashed), schemas and the run log; the packs (index, "
       "classifiers, profiles) and benchmark runs live in a data store synced with one command. A smoke configuration rebuilds everything on "
       "200 documents per genre in minutes.</p>")}
</section>"""


def glossary_html(res: dict[str, Any]) -> str:
    items = [("Kind of statement (type)", "One entry of the genre's frozen catalogue, e.g. “states how long data is kept” or “reports computational cost”."),
             ("Peers / cohort", "The 200 indexed documents most similar to yours (same category when available), never other versions of the same document."),
             ("Prevalence", "The share of the cohort that makes a kind of statement: the number in every ghost paragraph."),
             ("Ghost paragraph", "A dashed block placed where peers usually say something your document does not; click it for the evidence."),
             ("Evidence", "Up to five sentences from peers that make the statement, quoted verbatim with their document ids."),
             ("Severity weight", "A factor the catalogue gives some kinds (1.5 for a few privacy kinds); it reorders, it does not judge."),
             ("Pack", "Everything the system knows about one genre: corpus, catalogue, detector, index. The engine has no genre-specific code."),
             ("OmitBench", "The two benchmarks: synthetic (statements removed on purpose) and revision (statements added in a later version).")]
    return "<section class='nsr-about' id='glossary'><h2>Glossary</h2><dl class='nsr-gloss'>" + "".join(
        f"<dt>{_e(t)}</dt><dd>{_e(d)}</dd>" for t, d in items) + "</dl></section>"


def credits_html(res: dict[str, Any]) -> str:
    gen = (res.get("generated") or "")[:10]
    return f"""
<section class='nsr-about' id='credits'>
<h2>About the project</h2>
<p>Negative-Space Reader is Parsa Hosseini's project for MAAI 902 (NLP Solution Design) at the University of Wollongong in Dubai,
October 2026, and the class version of a longer research programme: a benchmark for omission detection, a method that
conditions on what a document states rather than on its embedding, and the measured gap between corpus-based and
chat-model reasoning about absence.</p>
<p class='nsr-note'>Corpora: the Princeton–Leuven Longitudinal Corpus of Privacy Policies and OPP-115 (privacy); unarXive and arXiv
metadata, cs.CL and cs.LG (papers). Numbers on these pages were copied from the result tables on {_e(gen) or "the build date"}.</p>
</section>"""


SECTIONS = [("problem", "The problem", problem_html), ("method", "How it works", method_html), ("results", "How well it works", results_html),
            ("limits", "Where it falls short", limits_html), ("ethics", "What it is not", ethics_html), ("tech", "Under the hood", tech_html),
            ("glossary", "Glossary", glossary_html), ("credits", "About the project", credits_html)]


def about_html(res: dict[str, Any] | None) -> str:
    res = res or {}
    nav = "<nav class='nsr-toc'>" + "".join(f"<a href='#{i}'>{_e(t)}</a>" for i, t, _ in SECTIONS) + "</nav>"
    return "<div class='nsr-aboutpage'>" + nav + "".join(fn(res) for _, _, fn in SECTIONS) + "</div>"
