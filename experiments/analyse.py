"""Analyse named corpus documents with a pack (plan 5.6, 5.8): one Report JSON per document in the run directory.

    python -m experiments.run --config configs/analyse.yaml --name explain_teacher_privacy --overlay openrouter \
        --set pack=privacy --set 'analyse.doc_ids=[privacy_pl:172230]' --set explain.mode=teacher --set 'run.seeds=[0]'

Documents are rebuilt as Markdown from their sections (as the gate does). In teacher mode the metrics list, per
document and absence, the words of the teacher's rationale and suggested wording that neither the quotes it was given
nor the type card contain (engine/explain.py unsupported_words): the check that the wording stays within the quotes.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def main(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from data.schema import Document, Section, read_jsonl
    from engine.explain import label, unsupported_words
    from engine.pipeline import analyse, document_markdown
    from packs.builder import _path
    from packs.spec import GenrePack

    genre, ids = cfg["pack"], list((cfg.get("analyse") or {}).get("doc_ids") or [])
    if not ids:
        raise ValueError("analyse.doc_ids is empty: name the documents to analyse")
    mode = (cfg.get("explain") or {}).get("mode", "template")
    processed = _path(cfg["paths"]["data_processed"]) / genre
    docs = {d.id: d for d in read_jsonl(processed / "documents.jsonl", Document) if d.id in ids}
    missing = [i for i in ids if i not in docs]
    if missing:
        raise KeyError(f"documents not in {processed}: {missing}")
    secs: dict[str, list[Any]] = defaultdict(list)
    for s in read_jsonl(processed / "sections.jsonl", Section):
        if s.doc_id in docs:
            secs[s.doc_id].append(s)
    pack = GenrePack.load(cfg, genre)
    cards = {t["id"]: t for t in pack.schema["types"]}
    info = (cfg.get("genre_info") or {}).get(genre) or {}
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seconds, unsupported, taught = [], {}, 0
    for i in ids:
        d = docs[i]
        r = analyse(document_markdown(d, secs[i]), pack, entity_id=d.entity_id, explain_mode=mode)
        (run_dir / f"report_{re.sub(r'[^A-Za-z0-9._-]', '_', i)}.json").write_text(r.to_json(indent=1) + "\n", encoding="utf-8")
        seconds.append(r.timings["total"])
        flagged = {}
        for a in r.absences:
            if a.get("explanation_mode") != "teacher":
                continue
            taught += 1
            card = cards[a["type"]]
            sources = [e["text"] for e in a.get("evidence", []) if e["source"] == "cohort"]
            sources += [label(card), card.get("definition", ""), info.get("document_kind", ""), info.get("author_role", "")]
            flagged[a["type"]] = unsupported_words(f"{a['rationale']} {a['suggested_wording']}", sources)
        if mode == "teacher":
            unsupported[i] = flagged
        print(f"[analyse {genre}] {i}: {r.sections} sections, {len(r.absences)} absences, {r.timings['total']:.2f} s",
              flush=True)
    m: dict[str, Any] = {"documents": len(ids), "explain_mode": mode, "max_seconds": max(seconds),
                         "device": (getattr(pack, "info", {}) or {}).get("device", "?")}
    if mode == "teacher":
        m.update(teacher_explanations=taught, unsupported_words=unsupported,
                 unsupported_total=sum(len(w) for f in unsupported.values() for w in f.values()))
    (run_dir / "unsupported_words.json").write_text(json.dumps(unsupported, indent=1) + "\n", encoding="utf-8")
    return m
