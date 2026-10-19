"""engine/pipeline.py (plan 5.6): analyse(document, pack) -> Report on a small pack built from simple parts.

Three types: "retention", "sharing", "cookies". The corpus has 60 documents; nearly all state retention, half state
sharing, few state cookies. The query document states sharing only, so retention should be its top absence, with
peer quotes from the cohort and a template explanation; cookies (rare among peers) ranks below it."""

import json

import numpy as np
import pandas as pd
import pytest

from engine.doc_presence import DocPresence
from engine.expectation import ProfileStore
from engine.index import make_index
from packs.spec import GenrePack

TYPES = ["retention", "sharing", "cookies"]
KEYS = {"retention": "keep", "sharing": "share", "cookies": "cookie"}


class KeywordEmbedder:
    def encode(self, texts, **kw):
        V = np.array([[float(k in t.lower()) for k in KEYS.values()] + [0.3, 0.1 * (len(t) % 7)] for t in texts])
        return V / np.linalg.norm(V, axis=1, keepdims=True)


class Heads:
    def predict_proba(self, V):
        return np.clip(V[:, :3] * 3, 0.02, 0.98)


class Cascade:
    threshold = np.full(3, 0.5)

    def predict(self, clf, proto, rr):
        return 0.8 * clf + 0.2 * proto


class Scorer:
    def score(self, texts):
        return np.array([[float(k in t.lower()) for k in KEYS.values()] for t in texts])


@pytest.fixture
def pack(tmp_path):
    from experiments.config import load_config

    cfg = load_config("configs/base.yaml", [])
    cfg["cohort"].update(k=20, category_filter=False)
    rng = np.random.default_rng(0)
    emb, ids, payloads, rows, texts, best = KeywordEmbedder(), [], [], [], {}, {}
    for i in range(60):
        d = f"p{i}"
        has = {"retention": i % 10 != 0, "sharing": i % 2 == 0, "cookies": i % 15 == 0}
        sec = [f"Section about the service number {i}."] + [f"We {KEYS[t]} things here for peer {i}." for t, v in has.items() if v]
        for j, s in enumerate(sec):
            texts[(d, j)] = s
        ids.append(d)
        payloads.append({"genre": "g", "entity_id": "own" if i < 3 else f"e{i}", "category": None, "date": "2020-01-01"})
        rows.append({"doc_id": d, **{f"bit_{t}": int(v) for t, v in has.items()},
                     **{f"prob_{t}": float(v) for t, v in has.items()}})
        best[d] = np.array([next((j for j, s in enumerate(sec) if KEYS[t] in s), 0) for t in TYPES])
    X = emb.encode([" ".join(s for (dd, _), s in texts.items() if dd == d) for d in ids])
    idx = make_index("faiss", X.shape[1])
    idx.add(ids, X, payloads)
    (tmp_path / "profiles").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "profiles" / "part-00000.parquet", index=False)
    docs = [np.clip(rng.random((4, 3)) * 0.2 + np.eye(3)[[k % 3]] * 0.7, 0, 1) for k in range(90)]
    Y = np.array([(P >= 0.5).any(0) for P in docs], dtype=int)
    doc_model = DocPresence(("logit_max",)).fit(docs, Y, C_grid=(1.0,))
    schema = {"version": 1, "types": [{"id": t, "name": t, "definition": f"States its {t} practice.", "severity_weight": 1.0,
                                       "section_hint": "middle"} for t in TYPES]}
    protos = {t: [{"text": f"Prototype for {t}.", "doc_id": "lab:1", "section_idx": 0}] for t in TYPES}
    return GenrePack(genre="g", cfg=cfg, schema=schema, types=TYPES, heads=Heads(), cascade=Cascade(), scorer=Scorer(),
                     doc_model=doc_model, index=idx, profiles=ProfileStore.load(tmp_path / "profiles", TYPES), best=best,
                     section_texts=lambda d, j: texts.get((d, j)), prototypes=protos, embedder=emb, info={"tier": "test"})


DOC = "# Our policy\nWe share your data with partners when you ask us to, and only then.\n\n# Contact\nWrite to us at the address below for anything."


def test_analyse_reports_present_types_ranked_absences_and_evidence(pack):
    from engine.pipeline import analyse

    r = analyse(DOC, pack, entity_id="own")
    assert [p["type"] for p in r.present] == ["sharing"] and r.sections >= 1 and r.title == "Our policy"
    assert [a["type"] for a in r.absences][:2] == ["retention", "cookies"]
    top = r.absences[0]
    assert top["n"] == 20 and top["count"] / top["n"] == pytest.approx(top["expected"])
    assert top["ci"][0] <= top["expected"] <= top["ci"][1]
    assert top["evidence"][0]["source"] == "cohort" and "keep" in top["evidence"][0]["text"]
    assert top["explanation"].startswith(f"{round(100 * top['expected'])}% of peers like this one")
    assert "this document does not" in top["explanation"]
    assert r.cohort["n"] == 20 and all(n["doc_id"] not in ("p0", "p1", "p2") for n in r.cohort["nearest"])  # own entity
    assert {"embed", "presence", "cohort", "evidence", "total"} <= set(r.timings)
    assert json.loads(r.to_json())["genre"] == "g"


def test_analyse_without_a_document_model_uses_the_v1_rule(pack):
    from engine.pipeline import analyse

    pack.doc_model = None
    r = analyse(DOC, pack)
    assert [p["type"] for p in r.present] == ["sharing"]


def test_analyse_reads_files_and_refuses_empty_documents(pack, tmp_path):
    from engine.pipeline import analyse

    f = tmp_path / "doc.html"
    f.write_text("<html><body><h1>Policy</h1><p>We share data with partners when asked.</p><script>x()</script></body></html>")
    r = analyse(f, pack)
    assert any(p["type"] == "sharing" for p in r.present)
    with pytest.raises(ValueError):
        analyse("   ", pack)


def test_teacher_explanations_are_opt_in_and_fill_the_same_fields(pack):
    from engine.explain import EXPLAIN_FIELDS
    from engine.pipeline import analyse

    class T:
        def generate_structured_batch(self, prompts, model, **kw):
            return [model(rationale="Peers say how long they keep data.", suggested_wording="We keep it for [period].")
                    for _ in prompts]

    pack.cfg["genre_info"]["g"] = {"document_kind": "test policy", "author_role": "the company"}
    plain = analyse(DOC, pack)
    taught = analyse(DOC, pack, explain_mode="teacher", teacher=T())
    assert all(set(EXPLAIN_FIELDS) <= set(a) for a in plain.absences + taught.absences)
    assert plain.absences[0]["explanation_mode"] == "template" and taught.absences[0]["explanation_mode"] == "teacher"
    assert taught.absences[0]["suggested_wording"] == "We keep it for [period]."


def test_analyse_gate_times_held_out_documents(pack, tmp_path, monkeypatch):
    from data.schema import Document, Section, write_jsonl
    from packs import builder

    processed = tmp_path / "processed" / "g"
    processed.mkdir(parents=True)
    docs = [Document(id=f"q{i}", genre="g", source="s", entity_id=f"qe{i}", title=f"Policy {i}") for i in range(4)]
    write_jsonl(processed / "documents.jsonl", docs)
    write_jsonl(processed / "sections.jsonl", [Section(doc_id=d.id, idx=0, heading="Data",
                                                       text="We share your data with partners when you ask.") for d in docs])
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "g.json").write_text(json.dumps({"doc_ids": {"train": [], "dev": [d.id for d in docs], "test": []}}))
    cfg = pack.cfg
    cfg["paths"].update(data_processed=str(tmp_path / "processed"), splits=str(tmp_path / "splits"),
                        tables=str(tmp_path / "tables"))
    cfg["build"] = {"analyse_gate": {"documents": 3, "max_seconds": 10}}
    m = builder.analyse_gate(cfg, "g", pack=pack)
    assert m["documents"] == 3 and m["gate"] == "pass" and m["max_seconds"] < 10 and m["median_seconds"] <= m["max_seconds"]
    assert "embed" in m["median_timings"] and m["device"]
    assert (tmp_path / "tables" / "analyse_gate_g.csv").is_file()
