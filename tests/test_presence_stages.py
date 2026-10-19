"""Presence stages of experiments/distil.py (plan 4.3-4.7) end to end on a tiny synthetic pack: each section is about
"alpha" and/or "beta"; its embedding encodes which; the teacher is right, the reference (judge) agrees."""

import json

import numpy as np
import pytest
import yaml

from data.schema import Section, write_jsonl

TYPES = [{"id": "alpha", "definition": "States alpha.", "positive_examples": ["Alpha is stated.", "Alpha again."],
          "label_source": ["teacher"]},
         {"id": "beta", "definition": "States beta.", "positive_examples": ["Beta is stated.", "Beta again."],
          "label_source": ["teacher"]}]


def _text(a, b, i):
    parts = [f"Section {i} has some neutral words about the service."]
    if a:
        parts.append("Alpha is stated here clearly.")
    if b:
        parts.append("Beta is described in this part.")
    return " ".join(parts)


@pytest.fixture
def pack(tmp_path):
    from engine.embed import EmbeddingStore
    from experiments.config import load_config

    rng = np.random.default_rng(0)
    (tmp_path / "packs" / "g").mkdir(parents=True)
    (tmp_path / "packs" / "g" / "schema.yaml").write_text(yaml.safe_dump({"version": 1, "status": "frozen", "types": TYPES}))
    secs, teacher, ref, split = [], [], [], {}
    for i in range(160):  # 120 teacher-set documents (train), 40 reference documents (dev)
        a, b = bool(rng.random() < 0.4), bool(rng.random() < 0.3)
        doc = f"d{i:03d}"
        s = Section(doc_id=doc, idx=0, heading="H", text=_text(a, b, i))
        secs.append(s)
        found = [t for t, v in (("alpha", a), ("beta", b)) if v]
        spans = {"alpha": "Alpha is stated here clearly.", "beta": "Beta is described in this part."}
        row = {"doc_id": doc, "section_idx": 0, "failed": False, "labels": {"alpha": int(a), "beta": int(b)},
               "found_by": {t: "first" for t in found}, "evidence": {t: f"{spans[t]}" for t in found},
               "grounded": {t: True for t in found}}
        teacher.append(row)
        if i >= 120:
            ref.append(row)
        split[doc] = "train" if i < 120 else "dev"
    processed = tmp_path / "processed" / "g"
    processed.mkdir(parents=True)
    write_jsonl(processed / "sections.jsonl", secs)
    from data.schema import Document

    write_jsonl(processed / "documents.jsonl", [Document(id=d, genre="g", source="s", entity_id=d, version=None,
                                                         date="2020-01-01", category="c", title=None, meta={})
                                                for d in split])
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "g.json").write_text(json.dumps(
        {"doc_ids": {s: [d for d, v in split.items() if v == s] for s in ("train", "dev", "test")}}))
    for name, rows in (("teacher", teacher), ("reference", ref)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    X = np.array([[float("Alpha" in s.text), float("Beta" in s.text)] + list(rng.normal(scale=0.3, size=6)) for s in secs],
                 dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    cfg = load_config("configs/distil.yaml", ["smoke"])
    store = tmp_path / "emb" / cfg["models"]["embedder"]["id"].replace("/", "__") / "g" / "sections"
    EmbeddingStore.save(store, [[s.doc_id, s.idx] for s in secs], [s.text for s in secs], X, {"model": "fake"})
    cfg["pack"] = "g"
    cfg["run"]["seed"] = 0
    cfg["paths"].update(packs=str(tmp_path / "packs"), data_processed=str(tmp_path / "processed"),
                        embeddings=str(tmp_path / "emb"), tables=str(tmp_path / "tables"), splits=str(tmp_path / "splits"))
    cfg["presence"]["labels"] = {"teacher": str(tmp_path / "teacher.jsonl"), "reference": str(tmp_path / "reference.jsonl"),
                                 "annotations": {}}
    cfg["genre_info"]["g"] = {"document_kind": "test document", "author_role": "the writer"}
    return cfg, tmp_path


def test_heads_stage(pack):
    from experiments import distil

    cfg, tmp = pack
    cfg["distil"]["stage"] = "heads"
    m = distil.main(cfg, tmp)
    assert m["g/teacher_train_sections"] == 120 and m["g/reference_fit"] + m["g/reference_eval"] == 40
    assert m["g/vs_reference/micro_f1"] > 0.9 and m["g/teacher_vs_reference_ceiling/micro_f1"] == 1.0
    assert m["g/heads"] == 2 and (tmp / "packs" / "g" / "classifier" / "linear_teacher" / "heads.npz").is_file()
    assert m["g/C"] in cfg["presence"]["linear"]["C_grid"]  # chosen on the fit half
    rows = list(__import__("csv").DictReader(open(tmp / "tables" / "presence_heads_g_teacher.csv")))
    assert [r["type_id"] for r in rows] == ["alpha", "beta"] and rows[0]["trained_on"] == "teacher"


def test_prototypes_stage(pack, monkeypatch):
    import huggingface_hub.constants

    from experiments import distil

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    cfg, tmp = pack
    cfg["run"]["device"] = "cpu"
    cfg["distil"]["stage"] = "prototypes"
    m = distil.main(cfg, tmp)
    proto = json.loads((tmp / "packs" / "g" / "prototypes" / "prototypes.json").read_text())
    assert m["g/types_without_prototypes"] == 0 and 1 <= len(proto["prototypes"]["alpha"]) <= 30
    assert all(p["doc_id"] < "d120" for t in proto["prototypes"].values() for p in t)  # training documents only
    z = np.load(tmp / "packs" / "g" / "prototypes" / "tokens.npz")
    assert z["lengths"].sum() == len(z["tokens"]) and len(z["lengths"]) == m["g/prototypes"]


def test_cascade_stage(pack, monkeypatch):
    import huggingface_hub.constants

    import engine.embed as E
    import engine.presence as P
    from experiments import distil

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)

    class FakeReranker:
        def __init__(self, *a, **k):
            pass

        def score(self, pairs):  # says "stated" when the type's word is in the section
            return np.array([float(q.split()[1].rstrip(".").lower() in t.lower()) for q, t in pairs], dtype=np.float32)

    class FakeEmbedder:
        def __init__(self, *a, **k):
            pass

        def encode(self, texts, **k):
            return np.zeros((len(texts), 8), dtype=np.float32)

    monkeypatch.setattr(P, "Reranker", FakeReranker)
    monkeypatch.setattr(E, "Embedder", FakeEmbedder)
    cfg, tmp = pack
    cfg["run"]["device"] = "cpu"
    cfg["paths"]["figures"] = str(tmp / "figures")
    cfg["presence"]["cascade"]["reranker"] = True
    for stage in ("heads", "prototypes", "cascade"):
        cfg["distil"]["stage"] = stage
        m = distil.main(cfg, tmp)
    rows = {r["variant"]: r for r in __import__("csv").DictReader(open(tmp / "tables" / "presence_ablation_g.csv"))}
    assert set(rows) == {"classifier only", "prototypes only", "prototypes + reranker", "classifier + prototypes",
                         "full cascade", "teacher labels (ceiling)"}
    assert float(rows["full cascade"]["micro_f1"]) > 0.8 and m["g/band_low"] <= m["g/band_high"]
    assert m["g/presence_model"] == "full cascade" and (tmp / "packs" / "g" / "cascade_full.json").is_file()
    assert (tmp / "packs" / "g" / "cascade.json").is_file() and (tmp / "figures" / "presence_band_g.png").is_file()

    cfg["presence"]["cascade"]["reranker"] = False  # the lean setting: no reranker is loaded or called
    monkeypatch.setattr(P, "Reranker", None)
    m = distil.main(cfg, tmp)
    rows = {r["variant"] for r in __import__("csv").DictReader(open(tmp / "tables" / "presence_ablation_g.csv"))}
    assert rows == {"classifier only", "prototypes only", "classifier + prototypes", "teacher labels (ceiling)"}
    assert m["g/presence_model"] == "classifier + prototypes"


def test_profiles_stage(pack, monkeypatch):
    import huggingface_hub.constants

    import engine.embed as E
    from experiments import distil
    from packs.profiles import read_profiles

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(E, "Embedder", lambda *a, **k: type("F", (), {"encode": lambda self, t, **kw: np.zeros((len(t), 8))})())
    cfg, tmp = pack
    cfg["run"]["device"] = "cpu"
    cfg["paths"]["figures"] = str(tmp / "figures")
    cfg["distil"]["profiles"] = {"splits": ["train"], "chunk_documents": 50, "limit": None}
    for stage in ("heads", "prototypes", "cascade", "profiles"):
        cfg["distil"]["stage"] = stage
        m = distil.main(cfg, tmp)
    df = read_profiles(tmp / "packs" / "g" / "profiles")
    assert len(df) == 120 and m["g/parts"] == 3 and set(df["split"]) == {"train"}
    assert 0.2 < df["bit_alpha"].mean() < 0.6  # about 40% of documents state alpha
    again = distil.main(cfg, tmp)
    assert again["g/parts_written"] == 0  # resumed: nothing recomputed

    (tmp / "prof_run").mkdir()
    (tmp / "prof_run" / "metrics.json").write_text(json.dumps({"metrics": {"g/sections_per_second": 50.0, "machine": "m"}}))
    cfg["distil"]["stage"], cfg["distil"]["gate"] = "gate", {"profiles_run": str(tmp / "prof_run")}
    gate = distil.main(cfg, tmp)
    assert gate["g/profiles_complete"] and gate["g/profiled_documents"] == 120 and gate["g/sections_per_second"] == 50.0
    assert gate["g/f1_gate"] in ("pass", "not met") and (tmp / "tables" / "presence_gate.csv").is_file()


def test_profiles_written_elsewhere_keep_their_table_out_of_reports(tmp_path):
    # a limited run (smoke, Colab check) writing profiles outside the pack must not overwrite the corpus table
    from experiments.distil import profile_table_path

    cfg = {"paths": {"tables": str(tmp_path / "reports" / "tables")}}
    assert profile_table_path(cfg, {}, "papers") == tmp_path / "reports" / "tables" / "profile_prevalence_papers.csv"
    out = tmp_path / "runs" / "x_out" / "papers"
    assert profile_table_path(cfg, {"out": str(tmp_path / "runs" / "x_out" / "{genre}")}, "papers") == \
        out / "profile_prevalence_papers.csv"


def test_section_probs_stage_matches_the_profiles(pack, monkeypatch):
    import huggingface_hub.constants

    import engine.embed as E
    from experiments import distil
    from packs.profiles import read_profiles

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(E, "Embedder", lambda *a, **k: type("F", (), {"encode": lambda self, t, **kw: np.zeros((len(t), 8))})())
    cfg, tmp = pack
    cfg["run"]["device"] = "cpu"
    cfg["paths"]["figures"] = str(tmp / "figures")
    cfg["distil"]["profiles"] = {"splits": ["train"], "chunk_documents": 50, "limit": None}
    cfg["distil"]["section_probs"] = {"splits": ["train", "dev"], "chunk_documents": 50, "limit": None, "out": None}
    for stage in ("heads", "prototypes", "cascade", "profiles", "section_probs"):
        cfg["distil"]["stage"] = stage
        m = distil.main(cfg, tmp)
    sp = read_profiles(tmp / "packs" / "g" / "section_probs")
    assert len(sp) == 160 and set(sp["split"]) == {"train", "dev"} and m["g/sections_scored"] == 160
    prof = read_profiles(tmp / "packs" / "g" / "profiles").set_index("doc_id")
    best = sp[sp.split == "train"].groupby("doc_id")["prob_alpha"].max()
    assert np.allclose(best.loc[prof.index].to_numpy(), prof["prob_alpha"].to_numpy(), atol=1e-6)
