"""Presence labels from human annotations through each type's label_source (plan §4.1, genre A without a teacher)."""

import pytest

from data.schema import Section
from packs.presence_labels import LabelSourceError, annotation_labels, parse_label_source, teacher_types

TYPES = [
    {"id": "retention", "label_source": ["Data Retention"]},
    {"id": "register", "label_source": ["First Party :: Mode = Explicit & User = With account"]},
    {"id": "cookies_or_ip", "label_source": ["First Party :: Info = Cookies", "First Party :: Info = IP"]},
    {"id": "web_beacons", "label_source": ["teacher"]},
]
ANN = [
    {"doc_id": "a", "section_idx": 0, "category": "Data Retention", "attributes": {"Period": "Stated"}, "annotation_id": "1"},
    {"doc_id": "a", "section_idx": 1, "category": "First Party", "attributes": {"Mode": "Explicit", "User": "With account",
                                                                             "Info": "Contact"}, "annotation_id": "2"},
    # the two conditions hold only across two different annotations of the same section: not a positive
    {"doc_id": "a", "section_idx": 2, "category": "First Party", "attributes": {"Mode": "Explicit", "User": "Unspecified",
                                                                             "Info": "IP"}, "annotation_id": "3"},
    {"doc_id": "a", "section_idx": 2, "category": "First Party", "attributes": {"Mode": "Implicit", "User": "With account",
                                                                             "Info": "Cookies"}, "annotation_id": "4"},
]
SECTIONS = {"a": [Section(doc_id="a", idx=i, heading=None, text=f"s{i}") for i in range(4)],
            "b": [Section(doc_id="b", idx=0, heading=None, text="unannotated document")]}


def test_parse_label_source():
    assert parse_label_source("Data Retention") == ("Data Retention", [])
    assert parse_label_source("First Party :: Mode = Explicit & User = With account") == \
        ("First Party", [("Mode", "Explicit"), ("User", "With account")])
    with pytest.raises(LabelSourceError):
        parse_label_source("First Party :: Mode Explicit")


def test_teacher_types():
    assert teacher_types(TYPES) == ["web_beacons"]


def test_annotation_labels_every_section_of_annotated_documents():
    rows, counts = annotation_labels(TYPES, ANN, SECTIONS, source="opp115")
    assert [(r["doc_id"], r["section_idx"]) for r in rows] == [("a", 0), ("a", 1), ("a", 2), ("a", 3)]  # b: no labels
    lab = {r["section_idx"]: r["labels"] for r in rows}
    assert lab[0] == {"retention": 1, "register": 0, "cookies_or_ip": 0}  # teacher types are left out
    assert lab[1] == {"retention": 0, "register": 1, "cookies_or_ip": 0}
    assert lab[2] == {"retention": 0, "register": 0, "cookies_or_ip": 1}
    assert lab[3] == {"retention": 0, "register": 0, "cookies_or_ip": 0}  # no annotation: negative for every type
    assert rows[2]["evidence"] == {"cookies_or_ip": ["3", "4"]} and rows[0]["source"] == "opp115"
    assert counts["positives"] == {"retention": 1, "register": 1, "cookies_or_ip": 1}
    assert counts["sections"] == 4 and counts["sections_without_annotation"] == 1


@pytest.mark.parametrize("bad", ["Data Retention :: Period = Forever", "Data Retenton", "First Party :: Moode = Explicit"])
def test_unknown_names_are_refused(bad):
    """Values must be the annotation file's exact names; a typo would silently produce zero positives."""
    with pytest.raises(LabelSourceError, match="not in the annotations"):
        annotation_labels([{"id": "t", "label_source": [bad]}], ANN, SECTIONS, source="opp115")


def test_labels_stage_end_to_end(tmp_path):
    import csv
    import json

    import yaml

    from data.schema import write_jsonl
    from experiments import distil
    from experiments.config import load_config

    (tmp_path / "packs" / "g").mkdir(parents=True)
    (tmp_path / "packs" / "g" / "schema.yaml").write_text(yaml.safe_dump({"version": 1, "status": "frozen", "types": TYPES}))
    processed = tmp_path / "processed" / "g"
    processed.mkdir(parents=True)
    write_jsonl(processed / "sections.jsonl", SECTIONS["a"] + SECTIONS["b"])
    with open(processed / "ann_labels.jsonl", "w") as f:
        f.writelines(json.dumps(a) + "\n" for a in ANN)
    cfg = load_config("configs/distil.yaml", ["smoke"])
    cfg["pack"] = "g"
    cfg["paths"].update(packs=str(tmp_path / "packs"), data_processed=str(tmp_path / "processed"),
                        tables=str(tmp_path / "tables"))
    cfg["presence"]["label_sources"]["g"] = "ann"
    cfg["distil"]["stage"] = "labels"
    metrics = distil.main(cfg, tmp_path)
    assert metrics["g/sections"] == 4 and metrics["g/teacher_types"] == 1 and metrics["g/types_under_20_positives"] == 3
    assert json.loads((tmp_path / "teacher_types_g.json").read_text()) == ["web_beacons"]
    rows = list(csv.DictReader(open(tmp_path / "tables" / "presence_labels_g.csv")))
    assert [r["label_source"] for r in rows] == ["ann", "ann", "ann", "teacher"] and rows[0]["positives"] == "1"


# ------------------------------------------------------------------------------------ teacher labels (two passes)
import json as _json

import numpy as np

from models.teacher import Teacher
from packs.presence_labels import sample_sections, select_candidates, teacher_labels

LTYPES = [{"id": "alpha", "definition": "States alpha.", "positive_examples": ["alpha one", "alpha two"]},
          {"id": "beta", "definition": "States beta.", "positive_examples": ["beta one"]},
          {"id": "gamma", "definition": "States gamma.", "positive_examples": ["gamma one"]},
          {"id": "delta", "definition": "States delta.", "positive_examples": ["delta one"]}]
GENRE = {"document_kind": "test document", "author_role": "the writer"}


class KeywordEmbedder:
    """Vectors over the four type words, so similarity is predictable."""
    words = ["alpha", "beta", "gamma", "delta"]

    def encode(self, texts, **kw):
        X = np.array([[t.lower().count(w) for w in self.words] for t in texts], dtype=np.float32) + 0.01
        return X / np.linalg.norm(X, axis=1, keepdims=True)


class LabelTeacher(Teacher):
    """First pass: maps the alpha sentence to alpha and the beta sentence to none. Check: says beta (a candidate)
    and gamma (not a candidate, must be ignored)."""
    backend = "fake"

    def __init__(self, fail_on=None):
        super().__init__("fake-model", max_retries=0)
        self.prompts, self.fail_on = [], fail_on

    def _complete(self, conversations, *, json_schema, max_tokens):
        outs = []
        for conv in conversations:
            p = conv[-1]["content"]
            self.prompts.append(p)
            if self.fail_on and self.fail_on in p:
                outs.append("not json")
            elif '{"statements"' in p:
                outs.append(_json.dumps({"statements": [
                    {"claim": "It states alpha.", "type": "alpha", "span": "Alpha is stated here."},
                    {"claim": "It mentions beta.", "type": "none", "span": "Beta is mentioned too."},
                    {"claim": "Delta, loosely.", "type": "delta", "span": "words that are not in the section"}]}))
            else:
                outs.append(_json.dumps({"present": [{"type": "beta", "span": "Beta is mentioned too."},
                                                     {"type": "gamma", "span": "Alpha is stated here."}]}))
            self._served()
        return outs


def test_select_candidates_excludes_found_and_caps():
    T = np.eye(4, dtype=np.float32)
    claims = np.array([[0.9, 0.5, 0.1, 0.0], [0.0, 0.2, 0.0, 0.8]], dtype=np.float32)
    section = np.array([0.1, 0.2, 0.9, 0.3], dtype=np.float32)
    # per claim: its k most similar types not yet found; then the section's; in that order, without repeats
    assert select_candidates(claims, section, T, found={0}, k_claim=2, k_section=1, cap=8) == [1, 2, 3]
    assert select_candidates(claims, section, T, found={0}, k_claim=2, k_section=1, cap=2) == [1, 2]
    assert select_candidates(np.zeros((0, 4)), None, T, found=set(), k_claim=2, k_section=2, cap=8) == []


def test_teacher_labels_two_passes():
    secs = [Section(doc_id="d", idx=0, heading="H", text="Alpha is stated here. Beta is mentioned too.")]
    teacher = LabelTeacher()
    rows = teacher_labels(teacher, KeywordEmbedder(), secs, GENRE, LTYPES, label_types=["alpha", "beta", "gamma", "delta"],
                          max_assertions=8, max_tokens=500, check_max_tokens=200, k_claim=1, k_section=1, cap=4)
    r = rows[0]
    assert r["labels"] == {"alpha": 1, "beta": 1, "gamma": 0, "delta": 1}
    assert r["found_by"] == {"alpha": "first", "beta": "second_look", "delta": "first"}
    assert r["grounded"] == {"alpha": True, "beta": True, "delta": False}
    assert r["evidence"]["beta"] == "Beta is mentioned too."
    assert "gamma" not in r["candidates"] and "beta" in r["candidates"] and not r["failed"]
    first_prompt = teacher.prompts[0]
    assert all(t["id"] in first_prompt for t in LTYPES) and "Alpha is stated here." in first_prompt


def test_teacher_labels_subset_and_failures():
    secs = [Section(doc_id="d", idx=0, heading="H", text="Alpha is stated here. Beta is mentioned too."),
            Section(doc_id="e", idx=3, heading="H", text="FAIL this section entirely.")]
    rows = teacher_labels(LabelTeacher(fail_on="FAIL"), KeywordEmbedder(), secs, GENRE, LTYPES, label_types=["beta"],
                          max_assertions=8, max_tokens=500, check_max_tokens=200, k_claim=1, k_section=1, cap=4)
    assert rows[0]["labels"] == {"beta": 1} and rows[1]["failed"] and rows[1]["labels"] == {}


def test_sample_sections_is_stratified_deterministic_and_spread_over_documents():
    pool = {f"d{i}": [Section(doc_id=f"d{i}", idx=j, heading=None, text="x" * (100 if j % 2 else 3000))
                      for j in range(10)] for i in range(40)}
    a = sample_sections(pool, n=30, seed=1, min_chars=80, per_doc=1)
    b = sample_sections(pool, n=30, seed=1, min_chars=80, per_doc=1)
    assert [(s.doc_id, s.idx) for s in a] == [(s.doc_id, s.idx) for s in b] and len(a) == 30
    assert len({s.doc_id for s in a}) == 30
    short = sum(len(s.text) < 600 for s in a)
    assert 12 <= short <= 18  # proportional: half the pool is short
    assert sample_sections(pool, n=30, seed=2, min_chars=80, per_doc=1) != a
    assert all(len(s.text) >= 200 for s in sample_sections(pool, n=10, seed=1, min_chars=200, per_doc=2))


def test_select_and_teacher_label_stages(tmp_path, monkeypatch):
    import yaml

    import engine.embed as E
    import models.teacher as M
    from data.schema import write_jsonl
    from experiments import distil
    from experiments.config import load_config

    (tmp_path / "packs" / "g").mkdir(parents=True)
    (tmp_path / "packs" / "g" / "schema.yaml").write_text(yaml.safe_dump({"version": 1, "status": "frozen", "types": LTYPES}))
    processed = tmp_path / "processed" / "g"
    processed.mkdir(parents=True)
    split = {f"d{i}": ("train" if i < 30 else "dev" if i < 45 else "test") for i in range(50)}
    sections = [Section(doc_id=d, idx=j, heading="H", text=f"Alpha is stated here. Beta is mentioned too. {d} {j} " * 4)
                for d in split for j in range(4)]
    write_jsonl(processed / "sections.jsonl", sections)
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "g.json").write_text(_json.dumps({"doc_ids": {s: [d for d in split if split[d] == s]
                                                                          for s in ("train", "dev", "test")}}))
    (tmp_path / "sample.json").write_text(_json.dumps(["d30", "d31"]))
    cfg = load_config("configs/distil.yaml", ["smoke"])
    cfg["pack"] = "g"
    cfg["paths"].update(packs=str(tmp_path / "packs"), data_processed=str(tmp_path / "processed"),
                        splits=str(tmp_path / "splits"), embeddings=str(tmp_path / "emb"))
    cfg["presence"]["label_sources"]["g"] = "teacher"
    cfg["genre_info"]["g"] = GENRE
    cfg["distil"].update(stage="select", human_set_per_genre=10, label_sections_per_genre=20)
    cfg["distil"]["label_sets"]["exclude_sample"] = str(tmp_path / "sample.json")
    m = distil.main(cfg, tmp_path)
    sets = _json.loads((tmp_path / "packs" / "g" / "label_sets.json").read_text())
    human, teacher = {r["doc_id"] for r in sets["human"]}, {r["doc_id"] for r in sets["teacher"]}
    assert m["g/human_sections"] == 10 and m["g/teacher_sections"] == 20 and len(human) == 10 and len(teacher) == 20
    assert all(split[d] == "dev" for d in human) and all(split[d] == "train" for d in teacher)
    assert not human & {"d30", "d31"}
    with pytest.raises(FileExistsError):
        distil.main(cfg, tmp_path)  # fixed once chosen

    monkeypatch.setattr(M, "load_teacher", lambda *a, **k: LabelTeacher())
    monkeypatch.setattr(E, "Embedder", lambda *a, **k: KeywordEmbedder())
    cfg["distil"]["stage"] = "teacher_labels"
    cfg["distil"]["teacher_labels"]["limit"] = 3
    cfg["run"]["seed"] = 0
    m = distil.main(cfg, tmp_path)
    rows = [_json.loads(line) for line in open(tmp_path / "teacher_labels_g.jsonl")]
    assert len(rows) == 6 and {r["set"] for r in rows} == {"teacher", "human"}
    # default k_section (4) makes gamma a candidate too, so the second look adds beta and gamma
    assert rows[0]["labels"] == {"alpha": 1, "beta": 1, "gamma": 1, "delta": 1} and rows[0]["source"].startswith("teacher:")
    assert m["g/human/second_look_labels"] == 6 and m["g/teacher/failed"] == 0

    changed = [Section(doc_id=s.doc_id, idx=s.idx, heading="H", text=s.text + " edited") for s in sections]
    write_jsonl(processed / "sections.jsonl", changed)
    with pytest.raises(ValueError, match="changed since it was selected"):
        distil.main(cfg, tmp_path)


def test_agreement_with_and_without_second_look():
    from packs.presence_labels import agreement

    human = {("d", 0): {"present": ["a"], "unsure": []}, ("d", 1): {"present": ["a", "b"], "unsure": []},
             ("d", 2): {"present": [], "unsure": ["b"]}}
    teacher = {("d", 0): {"labels": {"a": 1, "b": 1}, "found_by": {"a": "first", "b": "second_look"}},
               ("d", 1): {"labels": {"a": 1, "b": 1}, "found_by": {"a": "first", "b": "second_look"}},
               ("d", 2): {"labels": {"a": 0, "b": 1}, "found_by": {"b": "first"}}}
    rows = {r["type_id"]: r for r in agreement(human, teacher, ["a", "b"])}
    a, b = rows["a"], rows["b"]
    assert (a["tp"], a["fp"], a["fn"], a["f1"]) == (2, 0, 0, 1.0)
    # b: section 2 is unsure for the human, so it is left out for b; second look adds (d,0) fp and (d,1) tp
    assert (b["sections"], b["tp"], b["fp"], b["fn"]) == (2, 1, 1, 0)
    assert (b["tp_first_pass"], b["fp_first_pass"], b["fn_first_pass"]) == (0, 0, 1)
    assert b["human_positives"] == 1 and b["unsure"] == 1


def test_score_stage_against_a_labels_file(tmp_path):
    from experiments import distil
    from experiments.config import load_config

    teacher = [{"doc_id": "d", "section_idx": i, "failed": False, "labels": lab, "found_by": fb}
               for i, (lab, fb) in enumerate([({"a": 1, "b": 1}, {"a": "first", "b": "second_look"}),
                                              ({"a": 0, "b": 1}, {"b": "first"})])]
    ref = [{"doc_id": "d", "section_idx": 0, "labels": {"a": 1, "b": 0}},
           {"doc_id": "d", "section_idx": 1, "labels": {"a": 0, "b": 1}}]
    for name, rows in (("t.jsonl", teacher), ("r.jsonl", ref)):
        (tmp_path / name).write_text("".join(_json.dumps(r) + "\n" for r in rows))
    import yaml

    (tmp_path / "packs" / "g").mkdir(parents=True)
    schema = {"version": 1, "status": "frozen", "types": [{"id": "a"}, {"id": "b"}]}
    (tmp_path / "packs" / "g" / "schema.yaml").write_text(yaml.safe_dump(schema))
    cfg = load_config("configs/distil.yaml", ["smoke"])
    cfg["pack"], cfg["paths"]["tables"], cfg["paths"]["packs"] = "g", str(tmp_path / "tables"), str(tmp_path / "packs")
    cfg["distil"].update(stage="score", score={"name": "ref", "teacher": str(tmp_path / "t.jsonl"),
                                               "reference": str(tmp_path / "r.jsonl"), "min_positives": 1})
    m = distil.main(cfg, tmp_path)
    assert m["g/final/micro_precision"] == round(2 / 3, 3) and m["g/first_pass/micro_precision"] == 1.0
    assert m["g/final/micro_recall"] == 1.0 and m["g/first_pass/macro_f1"] == 1.0
    assert (tmp_path / "tables" / "teacher_vs_ref_g.csv").is_file()

    # a version-2 schema that merged a and b: the same v1 labels are scored as one type, and agree fully
    v2 = {"version": 2, "status": "frozen", "types": [{"id": "ab", "merged_from": ["a", "b"]}]}
    (tmp_path / "packs" / "g" / "schema.yaml").write_text(yaml.safe_dump(v2))
    m = distil.main(cfg, tmp_path)
    assert m["g/schema_version"] == 2 and m["g/final/micro_f1"] == 1.0


def test_reference_skips_failed_judge_sections(tmp_path):
    from experiments.distil import load_reference

    rows = [{"doc_id": "d", "section_idx": 0, "failed": False, "labels": {"a": 1, "b": 0}},
            {"doc_id": "d", "section_idx": 1, "failed": True, "labels": {}}]
    (tmp_path / "j.jsonl").write_text("".join(_json.dumps(r) + "\n" for r in rows))
    ref = load_reference(tmp_path / "j.jsonl")
    assert list(ref) == [("d", 0)] and ref[("d", 0)]["present"] == ["a"] and ref[("d", 0)]["types"] == ["a", "b"]


# ------------------------------------------------------------------ whole-document label set (document-level presence)
def test_sample_documents_is_stratified_by_year_and_category_and_deterministic():
    from packs.presence_labels import sample_documents

    docs = [{"id": f"d{i}", "date": f"{2016 + i % 4}-01-01", "category": "a" if i % 3 else "b"} for i in range(200)]
    a = sample_documents(docs, n=40, seed=7)
    assert a == sample_documents(list(reversed(docs)), n=40, seed=7) and len(a) == 40 == len(set(a))
    by = {d["id"]: d for d in docs}
    years = {by[d]["date"][:4] for d in a}
    cats = [by[d]["category"] for d in a]
    assert years == {"2016", "2017", "2018", "2019"} and 8 <= cats.count("b") <= 18  # b is a third of the pool
    assert len(sample_documents(docs, n=500, seed=7)) == 200  # never more than the pool


def test_select_documents_stage_takes_whole_corpus_documents_from_dev(tmp_path):
    import yaml

    from data.schema import Document, write_jsonl
    from experiments import distil
    from experiments.config import load_config

    (tmp_path / "packs" / "g").mkdir(parents=True)
    (tmp_path / "packs" / "g" / "schema.yaml").write_text(yaml.safe_dump({"version": 1, "status": "frozen", "types": LTYPES}))
    processed = tmp_path / "processed" / "g"
    processed.mkdir(parents=True)
    split = {f"d{i}": ("train" if i < 30 else "dev" if i < 60 else "test") for i in range(70)}
    split.update({f"a{i}": "dev" for i in range(10)})  # a source with its own segmentation, never indexed
    docs = [Document(id=d, genre="g", source="annot" if d.startswith("a") else "corp", entity_id=d,
                     date=f"{2018 + int(d[1:]) % 3}-05-01", category="x") for d in split]
    write_jsonl(processed / "documents.jsonl", docs)
    sections = [Section(doc_id=d, idx=j, heading="H", text=("short" if j == 0 else f"Alpha stated. {d} {j} " * 4))
                for d in split for j in range(1 + int(d[1:]) % 3)]
    write_jsonl(processed / "sections.jsonl", sections)
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "g.json").write_text(_json.dumps({"doc_ids": {s: [d for d in split if split[d] == s]
                                                                          for s in ("train", "dev", "test")}}))
    (tmp_path / "sample.json").write_text(_json.dumps(["d30", "d31"]))
    existing = {"genre": "g", "seed": 1, "source": "splits", "teacher": [{"doc_id": "d1", "section_idx": 0, "text_sha1": "x"}],
                "human": [{"doc_id": "d32", "section_idx": 0, "text_sha1": "y"}]}
    (tmp_path / "packs" / "g" / "label_sets.json").write_text(_json.dumps(existing))
    cfg = load_config("configs/distil.yaml", ["smoke"])
    cfg["pack"] = "g"
    cfg["paths"].update(packs=str(tmp_path / "packs"), data_processed=str(tmp_path / "processed"),
                        splits=str(tmp_path / "splits"))
    cfg["distil"].update(stage="select_documents")
    cfg["distil"]["label_sets"].update(exclude_sample=str(tmp_path / "sample.json"), documents_per_genre=12)
    m = distil.main(cfg, tmp_path)
    sets = _json.loads((tmp_path / "packs" / "g" / "label_sets.json").read_text())
    assert sets["teacher"] == existing["teacher"] and sets["human"] == existing["human"]  # untouched
    chosen = {r["doc_id"] for r in sets["documents"]}
    assert len(chosen) == 12 == m["g/documents_documents"]
    assert all(split[d] == "dev" and d.startswith("d") for d in chosen)  # dev, corpus source only
    assert not chosen & {"d30", "d31", "d32"}
    n_secs = {d: sum(s.doc_id == d for s in sections) for d in chosen}
    assert len(sets["documents"]) == sum(n_secs.values()) == m["g/documents_sections"]  # every section, short ones too
    with pytest.raises(FileExistsError):
        distil.main(cfg, tmp_path)  # fixed once chosen
