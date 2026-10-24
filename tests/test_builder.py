"""packs/builder.py (plan 5.7): pack stages; first the document index (5.2/5.3) from the stored embeddings."""

import json

import numpy as np

from data.schema import Document, Section, write_jsonl


def make_pack(tmp_path, cfg):
    from engine.embed import EmbeddingStore

    processed = tmp_path / "processed" / "g"
    processed.mkdir(parents=True)
    split = {f"d{i}": ("train" if i < 6 else "dev" if i < 8 else "test") for i in range(10)}
    docs = [Document(id=d, genre="g", source="s", entity_id=f"e{int(d[1:]) // 2}", date=f"20{10 + int(d[1:])}-01-01",
                     category="c" if int(d[1:]) % 2 else None, title=f"Title {d}") for d in split]
    secs = [Section(doc_id=d, idx=j, heading="H", text=f"{d} section {j}") for d in split for j in range(2)]
    write_jsonl(processed / "documents.jsonl", docs)
    write_jsonl(processed / "sections.jsonl", secs)
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "g.json").write_text(json.dumps({"doc_ids": {s: [d for d in split if split[d] == s]
                                                                       for s in ("train", "dev", "test")}}))
    rng = np.random.default_rng(0)
    emb = tmp_path / "emb" / cfg["models"]["embedder"]["id"].replace("/", "__") / "g"
    EmbeddingStore.save(emb / "sections", [[s.doc_id, s.idx] for s in secs], [s.text for s in secs],
                        rng.normal(size=(len(secs), 8)).astype(np.float32), {"model": cfg["models"]["embedder"]["id"]})
    EmbeddingStore.save(emb / "titles", [d.id for d in docs], [d.title for d in docs],
                        rng.normal(size=(len(docs), 8)).astype(np.float32), {"model": cfg["models"]["embedder"]["id"]})
    cfg["pack"] = "g"
    cfg["paths"].update(packs=str(tmp_path / "packs"), data_processed=str(tmp_path / "processed"),
                        splits=str(tmp_path / "splits"), embeddings=str(tmp_path / "emb"))
    return secs, docs


def test_index_stage_builds_document_vectors_for_the_indexed_split(tmp_path):
    from engine.embed import EmbeddingStore, document_vector
    from engine.index import load_index
    from experiments.config import load_config
    from packs.builder import build_index

    cfg = load_config("configs/base.yaml", ["smoke"])
    secs, docs = make_pack(tmp_path, cfg)
    m = build_index(cfg, "g")
    idx = load_index(tmp_path / "packs" / "g" / "index")
    assert len(idx) == 6 == m["documents"] and set(idx.ids) == {f"d{i}" for i in range(6)}  # train only
    p = idx.payloads[idx.pos["d3"]]
    assert p == {"genre": "g", "category": "c", "date": "2013-01-01", "entity_id": "e1", "split": "train", "source": "s"}
    assert idx.payloads[idx.pos["d2"]]["category"] is None
    emb = tmp_path / "emb" / cfg["models"]["embedder"]["id"].replace("/", "__") / "g"
    S, T = EmbeddingStore.load(emb / "sections"), EmbeddingStore.load(emb / "titles")
    rows = [i for i, r in enumerate(S.ids) if r[0] == "d3"]
    expect = document_vector(S.vectors[rows], T.vectors[T.ids.index("d3")], title_abstract_weight=cfg["models"]["embedder"]["title_abstract_weight"])
    assert np.allclose(idx.X[idx.pos["d3"]], expect, atol=1e-5)
    assert idx.meta["embedder"] == cfg["models"]["embedder"]["id"] and idx.meta["splits"] == ["train"]


def test_build_pack_runs_index_and_cohort_report(tmp_path):
    from experiments import build_pack
    from experiments.config import load_config

    cfg = load_config("configs/base.yaml", ["smoke"])
    make_pack(tmp_path, cfg)
    cfg["paths"].update(figures=str(tmp_path / "figures"), tables=str(tmp_path / "tables"))
    cfg["build"] = {"stages": ["index", "cohort_report"], "cohort_report": {"queries": 10, "k": 3}}
    m = build_pack.main(cfg, tmp_path)
    assert m["g/ran"] == ["index", "cohort_report"]
    man = json.loads((tmp_path / "packs" / "g" / "pack.json").read_text())
    assert man["stages"]["index"]["metrics"]["documents"] == 6 and man["stages"]["cohort_report"]["metrics"]["queries"] == 2
    assert man["stages"]["cohort_report"]["metrics"]["own_entity_in_cohort"] == 0 and man["models"]["embedder"]
    assert (tmp_path / "figures" / "cohort_similarity_g.png").is_file()
    assert (tmp_path / "tables" / "cohort_similarity_g.csv").is_file()
    assert build_pack.main(cfg, tmp_path)["g/skipped"] == ["index", "cohort_report"]  # nothing changed


def test_section_texts_are_read_by_offset_for_indexed_documents(tmp_path):
    from experiments.config import load_config
    from packs.builder import SectionTexts, build_index

    cfg = load_config("configs/base.yaml", ["smoke"])
    secs, _ = make_pack(tmp_path, cfg)
    build_index(cfg, "g")
    st = SectionTexts.load(tmp_path / "packs" / "g" / "index", tmp_path / "processed" / "g" / "sections.jsonl")
    assert st("d3", 1) == "d3 section 1" and st("d9", 0) is None  # test documents are not indexed
    assert st("d3", 7) is None


def test_evidence_stage_stores_the_best_section_per_document_and_type(tmp_path):
    import pandas as pd
    import yaml

    from experiments.config import load_config
    from packs.builder import build_evidence, load_best_sections

    cfg = load_config("configs/base.yaml", ["smoke"])
    make_pack(tmp_path, cfg)
    pack = tmp_path / "packs" / "g"
    (pack / "section_probs").mkdir(parents=True)
    (pack / "schema.yaml").write_text(yaml.safe_dump({"version": 1, "status": "frozen", "types": [{"id": "a"}, {"id": "b"}]}))
    pd.DataFrame({"doc_id": ["d0", "d0", "d1", "d7"], "split": ["train", "train", "train", "dev"], "section_idx": [0, 1, 0, 0],
                  "prob_a": [0.1, 0.8, 0.5, 0.9], "prob_b": [0.7, 0.2, 0.4, 0.1]}).to_parquet(pack / "section_probs" / "part-00000.parquet")
    m = build_evidence(cfg, "g")
    best = load_best_sections(pack / "index")
    assert m["documents"] == 2 and set(best) == {"d0", "d1"}  # the indexed split only
    assert best["d0"].tolist() == [1, 0] and best["d1"].tolist() == [0, 0]


# ------------------------------------------------------------------ the builder: skip unchanged stages, manifest
def toy_stages(root, calls):
    from packs.builder import Stage

    def writer(name, text):
        def run(cfg, genre, run_dir):
            calls.append(name)
            (root / f"{name}.out").write_text(text(cfg))
            return {"wrote": name}
        return run

    src = root / "src.txt"
    return [
        Stage("a", writer("a", lambda c: src.read_text().upper()), inputs=lambda c, g: [src],
              outputs=lambda c, g: [root / "a.out"], config_keys=["x"]),
        Stage("b", writer("b", lambda c: (root / "a.out").read_text() + "!"), inputs=lambda c, g: [root / "a.out"],
              outputs=lambda c, g: [root / "b.out"], config_keys=["y"]),
        Stage("gpu", writer("gpu", lambda c: "g"), inputs=lambda c, g: [root / "b.out"],
              outputs=lambda c, g: [root / "gpu.out"], config_keys=[], gpu=True, colab_job="toy_gpu_job"),
    ]


def test_builder_creates_the_pack_folder_of_a_pack_built_from_nothing(tmp_path):
    from packs.builder import build

    calls = []
    (tmp_path / "src.txt").write_text("hello")
    cfg = {"x": 1, "y": 2, "tier": "lean", "models": {}, "run": {"device": "cuda"}}
    man = tmp_path / "packs" / "g" / "pack.json"  # no packs/ folder yet: the fresh-clone smoke build (7.6)
    build(cfg, "g", toy_stages(tmp_path, calls)[:1], manifest_path=man, run_dir=tmp_path)
    assert calls == ["a"] and json.loads(man.read_text())["stages"]["a"]["adopted"] is False


def test_builder_skips_unchanged_stages_and_reruns_what_changed(tmp_path):
    import pytest

    from packs.builder import GpuStageNeeded, build

    calls = []
    (tmp_path / "src.txt").write_text("hello")
    cfg = {"x": 1, "y": 2, "tier": "lean", "models": {"embedder": {"id": "emb"}}, "run": {"device": "cuda"}}
    stages = toy_stages(tmp_path, calls)
    man = tmp_path / "pack.json"
    build(cfg, "g", stages, manifest_path=man, run_dir=tmp_path)
    assert calls == ["a", "b", "gpu"]
    calls.clear()
    m = build(cfg, "g", stages, manifest_path=man, run_dir=tmp_path)
    assert calls == [] and m["skipped"] == ["a", "b", "gpu"]
    cfg["y"] = 3  # a setting of b changed: b reruns; gpu's input (b.out) is unchanged in content, so it stays
    build(cfg, "g", stages, manifest_path=man, run_dir=tmp_path)
    assert calls == ["b"]
    calls.clear()
    (tmp_path / "src.txt").write_text("changed")  # a's input changed: a, then b (its input changed) and gpu
    build(cfg, "g", stages, manifest_path=man, run_dir=tmp_path)
    assert calls == ["a", "b", "gpu"]
    info = json.loads(man.read_text())
    assert info["tier"] == "lean" and info["models"]["embedder"] == "emb" and set(info["stages"]) == {"a", "b", "gpu"}
    assert all(len(info["stages"][s]["outputs"]) == 1 for s in info["stages"])

    calls.clear()
    cfg["run"]["device"] = "mps"
    (tmp_path / "src.txt").write_text("again")
    with pytest.raises(GpuStageNeeded, match="toy_gpu_job"):  # a GPU stage never runs silently on the Mac
        build(cfg, "g", stages, manifest_path=man, run_dir=tmp_path)
    assert calls == ["a", "b"]


def test_builder_adopts_existing_outputs_without_rerunning(tmp_path):
    from packs.builder import build

    calls = []
    for name, text in (("src.txt", "hello"), ("a.out", "HELLO"), ("b.out", "HELLO!"), ("gpu.out", "g")):
        (tmp_path / name).write_text(text)
    cfg = {"x": 1, "y": 2, "run": {"device": "mps"}, "models": {}}
    m = build(cfg, "g", toy_stages(tmp_path, calls), manifest_path=tmp_path / "pack.json", run_dir=tmp_path, adopt=True)
    assert calls == [] and m["adopted"] == ["a", "b", "gpu"]
    assert json.loads((tmp_path / "pack.json").read_text())["stages"]["a"]["adopted"] is True
    (tmp_path / "pack.json").unlink()
    m = build(cfg, "g", toy_stages(tmp_path, calls), manifest_path=tmp_path / "pack.json", run_dir=tmp_path, adopt=True,
              force=["b"])
    assert calls == ["b"] and m["adopted"] == ["a", "gpu"]  # a forced stage is rebuilt, never adopted


def test_fingerprints_inside_the_repository_are_relative_paths():
    from packs.builder import REPO, fingerprint

    fp = fingerprint([REPO / "configs" / "base.yaml"])
    assert list(fp) == ["configs/base.yaml"]  # the same key on the Mac and on Colab
