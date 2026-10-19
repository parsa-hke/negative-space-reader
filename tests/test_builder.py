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
    assert m["g/index/documents"] == 6 and m["g/cohort_report/queries"] == 2  # the two dev documents
    assert m["g/cohort_report/own_entity_in_cohort"] == 0 and 0 <= m["g/cohort_report/fallback_rate"] <= 1
    assert (tmp_path / "figures" / "cohort_similarity_g.png").is_file()
    assert (tmp_path / "tables" / "cohort_similarity_g.csv").is_file()


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
