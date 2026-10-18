"""engine/embed.py store and experiments/embed.py wiring (plan §5.1). The model itself is exercised in the lab."""

import numpy as np

from engine.embed import EmbeddingStore, text_sha1


def test_store_round_trip_and_lookup(tmp_path):
    texts = ["first section", "second section", "third"]
    vecs = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    EmbeddingStore.save(tmp_path / "s", [["d", 0], ["d", 1], ["e", 0]], texts, vecs, {"model": "m"})
    st = EmbeddingStore.load(tmp_path / "s")
    assert st.ids == [["d", 0], ["d", 1], ["e", 0]] and st.sha1 == [text_sha1(t) for t in texts]
    assert st.meta["rows"] == 3 and st.meta["dim"] == 8 and st.meta["model"] == "m"
    assert np.allclose(st.vectors, vecs, atol=1e-3)
    hit, miss = st.lookup(["second section", "never seen"])
    assert np.allclose(hit, vecs[1], atol=1e-3) and miss is None


def test_embed_task_is_registered_and_lab_config_resolves():
    from experiments.config import load_config
    from experiments.run import TASKS, resolve_task

    assert TASKS["embed"] == "experiments.embed:main" and callable(resolve_task("embed"))
    cfg = load_config("configs/embed.yaml", ["lab"])
    assert cfg["task"] == "embed" and cfg["run"]["devices"] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert cfg["models"]["embedder"]["id"] == "Qwen/Qwen3-Embedding-0.6B"
    t = load_config("configs/induce.yaml", ["lab"])["models"]["teacher"]
    assert t["backend"] == "openai" and len(t["base_url"]) == 4 and t["concurrency"] == 32
    assert "Qwen2.5-32B" in t["served_as"]


class CountingEmbedder:
    """Stands in for Embedder: deterministic unit vectors, records which texts it was asked to encode."""
    model_id, devices, normalize, truncate_dim, dim = "fake", ["cpu"], True, None, 8

    def __init__(self):
        self.seen = []

    def encode(self, texts, show_progress=False):
        self.seen += list(texts)
        rows = [np.random.default_rng(int(text_sha1(t)[:8], 16)).normal(size=self.dim) for t in texts]
        out = np.asarray(rows, dtype=np.float32).reshape(len(texts), self.dim)
        return out / np.linalg.norm(out, axis=1, keepdims=True) if len(texts) else out


def test_embed_reuses_stored_rows_by_text_hash(tmp_path):
    """After a rebuild that split some sections, only the new texts are encoded; reused rows keep their vectors."""
    from engine.embed import embed_and_store

    first = CountingEmbedder()
    embed_and_store(first, [["d", 0], ["d", 1]], ["kept section", "long section"], tmp_path / "s")
    before = EmbeddingStore.load(tmp_path / "s")
    second = CountingEmbedder()
    stats = embed_and_store(second, [["d", 0], ["d", 1], ["d", 2]], ["kept section", "long", "section"],
                            tmp_path / "s", reuse=True)
    assert second.seen == ["long", "section"] and stats["reused_rows"] == 1 and stats["new_rows"] == 2
    after = EmbeddingStore.load(tmp_path / "s")
    assert after.ids == [["d", 0], ["d", 1], ["d", 2]] and after.meta["reused_rows"] == 1
    assert np.allclose(after.vectors[0], before.vectors[0]) and after.lookup(["long section"]) == [None]
    third = CountingEmbedder()
    embed_and_store(third, [["d", 0]], ["kept section"], tmp_path / "s")  # reuse off: everything is encoded
    assert third.seen == ["kept section"]


def _embed_cfg(tmp_path, mode, **embed):
    from data.schema import Document, Section, write_jsonl
    from experiments.config import load_config

    processed = tmp_path / "processed" / "g"
    processed.mkdir(parents=True, exist_ok=True)
    texts = ["kept section", "new chunk one", "new chunk two", "new chunk one"]  # a repeated text is sent once
    write_jsonl(processed / "sections.jsonl", [Section(doc_id="d", idx=i, heading=None, text=t) for i, t in enumerate(texts)])
    write_jsonl(processed / "documents.jsonl", [Document(id="d", genre="g", source="s", entity_id="d", version=None,
                                                         date=None, category=None, title="kept title", meta={})])
    cfg = load_config("configs/embed.yaml", ["smoke"])
    cfg["pack"] = "g"
    cfg["paths"].update(data_processed=str(tmp_path / "processed"), embeddings=str(tmp_path / "emb"))
    cfg["embed"].update(mode=mode, **embed)
    return cfg, texts


def test_export_colab_import_round_trip(tmp_path):
    import json
    import zipfile

    from engine.embed import embed_and_store, embedding_slug
    from experiments import colab_embed, embed
    from experiments.config import load_config

    cfg, texts = _embed_cfg(tmp_path, "export")
    store = tmp_path / "emb" / embedding_slug(cfg["models"]["embedder"]) / "g"
    first = CountingEmbedder()
    embed_and_store(first, [["d", 0]], ["kept section"], store / "sections")
    embed_and_store(first, ["d"], ["kept title"], store / "titles")
    (tmp_path / "run").mkdir()
    m = embed.main(cfg, tmp_path / "run")
    assert m["g/sections_missing"] == 2 and m["g/titles_missing"] == 0
    with zipfile.ZipFile(tmp_path / "run" / "colab_bundle.zip") as z:
        z.extractall(tmp_path / "bundle")
        assert "colab_embed.py" in z.namelist()
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert list(manifest["files"]) == ["g_sections"] and manifest["files"]["g_sections"]["rows"] == 2

    colab = CountingEmbedder()
    colab_embed.encode_bundle(tmp_path / "bundle", tmp_path / "colab_out", colab.encode, {"device": "fake"})
    assert colab.seen == ["new chunk one", "new chunk two"]

    cfg["embed"].update(mode="import", import_from=str(tmp_path / "colab_out"), import_check=0)
    m = embed.main(cfg, tmp_path / "run")
    st = EmbeddingStore.load(store / "sections")
    assert st.ids == [["d", 0], ["d", 1], ["d", 2], ["d", 3]] and m["g/sections_imported_rows"] == 3
    assert np.allclose(st.vectors[1], st.vectors[3]) and st.meta["imported_from"]["device"] == "fake"
    assert np.allclose(st.vectors[2], CountingEmbedder().encode(["new chunk two"])[0], atol=1e-3)


def test_import_refuses_when_texts_are_still_missing(tmp_path):
    import pytest

    from experiments import embed

    cfg, _ = _embed_cfg(tmp_path, "import", import_from=str(tmp_path / "empty"), import_check=0)
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "meta.json").write_text('{"files": {}}')
    with pytest.raises(ValueError, match="no vector"):
        embed.main(cfg, tmp_path)


def test_colab_model_settings_match_the_embedder(monkeypatch):
    """The Colab script must encode exactly like engine/embed.Embedder (checked with a small model on CPU)."""
    import huggingface_hub.constants

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # cached model; network checks made this test take 11 s
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    from engine.embed import Embedder
    from experiments.colab_embed import load_model

    texts = ["We keep your data for two years.", "The authors report accuracy on three datasets.", "x" * 600]
    ref = Embedder("BAAI/bge-small-en-v1.5", devices=["cpu"], max_length=64).encode(texts)
    model = load_model({"model": "BAAI/bge-small-en-v1.5", "max_length": 64, "truncate_dim": None}, "cpu")
    got = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    assert np.allclose(ref, got, atol=1e-5)
