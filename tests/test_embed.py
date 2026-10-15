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
