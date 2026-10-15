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
