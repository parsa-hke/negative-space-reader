"""engine/onthefly.py detect_pack (plan §0.2 #20, used by the 7.1 app's "auto" pack): the pack whose index holds the
upload's nearest neighbours (mean similarity of the top n), with the margin over the next pack; packs that share an
embedder and segmentation embed the upload once; below the threshold, no pack matches."""

import dataclasses

import numpy as np

from engine.index import make_index
from test_pipeline import DOC, pack  # noqa: F401  (the small pack fixture)


def other_pack(pack):
    X = np.tile([0.0, 0.0, 1.0, 0.0, 0.0], (30, 1))  # every document about cookies only, the upload never is
    X[:, 1] += np.linspace(0, 0.2, 30)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    idx = make_index("faiss", X.shape[1])
    idx.add([f"o{i}" for i in range(30)], X, [{"genre": "h", "entity_id": f"o{i}", "date": "2020-01-01"} for i in range(30)])
    return dataclasses.replace(pack, genre="h", index=idx)


def test_detect_pack_picks_the_pack_with_the_closest_neighbours_and_embeds_once(pack):
    from engine.onthefly import detect_pack

    calls = []
    encode = pack.embedder.encode
    pack.embedder.encode = lambda texts, **kw: calls.append(1) or encode(texts, **kw)
    packs = {"g": pack, "h": other_pack(pack)}
    d = detect_pack(DOC, packs, neighbours=5)
    assert d["pack"] == "g" and d["matched"] and d["scores"]["g"] > d["scores"]["h"]
    assert d["margin"] == round(d["scores"]["g"] - d["scores"]["h"], 4)
    assert len(calls) == 1 and set(d["embedded"]) == {"g", "h"}


def test_below_the_threshold_no_pack_matches(pack):
    from engine.onthefly import detect_pack

    d = detect_pack(DOC, {"g": pack}, neighbours=5, threshold=1.01)
    assert d["pack"] is None and not d["matched"] and d["margin"] is None


def test_the_detected_embedding_is_reused_by_prepare(pack):
    from engine.onthefly import detect_pack
    from engine.pipeline import analyse, prepare, score

    d = detect_pack(DOC, {"g": pack}, neighbours=5)
    a = score(prepare(DOC, pack, embedded=d["embedded"]["g"]), pack).to_dict()
    b = analyse(DOC, pack).to_dict()
    a.pop("timings"), b.pop("timings")
    assert a == b
