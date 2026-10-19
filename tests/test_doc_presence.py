"""engine/doc_presence.py: document-level presence from every section's probabilities (Phase 5 first job). Synthetic
documents: a type is present when some section states it; the section model is calibrated but has a small
false-positive rate per section, so 'any section above the section threshold' inflates presence in long documents."""

import numpy as np
import pytest

from engine.doc_presence import FEATURES, DocPresence, doc_features, evaluate, split_documents


def make_docs(n_docs, T=6, seed=0):
    rng = np.random.default_rng(seed)
    docs, Y = [], []
    for _ in range(n_docs):
        n = int(rng.integers(3, 30))
        truth = rng.random((n, T)) < np.linspace(0.01, 0.12, T)  # rare to common per section
        noise = rng.beta(1, 12, size=(n, T))  # small probabilities everywhere: false positives compound
        P = np.where(truth, rng.beta(6, 2, size=(n, T)), noise)
        docs.append(P.astype(np.float32))
        Y.append(truth.any(0))
    return docs, np.array(Y, dtype=int)


def test_features_shape_and_single_section_documents():
    P = np.array([[0.9, 0.1], [0.2, 0.05], [0.4, 0.6]])
    F = doc_features(P)
    assert F.shape == (2, len(FEATURES))
    one = doc_features(np.array([[0.7, 0.2]]))  # no second section: finite features
    assert np.isfinite(one).all()


def test_calibrator_fixes_inflation_and_is_calibrated(tmp_path):
    docs, Y = make_docs(600)
    fit, ev = split_documents([f"d{i}" for i in range(600)], seed=3)
    fi, ei = [int(d[1:]) for d in fit], [int(d[1:]) for d in ev]
    model = DocPresence(features=FEATURES).fit([docs[i] for i in fi], Y[fi], C_grid=(0.1, 1.0, 10.0), seed=0)
    P = np.stack([model.predict_proba(docs[i]) for i in ei])
    base = np.stack([d.max(0) for d in (docs[i] for i in ei)]) >= 0.5  # the v1 rule: best section >= threshold
    m_new, m_old = evaluate(Y[ei], P, P >= 0.5), evaluate(Y[ei], np.stack([d.max(0) for d in (docs[i] for i in ei)]), base)
    assert m_new["prevalence_abs_error"] < m_old["prevalence_abs_error"]
    assert m_new["macro_f1"] >= m_old["macro_f1"] - 0.02 and m_new["ece"] < 0.08  # the gain is prevalence, not F1
    model.save(tmp_path / "doc_presence.json")
    again = DocPresence.load(tmp_path / "doc_presence.json")
    assert np.allclose(again.predict_proba(docs[ei[0]]), model.predict_proba(docs[ei[0]]))


def test_split_is_by_document_and_deterministic():
    ids = [f"d{i}" for i in range(100)]
    a, b = split_documents(ids, seed=1)
    assert not set(a) & set(b) and len(a) + len(b) == 100 and (a, b) == split_documents(list(reversed(ids)), seed=1)


def test_types_must_match_on_predict():
    docs, Y = make_docs(80, T=3)
    model = DocPresence(features=("logit_max",)).fit(docs, Y, C_grid=(1.0,), seed=0)
    with pytest.raises(ValueError):
        model.predict_proba(np.zeros((4, 5)))


def test_doc_presence_stage_fits_scores_and_writes_profiles_v2(tmp_path):
    import json

    import pandas as pd
    import yaml

    from experiments import distil
    from experiments.config import load_config

    T = ["a", "b", "c"]
    pack = tmp_path / "packs" / "g"
    (pack / "section_probs").mkdir(parents=True)
    (pack / "profiles").mkdir()
    (pack / "profiles" / "part-00000.parquet").write_bytes(b"v1")  # the v1 profiles are kept aside, not lost
    (pack / "schema.yaml").write_text(yaml.safe_dump({"version": 1, "status": "frozen", "types": [{"id": t} for t in T]}))
    (pack / "cascade.json").write_text(json.dumps({"types": T, "threshold": [0.5, 0.5, 0.5]}))
    docs, Y = make_docs(160, T=3, seed=4)
    rows, labels, refs = [], [], []
    for i, P in enumerate(docs):
        d = f"d{i:03d}"
        split = "dev" if i < 120 else "train"
        for j, p in enumerate(P):
            rows.append({"doc_id": d, "split": split, "section_idx": j, **{f"prob_{t}": float(v) for t, v in zip(T, p)}})
            if split == "dev":
                refs.append({"doc_id": d, "section_idx": j, "text_sha1": "x"})
                # the judge labels each section; the document's label is "some section states it"
                lab = {t: int(j == 0 and Y[i, k]) for k, t in enumerate(T)}
                labels.append({"doc_id": d, "section_idx": j, "failed": i == 7, "labels": lab, "set": "documents"})
    pd.DataFrame(rows).to_parquet(pack / "section_probs" / "part-00000.parquet", index=False)
    (pack / "label_sets.json").write_text(json.dumps({"genre": "g", "documents": refs}))
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
    cfg = load_config("configs/distil.yaml", ["smoke"])
    cfg["pack"] = "g"
    cfg["paths"].update(packs=str(tmp_path / "packs"), tables=str(tmp_path / "tables"))
    cfg["presence"]["labels"]["documents"] = str(tmp_path / "labels.jsonl")
    cfg["distil"]["stage"] = "doc_presence"
    cfg["distil"]["doc_presence"] = {"write_profiles": True, "min_positives": 2, "C_grid": [0.1, 1.0]}
    m = distil.main(cfg, tmp_path)
    assert m["g/labelled_documents"] == 119  # d007 has a failed section: left out, never counted as all-negative
    assert m["g/fit_documents"] + m["g/eval_documents"] == 119
    assert m["g/v2/eval/prevalence_abs_error"] <= m["g/v1/eval/prevalence_abs_error"]
    assert m["g/selected_features"] in ("logit_max", "logit_max,logit_second,log_sum,log_n",
                                        "logit_max,logit_second,log_sum,log_n,logit_noisy_or")
    assert (pack / "doc_presence.json").is_file() and (tmp_path / "tables" / "doc_presence_g.csv").is_file()
    assert (tmp_path / "tables" / "doc_presence_types_g.csv").is_file()
    assert (pack / "profiles_v1" / "part-00000.parquet").read_bytes() == b"v1"
    prof = pd.read_parquet(pack / "profiles" / "part-00000.parquet")
    assert len(prof) == 40 and set(prof["split"]) == {"train"}  # only the indexed split
    assert {"prob_a", "bit_a", "best_a"} <= set(prof.columns) and json.loads((pack / "profiles" / "manifest.json").read_text())["version"] == 2
    assert m["g/profiles_v2/types_present_per_document"] <= m["g/profiles_v2/v1_types_present_per_document"]
