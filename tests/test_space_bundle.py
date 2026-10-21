"""app/space_bundle.py (plan 7.1): the Space bundle holds the code, configs, examples and the pack subset the app
reads, with Git LFS for the vectors and nothing from the corpus or the training artefacts; and a pack loads without
the corpus when it has a stored evidence set (packs/spec.py section_texts_or_none)."""

import json

import numpy as np
import pytest


def fake_project(tmp_path):
    root = tmp_path / "proj"
    for d in ["app/static", "app/examples", "engine", "packs/g/classifier/linear_teacher", "packs/g/prototypes", "packs/g/profiles",
              "packs/g/evidence", "packs/g/index", "packs/g/section_probs", "packs/g/profiles_v1", "data/processed/g", "experiments",
              "models", "configs", "prompts", "tests"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "app" / "gradio_app.py").write_text("x = 1\n")
    (root / "app" / "static" / "nsr.css").write_text("body{}\n")
    (root / "app" / "examples" / "g.json").write_text("[]\n")
    (root / "engine" / "pipeline.py").write_text("x = 1\n")
    (root / "engine" / "notes.txt").write_text("not code\n")
    (root / "packs" / "spec.py").write_text("x = 1\n")
    (root / "experiments" / "config.py").write_text("x = 1\n")
    (root / "configs" / "serve.yaml").write_text("task: serve\n")
    (root / "prompts" / "explain.md").write_text("---\nid: explain\nversion: 1\n---\nx\n")
    (root / "tests" / "test_x.py").write_text("x = 1\n")
    for name in ("schema.yaml", "pack.json", "cascade.json", "doc_presence.json"):
        (root / "packs" / "g" / name).write_text("{}\n")
    (root / "packs" / "g" / "schema_draft.yaml").write_text("draft\n")
    (root / "packs" / "g" / "label_sets.json").write_text("{}\n")
    (root / "data" / "segment.py").write_text("x = 1\n")
    (root / "packs" / "g" / "classifier" / "linear_teacher" / "heads.npz").write_bytes(b"0" * 10)
    (root / "packs" / "g" / "prototypes" / "prototypes.json").write_text("{}\n")
    (root / "packs" / "g" / "profiles" / "part-00000.parquet").write_bytes(b"0" * 10)
    (root / "packs" / "g" / "evidence" / "part-00000.parquet").write_bytes(b"0" * 10)
    (root / "packs" / "g" / "section_probs" / "part-00000.parquet").write_bytes(b"0" * 1000)
    (root / "packs" / "g" / "profiles_v1" / "part-00000.parquet").write_bytes(b"0" * 1000)
    for name in ("index.json", "payloads.jsonl", "vectors.npy", "best_sections.npz", "sections.npz"):
        (root / "packs" / "g" / "index" / name).write_bytes(b"0" * 100)
    (root / "data" / "processed" / "g" / "sections.jsonl").write_text("{}\n")
    return root


def test_bundle_copies_what_the_app_reads_and_nothing_else(tmp_path):
    from app.space_bundle import bundle

    root = fake_project(tmp_path)
    m = bundle(root, tmp_path / "space", {"genres": ["g"], "serve": {"packs": ["g"]}}, clean=True)
    out = tmp_path / "space"
    assert (out / "app.py").is_file() and (out / "README.md").read_text().startswith("---\ntitle: Negative-Space Reader")
    assert "sdk: gradio" in (out / "README.md").read_text() and "faiss-cpu==" in (out / "requirements.txt").read_text()
    app_py, req = (out / "app.py").read_text(), (out / "requirements.txt").read_text()
    assert app_py.startswith('"""') and "import spaces" in app_py and '"configs/space_zerogpu.yaml"' in app_py  # zerogpu default
    assert "\nspaces\n" in req and "whl/cpu" not in req and m["variant"] == "zerogpu"
    cpu = bundle(root, tmp_path / "space_cpu", {"genres": ["g"]}, clean=True, variant="cpu")
    cpu_app, cpu_req = (tmp_path / "space_cpu" / "app.py").read_text(), (tmp_path / "space_cpu" / "requirements.txt").read_text()
    assert "import spaces" not in cpu_app and "space_zerogpu" not in cpu_app and cpu_req.startswith("--extra-index-url") and cpu["overlays"] == ["configs/space.yaml"]
    assert (out / ".gitattributes").read_text().startswith("*.npy filter=lfs")
    assert (out / "app" / "static" / "nsr.css").is_file() and (out / "app" / "examples" / "g.json").is_file()
    assert (out / "engine" / "pipeline.py").is_file() and not (out / "engine" / "notes.txt").exists()
    assert (out / "packs" / "g" / "index" / "vectors.npy").is_file() and not (out / "packs" / "g" / "index" / "sections.npz").exists()
    assert (out / "packs" / "g" / "evidence" / "part-00000.parquet").is_file()
    for absent in ("packs/g/section_probs", "packs/g/profiles_v1", "packs/g/schema_draft.yaml", "data/processed", "tests", "packs/g/label_sets.json"):
        assert not (out / absent).exists(), absent
    assert (out / "packs" / "spec.py").is_file() and (out / "data" / "segment.py").is_file()
    assert m["missing"] == [] and m["lfs_bytes"] == 100 and m["parts"]["packs/g"]["files"] >= 9
    assert json.loads((out / "BUNDLE.json").read_text())["genres"] == ["g"]


def test_a_missing_pack_artefact_is_reported(tmp_path):
    from app.space_bundle import bundle

    root = fake_project(tmp_path)
    (root / "packs" / "g" / "index" / "vectors.npy").unlink()
    m = bundle(root, tmp_path / "space", {"genres": ["g"]}, clean=True)
    assert m["missing"] == ["packs/g/index/vectors.npy"]


def test_section_texts_are_optional_only_with_a_stored_evidence_set(tmp_path):
    from packs.spec import section_texts_or_none

    idx = tmp_path / "index"
    idx.mkdir()
    sections = tmp_path / "sections.jsonl"
    sections.write_text('{"doc_id": "d", "idx": 0, "text": "Hello."}\n')
    np.savez(idx / "sections.npz", doc_id=np.array(["d"]), idx=np.array([0]), offset=np.array([0]), size=np.array([sections.stat().st_size]))
    assert section_texts_or_none(idx, sections, has_store=False)("d", 0) == "Hello."
    assert section_texts_or_none(idx, tmp_path / "absent.jsonl", has_store=True)("d", 0) is None
    with pytest.raises(FileNotFoundError):
        section_texts_or_none(idx, tmp_path / "absent.jsonl", has_store=False)
