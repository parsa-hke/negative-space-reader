"""experiments/hub_sync.py: bundles of files between this repo and the private HF dataset repo (no network here)."""

import csv
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

from experiments import hub_sync as H

BUNDLES = {"data": {"include": ["data/processed/*"], "exclude": ["data/processed/_backup*"]},
           "cache": {"archive": ".cache/negspace/teacher"}}


def make_repo(root):
    (root / "data" / "processed" / "g").mkdir(parents=True)
    (root / "data" / "processed" / "g" / "a.jsonl").write_text("one\n")
    (root / "data" / "processed" / "g" / "b.jsonl").write_text("two\n")
    (root / "data" / "processed" / "_backup_x").mkdir()
    (root / "data" / "processed" / "_backup_x" / "old.jsonl").write_text("old\n")
    (root / ".cache" / "negspace" / "teacher").mkdir(parents=True)
    (root / ".cache" / "negspace" / "teacher" / "k1.json").write_text('{"x": 1}')
    (root / "runs").mkdir()
    (root / "runs" / "log.csv").write_text("timestamp,run_id,status\n2026-10-18T10:00:00,a_s0,ok\n")


class FakeApi:
    def __init__(self):
        self.files, self.commits = {}, []

    def create_repo(self, *a, **k):
        pass

    def list_repo_tree(self, repo_id, repo_type, recursive, expand):
        for p, data in self.files.items():
            yield SimpleNamespace(path=p, size=len(data), lfs=None, blob_id=H.git_blob_sha1_bytes(data))

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append([op.path_in_repo for op in operations])
        for op in operations:
            src = op.path_or_fileobj
            self.files[op.path_in_repo] = src.read_bytes() if hasattr(src, "read_bytes") else (
                src.getvalue() if hasattr(src, "getvalue") else open(src, "rb").read())

    def hf_hub_download(self, repo_id, filename, repo_type, local_dir):
        out = local_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.files[filename])
        return str(out)


def test_select_follows_patterns(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "data" / "processed" / "g" / ".DS_Store").write_bytes(b"\0")  # Finder junk never travels
    files = H.select(tmp_path, BUNDLES["data"])
    assert files == ["data/processed/g/a.jsonl", "data/processed/g/b.jsonl"]


def test_push_uploads_only_changes_and_archives_the_cache(tmp_path):
    make_repo(tmp_path)
    api = FakeApi()
    m = H.push(api, "u/r", tmp_path, BUNDLES, ["data", "cache"])
    assert m["uploaded"] == 3 and "archives/teacher.tar.gz" in api.files  # 2 data files + the cache archive
    import time

    time.sleep(1.1)  # a new second: the archive must still come out byte-identical
    m = H.push(api, "u/r", tmp_path, BUNDLES, ["data", "cache"])
    assert m["uploaded"] == 0 and len(api.commits) == 1  # nothing changed: no second commit
    (tmp_path / "data" / "processed" / "g" / "b.jsonl").write_text("two, edited\n")
    m = H.push(api, "u/r", tmp_path, BUNDLES, ["data"])
    assert m["uploaded"] == 1 and api.commits[-1] == ["data/processed/g/b.jsonl"]
    with tarfile.open(fileobj=io.BytesIO(api.files["archives/teacher.tar.gz"]), mode="r:gz") as t:
        assert t.getnames() == ["k1.json"]


def test_run_logs_are_merged_not_overwritten(tmp_path):
    a, b = tmp_path / "mac", tmp_path / "colab"
    for r in (a, b):
        make_repo(r)
    (b / "runs" / "log.csv").write_text("timestamp,run_id,status\n2026-10-18T10:00:00,a_s0,ok\n"
                                        "2026-10-18T12:00:00,colab_s0,ok\n")
    api = FakeApi()
    H.push_log(api, "u/r", b)
    H.push_log(api, "u/r", a)  # the Mac's log has nothing new: the hub keeps Colab's row
    added = H.pull_log(api, "u/r", a)
    rows = list(csv.DictReader(open(a / "runs" / "log.csv")))
    assert added == 1 and [r["run_id"] for r in rows] == ["a_s0", "colab_s0"]
    assert H.pull_log(api, "u/r", a) == 0  # idempotent


def test_pull_restores_the_cache_archive(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_repo(src)
    dst.mkdir()
    api = FakeApi()
    H.push(api, "u/r", src, BUNDLES, ["cache"])
    monkeypatch.setattr(H, "_snapshot", lambda *a, **k: None)
    H.pull(api, "u/r", dst, BUNDLES, ["cache"])
    assert json.loads((dst / ".cache" / "negspace" / "teacher" / "k1.json").read_text()) == {"x": 1}


def test_git_blob_sha1_matches_git():
    assert H.git_blob_sha1_bytes(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert hashlib.sha1(b"blob 6\x00hello\n").hexdigest() == H.git_blob_sha1_bytes(b"hello\n")


def test_push_retries_a_failed_commit(tmp_path, monkeypatch):
    make_repo(tmp_path)
    api = FakeApi()
    real, calls = api.create_commit, []

    def flaky(**kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("socket timeout")
        return real(**kw)

    api.create_commit = flaky
    monkeypatch.setattr(H, "RETRY_WAIT_S", 0)
    m = H.push(api, "u/r", tmp_path, BUNDLES, ["data"])
    assert m["uploaded"] == 2 and len(calls) == 2 and "data/processed/g/a.jsonl" in api.files


def test_status_reports_differences_without_committing(tmp_path):
    make_repo(tmp_path)
    api = FakeApi()
    s = H.status(api, "u/r", tmp_path, BUNDLES, ["data", "cache"])
    assert s["differs"] == 3 and s["missing_on_hub"] == 3 and api.commits == []  # read-only
    H.push(api, "u/r", tmp_path, BUNDLES, ["data", "cache"])
    s = H.status(api, "u/r", tmp_path, BUNDLES, ["data", "cache"])
    assert s["differs"] == 0 and s["matched"] == 3 and s["files"] == []
    (tmp_path / "data" / "processed" / "g" / "a.jsonl").write_text("one, edited\n")
    s = H.status(api, "u/r", tmp_path, BUNDLES, ["data"])
    assert s["differs"] == 1 and s["missing_on_hub"] == 0 and s["files"] == ["data/processed/g/a.jsonl"]
