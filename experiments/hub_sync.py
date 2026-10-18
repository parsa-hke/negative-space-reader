"""Move data and artefacts between this repository and a private Hugging Face dataset repo (plan §0.6: the Mac and
Colab share one store).

    python -m experiments.hub_sync push --bundles data,embeddings,cache,packs,runs
    python -m experiments.hub_sync pull --bundles data,embeddings,cache,packs,runs

Bundles are defined in configs/base.yaml (`hub.bundles`): include/exclude patterns relative to the repository root
(fnmatch, `*` crosses folders), or `archive: <folder>` for a folder of many small files, which travels as one
tar.gz under archives/. Push uploads only files whose content differs from the hub's copy (git blob sha1 for small
files, sha256 for LFS files), in commits of bounded size, and never deletes anything on the hub. runs/log.csv is
merged rather than overwritten (push_log / pull_log): the hub keeps the union of every machine's rows, and each
machine appends the rows it lacks. Credentials come from the logged-in Hugging Face client (HF_TOKEN or `hf auth
login`); none is read or printed here. Symlinked folders (a git worktree) are followed.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import os
import tarfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
LOG_IN_HUB = "logs/runs_log.csv"
MAX_COMMIT_BYTES = 512 * 1024 ** 2   # a failed upload (network timeout) loses at most one such chunk
MAX_COMMIT_FILES = 500
COMMIT_RETRIES = 4
RETRY_WAIT_S = 30.0


def git_blob_sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def _file_hashes(path: Path) -> tuple[str, str]:
    """(git blob sha1, sha256) of a file, streamed."""
    size = path.stat().st_size
    g, s = hashlib.sha1(b"blob %d\x00" % size), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            g.update(chunk)
            s.update(chunk)
    return g.hexdigest(), s.hexdigest()


def _matches(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p.rstrip("/*") + "/*") for p in patterns)


def select(root: Path, bundle: dict[str, Any]) -> list[str]:
    """Repository-relative paths (posix) of the bundle's files, following symlinked folders."""
    include, exclude = bundle.get("include", []), bundle.get("exclude", [])
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", "__pycache__")
                       and not _matches(f"{rel_dir}/{d}".lstrip("./"), exclude)]
        for name in filenames:
            rel = f"{rel_dir}/{name}".lstrip("./") if rel_dir != "." else name
            if _matches(rel, include) and not _matches(rel, exclude):
                out.append(rel)
    return sorted(out)


def _remote(api: Any, repo_id: str) -> dict[str, Any]:
    try:
        return {e.path: e for e in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True, expand=True)
                if hasattr(e, "size")}
    except Exception:  # an empty repo has no tree yet
        return {}


def _archive(folder: Path) -> bytes:
    import gzip

    buf = io.BytesIO()
    # gzip writes the current time into its header; mtime=0 keeps the archive byte-identical for identical files
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                info = tar.gettarinfo(str(p), arcname=p.relative_to(folder).as_posix())
                info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""  # same bytes for the same files
                with open(p, "rb") as f:
                    tar.addfile(info, f)
    return buf.getvalue()


def push(api: Any, repo_id: str, root: Path, bundles: dict[str, Any], names: list[str]) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd

    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    remote = _remote(api, repo_id)
    ops: list[Any] = []
    sizes: list[int] = []
    for name in names:
        b = bundles[name]
        if "archive" in b:
            folder = root / b["archive"]
            if not folder.is_dir():
                continue
            data = _archive(folder)
            path = f"archives/{Path(b['archive']).name}.tar.gz"
            r = remote.get(path)
            if r is None or getattr(r, "blob_id", None) != git_blob_sha1_bytes(data) and not (
                    getattr(r, "lfs", None) and r.lfs.sha256 == hashlib.sha256(data).hexdigest()):
                staged = root / ".cache" / "hub_sync" / path  # a file, so the hub's chunked upload can be used
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(data)
                ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(staged)))
                sizes.append(len(data))
            continue
        for rel in select(root, b):
            p = (root / rel).resolve()
            r = remote.get(rel)
            if r is not None and r.size == p.stat().st_size:
                g, s = _file_hashes(p)
                if getattr(r, "blob_id", None) == g or (getattr(r, "lfs", None) and r.lfs.sha256 == s):
                    continue
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(p)))
            sizes.append(p.stat().st_size)
    commits, start = 0, 0
    while start < len(ops):  # bounded commits: a failed upload loses one chunk, not everything
        end, total = start, 0
        while end < len(ops) and end - start < MAX_COMMIT_FILES and (total + sizes[end] <= MAX_COMMIT_BYTES or end == start):
            total += sizes[end]
            end += 1
        for attempt in range(COMMIT_RETRIES + 1):  # network timeouts happen on long uploads: retry the chunk
            try:
                api.create_commit(repo_id=repo_id, repo_type="dataset", operations=ops[start:end],
                                  commit_message=f"sync {','.join(names)}: {end - start} files")
                break
            except Exception as e:
                if attempt == COMMIT_RETRIES:
                    raise
                wait = RETRY_WAIT_S * 2 ** attempt
                print(f"[hub_sync] commit of {end - start} files failed ({type(e).__name__}); retry {attempt + 1}/"
                      f"{COMMIT_RETRIES} in {wait:.0f}s", flush=True)
                time.sleep(wait)
        commits += 1
        print(f"[hub_sync] committed {end}/{len(ops)} files", flush=True)
        start = end
    return {"uploaded": len(ops), "bytes": sum(sizes), "commits": commits}


def _read_log(text: str) -> tuple[list[str], list[dict[str, str]]]:
    r = csv.DictReader(io.StringIO(text))
    return list(r.fieldnames or []), list(r)


def _key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("timestamp", ""), row.get("run_id", "")


def push_log(api: Any, repo_id: str, root: Path) -> int:
    """Union of the hub's run log and this machine's runs/log.csv, written back to the hub; returns rows added."""
    from huggingface_hub import CommitOperationAdd

    local_path = root / "runs" / "log.csv"
    if not local_path.is_file():
        return 0
    fields, local = _read_log(local_path.read_text(encoding="utf-8"))
    hub_fields, hub = fields, []
    try:
        hub_fields, hub = _read_log(Path(api.hf_hub_download(repo_id, LOG_IN_HUB, repo_type="dataset",
                                                             local_dir=root / ".cache" / "hub_sync")).read_text())
    except Exception:
        pass
    have = {_key(r) for r in hub}
    new = [r for r in local if _key(r) not in have]
    if not new and hub:
        return 0
    rows = sorted(hub + new, key=_key)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=hub_fields or fields)
    w.writeheader()
    w.writerows(rows)
    api.create_commit(repo_id=repo_id, repo_type="dataset", commit_message=f"run log: +{len(new)} rows",
                      operations=[CommitOperationAdd(path_in_repo=LOG_IN_HUB,
                                                     path_or_fileobj=io.BytesIO(buf.getvalue().encode()))])
    return len(new)


def pull_log(api: Any, repo_id: str, root: Path) -> int:
    """Append to runs/log.csv the hub's rows this machine lacks (never rewrites existing rows); returns rows added."""
    try:
        path = Path(api.hf_hub_download(repo_id, LOG_IN_HUB, repo_type="dataset", local_dir=root / ".cache" / "hub_sync"))
    except Exception:
        return 0
    _, hub = _read_log(path.read_text(encoding="utf-8"))
    local_path = root / "runs" / "log.csv"
    fields, local = _read_log(local_path.read_text(encoding="utf-8"))
    have = {_key(r) for r in local}
    new = [r for r in hub if _key(r) not in have]
    if new:
        with open(local_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields, extrasaction="ignore").writerows(new)
    return len(new)


def _snapshot(repo_id: str, root: Path, include: list[str], exclude: list[str]) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", local_dir=root, allow_patterns=include,
                      ignore_patterns=exclude or None)


def pull(api: Any, repo_id: str, root: Path, bundles: dict[str, Any], names: list[str]) -> dict[str, Any]:
    pulled = []
    for name in names:
        b = bundles[name]
        if "archive" in b:
            path = f"archives/{Path(b['archive']).name}.tar.gz"
            local = Path(api.hf_hub_download(repo_id, path, repo_type="dataset", local_dir=root / ".cache" / "hub_sync"))
            dest = root / b["archive"]
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(local, mode="r:gz") as tar:
                tar.extractall(dest, filter="data")  # the "data" filter refuses absolute paths and links out of dest
            pulled.append(name)
            continue
        _snapshot(repo_id, root, [p if p.endswith("*") else p for p in b.get("include", [])], b.get("exclude", []))
        pulled.append(name)
    return {"pulled": pulled}


def main() -> None:
    import yaml
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--bundles", default="data,embeddings,cache,packs,runs")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    hub = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))["hub"]
    names = [n.strip() for n in args.bundles.split(",") if n.strip()]
    api = HfApi()
    if args.action == "push":
        print(push(api, hub["repo"], REPO, hub["bundles"], names), flush=True)
        if "runs" in names:
            print({"log_rows_added_to_hub": push_log(api, hub["repo"], REPO)})
    else:
        print(pull(api, hub["repo"], REPO, hub["bundles"], names), flush=True)
        if "runs" in names:
            print({"log_rows_added_here": pull_log(api, hub["repo"], REPO)})


if __name__ == "__main__":
    main()
