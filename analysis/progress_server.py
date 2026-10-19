"""Live progress of the teacher-labelling runs (plan 4.1/4.2), served on localhost for the app's browser pane.

    .venv/bin/python -m analysis.progress_server --port 8765     # then open http://localhost:8765

Also any other run log written in the last 24 hours (runs/<name>.out: state and last lines, e.g. a Message Batch's
progress), and the corpus-profiles run (plan 4.7): finished Parquet parts per genre against the parts the indexable corpus needs,
the rate from the run log, and an ETA from the parts' timestamps.

Counts are exact, read from the teacher's response cache (.cache/negspace/teacher/*.json: route, prompt, output):
each answer is matched to its section by the sha1 of the section text inside the prompt and to the fixed label sets
(packs/<genre>/label_sets.json), and to its pass by the prompt (first pass: label_presence; second look:
label_presence_check). Every new cache file is read once. Run state comes from the process list and the run logs
(runs/<name>.out). Read-only: it never writes to the cache, the runs or the data. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / ".cache" / "negspace" / "teacher"

# The runs to watch: which model route they use, which label sets they label, their log and process name.
RUNS = [
    {"key": "teacher", "title": "Teacher: Qwen2.5-72B", "route": "qwen/qwen-2.5-72b-instruct",
     "sets": ["teacher", "human"], "log": "runs/label_72b.out", "process": "name label_72b --set"},
    {"key": "judge", "title": "Second judge: Claude Haiku 5.5", "route": "anthropic/claude-haiku-5.5",
     "sets": ["human"], "log": "runs/judge_haiku.out", "process": "name judge_haiku --set"},
    {"key": "judge_docs", "title": "Judge on whole documents: Claude Haiku 5.5 (Anthropic API, batches)",
     "route": "anthropic:claude-haiku-5-5", "sets": ["documents"], "log": "runs/judge_docs_haiku.out",
     "process": "name judge_docs_haiku --set"},
]
SET_NAMES = {"teacher": "teacher set", "human": "reference set", "documents": "whole documents"}
_TEXT = re.compile(r'Section text:\n"""\n(.*?)\n"""\n\n', re.S)


def classify(entry: dict[str, Any]) -> tuple[str, str, str] | None:
    """(route, pass, section sha1) of one cached answer, or None when it is not a labelling call."""
    prompt = (entry.get("messages") or [{}])[-1].get("content", "")
    if '{"statements"' in prompt:
        kind = "first"
    elif '{"present"' in prompt:
        kind = "check"
    else:
        return None
    m = _TEXT.search(prompt)
    if not m:
        return None
    return entry.get("route", ""), kind, hashlib.sha1(m.group(1).encode("utf-8")).hexdigest()


def load_sets(repo: Path = REPO) -> dict[str, tuple[str, str]]:
    """section text sha1 -> (genre, set name), for every fixed label set."""
    out = {}
    for path in sorted((repo / "packs").glob("*/label_sets.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        for name in SET_NAMES:
            for r in d.get(name, []):
                out[r["text_sha1"]] = (d["genre"], name)
    return out


class Tracker:
    """Incremental index of the cache: each new file is read once; done[(route, kind)] = set of section sha1."""

    def __init__(self, cache: Path, sets: dict[str, tuple[str, str]]):
        self.cache, self.sets = cache, sets
        self.seen: set[str] = set()
        self.done: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.events: dict[str, deque] = defaultdict(lambda: deque(maxlen=20000))  # route -> answer times
        self.last: dict[str, float] = {}
        self.lock = threading.Lock()

    def scan(self) -> None:
        if not self.cache.is_dir():
            return
        new = []
        with os.scandir(self.cache) as it:
            for e in it:
                if e.name not in self.seen and e.name.endswith(".json"):
                    new.append(e)
        for e in new:
            try:
                entry = json.loads(Path(e.path).read_text(encoding="utf-8"))
                mtime = e.stat().st_mtime
            except (OSError, ValueError):
                continue  # being written: next scan
            with self.lock:
                self.seen.add(e.name)
                c = classify(entry)
                if c is None:
                    continue
                route, kind, sha = c
                self.done[(route, kind)].add(sha)
                self.events[route].append(mtime)
                self.last[route] = max(self.last.get(route, 0.0), mtime)

    def counts(self, route_part: str, genre: str, name: str, kind: str) -> int:
        with self.lock:
            return sum(1 for (route, k), shas in self.done.items() if route_part in route and k == kind
                       for s in shas if self.sets.get(s) == (genre, name))


def _running(pattern: str) -> bool:
    try:
        return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0
    except OSError:
        return False


def _log_tail(path: Path) -> tuple[str, list[str]]:
    if not path.is_file():
        return "not started", []
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    keep = [ln for ln in lines if re.search(r"teacher_labels|\[teacher\] (batch|submitted|resuming)|\[ok\]|\[failed\]|Error|Traceback|WARNING", ln)]
    state = "finished" if any(ln.startswith("[ok]") for ln in lines) else (
        "failed" if any(ln.startswith("[failed]") or "Traceback" in ln for ln in lines) else "")
    return state, keep[-6:]


def recent_runs(repo: Path, now: float, *, hours: float = 24, skip: set[str]) -> list[dict[str, Any]]:
    """Every run log runs/<name>.out written in the last `hours` that no dedicated card shows: state (running from the
    process list, else finished / failed from the log), the last lines, the age of the last write. Any long run
    started with `> runs/<name>.out` shows up here without code changes."""
    out = []
    for path in sorted((repo / "runs").glob("*.out"), key=lambda p: -p.stat().st_mtime):
        age = now - path.stat().st_mtime
        if age > hours * 3600 or path.stem in skip:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        alive = _running(f"name {path.stem}")
        state = "running" if alive else ("finished" if any(ln.startswith("[ok]") for ln in lines) else (
            "failed" if any(ln.startswith("[failed]") or "Traceback" in ln for ln in lines) else "stopped"))
        out.append({"name": path.stem, "state": state, "alive": alive, "last_write_s": round(age), "log": lines[-8:]})
    return out


def status(tracker: Tracker, totals: dict[tuple[str, str], int]) -> dict[str, Any]:
    tracker.scan()
    now = time.time()
    runs = []
    for run in RUNS:
        alive = _running(run["process"])
        log_state, log_lines = _log_tail(REPO / run["log"])
        with tracker.lock:
            times = [t for r, q in tracker.events.items() if run["route"] in r for t in q]
            last = max([v for r, v in tracker.last.items() if run["route"] in r], default=None)
        recent = [t for t in times if t >= now - 600]
        per_min = len([t for t in recent if t >= now - 120]) / 2
        buckets = [0] * 60  # answers per 10 s over the last 10 minutes
        for t in recent:
            buckets[min(59, int((now - t) // 10))] += 1
        groups, remaining = [], 0
        for genre in ("privacy", "papers"):
            for name in run["sets"]:
                n = totals.get((genre, name), 0)
                first = tracker.counts(run["route"], genre, name, "first")
                check = tracker.counts(run["route"], genre, name, "check")
                remaining += (n - first) + max(0, first - check) + (n - first)  # checks follow first passes
                groups.append({"genre": genre, "set": SET_NAMES[name], "total": n, "first": first, "check": check})
        state = "running" if alive else (log_state or "stopped")
        runs.append({**{k: run[k] for k in ("key", "title", "route")}, "state": state, "alive": alive,
                     "per_min": per_min, "eta_min": round(remaining / per_min) if per_min and alive else None,
                     "last_answer_s": round(now - last) if last else None, "spark": buckets[::-1],
                     "groups": groups, "log": log_lines})
    skip = {Path(r["log"]).stem for r in RUNS} | {Path(PROFILES["log"]).stem}
    skip |= {p.stem for p in (REPO / "runs").glob("hub_push_*.out")}
    return {"now": now, "runs": runs, "profiles": profiles_status(now), "upload": upload_status(now),
            "recent": recent_runs(REPO, now, skip=skip)}


PROFILES = {"title": "Corpus profiles (4.7)", "log": "runs/profiles_v1.out", "process": "name profiles_v1",
            "chunk": 500, "splits": ("train",)}


def profiles_status(now: float) -> dict[str, Any]:
    """Parts written per genre vs parts needed, rate and ETA, for the profiles run."""
    alive = _running(PROFILES["process"])
    log_path = REPO / PROFILES["log"]
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    rates = {g: float(m.group(2)) for g, m in
             ((m.group(1), m) for m in re.finditer(r"\[profiles (\w+)\] part \d+/\d+: .*?\(([\d.]+) sections/s\)", text))}
    state = "finished" if "\n[ok]" in text or text.startswith("[ok]") else (
        "failed" if "[failed]" in text or "Traceback" in text else "")
    genres, last_part, times = [], None, []
    for genre in ("privacy", "papers"):
        manifest = REPO / "data" / "processed" / "splits" / f"{genre}.json"
        docs = 0
        if manifest.is_file():
            ids = json.loads(manifest.read_text(encoding="utf-8")).get("doc_ids", {})
            docs = sum(len(ids.get(s, [])) for s in PROFILES["splits"])
        total = -(-docs // PROFILES["chunk"])
        parts = sorted((REPO / "packs" / genre / "profiles").glob("part-*.parquet"))
        mt = sorted(p.stat().st_mtime for p in parts)
        times += mt
        if mt:
            last_part = max(last_part or 0, mt[-1])
        recent = [t for t in mt if t >= now - 900]
        per_part = (recent[-1] - recent[0]) / (len(recent) - 1) if len(recent) >= 2 else None
        genres.append({"genre": genre, "parts": len(parts), "total": total, "documents": docs,
                       "rate": rates.get(genre), "per_part_s": per_part})
    eta = None
    if alive:
        pending = [g for g in genres if g["parts"] < g["total"]]
        if pending:
            cur = pending[0]
            known = cur["per_part_s"] or next((g["per_part_s"] for g in genres if g["per_part_s"]), None)
            if known:  # later genres: scale by the ratio of their rates when only one rate is known
                eta = round(sum((g["total"] - g["parts"]) * (cur["per_part_s"] or known) for g in pending) / 60)
    lines = [ln.strip() for ln in text.replace("\r", "\n").splitlines()
             if re.search(r"\[profiles|\[ok\]|\[failed\]|Error|Traceback", ln)]
    return {"title": PROFILES["title"], "state": "running" if alive else (state or "stopped"), "alive": alive,
            "eta_min": eta, "last_part_s": round(now - last_part) if last_part else None, "genres": genres,
            "log": lines[-5:]}


def upload_status(now: float) -> dict[str, Any] | None:
    """The latest hub_sync push (runs/hub_push_*.out): files committed, the current chunk's bytes and rate, retries."""
    logs = sorted((REPO / "runs").glob("hub_push_*.out"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    log = logs[-1]
    text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    committed = re.findall(r"\[hub_sync\] committed (\d+)/(\d+) files", text)
    plan = re.findall(r"\[hub_sync\] plan: (\d+) files, ([\d.]+) GB", text)
    chunk = re.findall(r"Processing Files \((\d+) / (\d+)\)\s*:\s*(\d+)%\|[^|]*\|\s*([\d.]+\s*\w?B) /\s*([\d.]+\s*\w?B),\s*([\d.]+\s*\w?B/s)", text)
    retries = re.findall(r"\[hub_sync\] commit of .*", text)
    done = re.search(r"\{'uploaded': (\d+)", text)
    failed = "Traceback" in text
    alive = _running("experiments.hub_sync push")
    c = committed[-1] if committed else None
    k = chunk[-1] if chunk else None
    return {"title": "Upload to the private dataset repo", "log": log.name,
            "state": "running" if alive else ("finished" if done else ("failed" if failed else "stopped")),
            "files_done": int(c[0]) if c else 0, "files_total": int(c[1]) if c else (int(plan[-1][0]) if plan else None),
            "planned_gb": float(plan[-1][1]) if plan else None,
            "chunk": {"files": f"{k[0]}/{k[1]}", "pct": int(k[2]), "done": k[3], "total": k[4], "rate": k[5]} if k else None,
            "retries": retries[-3:], "last_write_s": round(now - log.stat().st_mtime)}


PAGE = (Path(__file__).resolve().parent / "progress_page.html")


class Handler(BaseHTTPRequestHandler):
    tracker: Tracker
    totals: dict[tuple[str, str], int]

    def do_GET(self) -> None:
        if self.path.startswith("/status.json"):
            body = json.dumps(status(self.tracker, self.totals)).encode("utf-8")
            ctype = "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = PAGE.read_bytes(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    sets = load_sets()
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for g_n in sets.values():
        totals[g_n] += 1
    Handler.tracker, Handler.totals = Tracker(CACHE, sets), dict(totals)
    t0 = time.time()
    Handler.tracker.scan()
    print(f"indexed {len(Handler.tracker.seen)} cache files in {time.time() - t0:.1f}s; http://localhost:{args.port}",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
