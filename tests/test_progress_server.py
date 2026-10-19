"""analysis/progress_server.py: answers in the teacher cache are matched to their run, pass and label set."""

import hashlib
import json

from analysis.progress_server import Tracker, classify, load_sets
from data.schema import Section
from packs.presence_labels import render_check_prompt, render_label_prompt

GENRE = {"document_kind": "test document", "author_role": "the writer"}
TYPES = [{"id": "a", "definition": "States a."}]


def entry(route, prompt):
    return {"route": route, "messages": [{"role": "user", "content": prompt}], "output": "{}"}


def test_classify_and_count(tmp_path):
    sec = Section(doc_id="d", idx=0, heading="H", text='Alpha is stated.\nA second "line".')
    sha = hashlib.sha1(sec.text.encode("utf-8")).hexdigest()
    first = render_label_prompt(sec, GENRE, TYPES, max_assertions=5)
    check = render_check_prompt(sec, GENRE, TYPES, TYPES)
    assert classify(entry("r/qwen", first)) == ("r/qwen", "first", sha)
    assert classify(entry("r/qwen", check)) == ("r/qwen", "check", sha)
    assert classify(entry("r/qwen", "Name this cluster.")) is None

    (tmp_path / "packs" / "g").mkdir(parents=True)
    (tmp_path / "packs" / "g" / "label_sets.json").write_text(json.dumps(
        {"genre": "g", "human": [{"doc_id": "d", "section_idx": 0, "text_sha1": sha}], "teacher": []}))
    sets = load_sets(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, e in (("1", entry("x:qwen/q@p", first)), ("2", entry("x:qwen/q@p", check)),
                    ("3", entry("x:claude@p", first)), ("4", entry("x:qwen/q@p", first))):  # a retry: same section
        (cache / f"{name}.json").write_text(json.dumps(e))
    t = Tracker(cache, sets)
    t.scan()
    assert t.counts("qwen", "g", "human", "first") == 1 and t.counts("qwen", "g", "human", "check") == 1
    assert t.counts("claude", "g", "human", "first") == 1 and t.counts("claude", "g", "human", "check") == 0
    t.scan()  # files are read once
    assert len(t.seen) == 4


def test_documents_set_is_counted_and_any_recent_run_log_is_shown(tmp_path):
    import os
    import time

    from analysis.progress_server import recent_runs

    (tmp_path / "packs" / "g").mkdir(parents=True)
    (tmp_path / "packs" / "g" / "label_sets.json").write_text(json.dumps(
        {"genre": "g", "human": [], "teacher": [], "documents": [{"doc_id": "d", "section_idx": 0, "text_sha1": "abc"}],
         "documents_selection": {"documents": 1}}))
    assert load_sets(tmp_path) == {"abc": ("g", "documents")}

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "judge_docs_haiku.out").write_text("loading\r[teacher] batch: 10/20 still processing (60s)\n")
    (runs / "section_probs_v1.out").write_text("[section_probs privacy] part 1/2\n[ok] runs/section_probs_v1_s0\n")
    (runs / "old_run.out").write_text("[ok] long ago\n")
    old = time.time() - 3 * 86400
    os.utime(runs / "old_run.out", (old, old))
    rr = {r["name"]: r for r in recent_runs(tmp_path, time.time(), hours=24, skip=set())}
    assert set(rr) == {"judge_docs_haiku", "section_probs_v1"}
    assert rr["section_probs_v1"]["state"] == "finished"
    assert rr["judge_docs_haiku"]["log"][-1].startswith("[teacher] batch: 10/20")
