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
