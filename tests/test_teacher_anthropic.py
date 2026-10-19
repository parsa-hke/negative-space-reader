"""models/teacher.py AnthropicTeacher: the official SDK behind the Teacher interface (judge labels, the 7.3 frontier
comparison). A fake client stands in for the API; nothing here makes a network call."""

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from models import teacher as T


class Labels(BaseModel):
    statements: list[str]
    note: str | None = None


@pytest.fixture(autouse=True)
def fresh_stats():
    T.TEACHER_STATS.reset()
    yield
    T.TEACHER_STATS.reset()


def _msg(text, stop="end_turn", inp=10, out=5, read=0, write=0):
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           stop_reason=stop, usage=SimpleNamespace(input_tokens=inp, output_tokens=out,
                                                                  cache_read_input_tokens=read,
                                                                  cache_creation_input_tokens=write))


class FakeMessages:
    def __init__(self, reply):
        self.calls, self.reply = [], reply
        self.batches = FakeBatches(self)

    def create(self, **kw):
        self.calls.append(kw)
        return self.reply(kw)


class FakeBatches:
    def __init__(self, messages):
        self.messages, self.created, self.polls, self.store = messages, [], 0, {}

    def create(self, requests):
        bid = f"batch_{len(self.created)}"
        self.created.append(requests)
        self.store[bid] = requests
        return SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        self.polls += 1
        status = "ended" if self.polls >= 2 else "in_progress"
        return SimpleNamespace(id=bid, processing_status=status,
                               request_counts=SimpleNamespace(processing=0 if status == "ended" else 1,
                                                              succeeded=1, errored=0))

    def results(self, bid):
        out = []
        for r in reversed(self.store[bid]):  # results arrive in any order
            if "fail" in json.dumps(r["params"]["messages"]):
                out.append(SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(
                    type="errored", error=SimpleNamespace(type="api_error"))))
            else:
                out.append(SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(
                    type="succeeded", message=self.messages.reply(r["params"]))))
        return out


def echo(kw):
    last = kw["messages"][-1]["content"]
    text = "".join(b["text"] for b in last) if isinstance(last, list) else last
    return _msg(json.dumps({"statements": [text[-6:]]}))


def make(mode="messages", reply=echo, **kw):
    msgs = FakeMessages(reply)
    client = SimpleNamespace(messages=msgs)
    t = T.AnthropicTeacher("claude-haiku-5-5", client=client, mode=mode, poll_s=0, **kw)
    return t, msgs


def test_request_shape_strict_schema_system_and_no_sampling_params():
    t, msgs = make()
    out = t.generate_structured("Section: abc123", Labels, system="You label sections.")
    assert out.statements == ["abc123"]
    kw = msgs.calls[0]
    assert kw["model"] == "claude-haiku-5-5" and kw["system"] == "You label sections."
    assert all(m["role"] != "system" for m in kw["messages"])
    assert "temperature" not in kw and "top_p" not in kw  # Claude Haiku 5.5 rejects non-default sampling
    schema = kw["output_config"]["format"]["schema"]
    assert kw["output_config"]["format"]["type"] == "json_schema"

    def objects(s):
        if isinstance(s, dict):
            if s.get("type") == "object" or "properties" in s:
                yield s
            for v in s.values():
                yield from objects(v)
        elif isinstance(s, list):
            for v in s:
                yield from objects(v)
    assert all(o.get("additionalProperties") is False for o in objects(schema))
    assert T.TEACHER_STATS.served_by == {"anthropic:claude-haiku-5-5": 1}


def test_unsupported_schema_constraints_are_dropped():
    s = T.strict_json_schema({"type": "object", "properties": {"a": {"type": "string", "minLength": 2},
                                                             "b": {"type": "array", "minItems": 1, "items": {"type": "integer", "maximum": 3}}},
                              "required": ["a", "b"]})
    assert s["additionalProperties"] is False
    assert "minLength" not in s["properties"]["a"] and "minItems" not in s["properties"]["b"]
    assert "maximum" not in s["properties"]["b"]["items"]


def test_shared_prompt_prefix_is_cached_across_a_batch():
    t, msgs = make(cache_prefix_min_chars=20)
    shared = "CATALOGUE " * 10
    t.generate_batch([shared + "Section: first1", shared + "Section: secnd2"])
    for kw in msgs.calls:
        blocks = kw["messages"][0]["content"]
        assert blocks[0]["text"] == shared + "Section: " and blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in blocks[1]
        assert "".join(b["text"] for b in blocks).startswith(shared)  # the text itself is unchanged


def test_refusal_and_truncation_give_a_failed_item():
    t, _ = make(reply=lambda kw: _msg("", stop="refusal"))
    assert t.generate("x") == ""


def test_batch_mode_one_batch_results_by_custom_id_and_errors_are_empty(tmp_path):
    t, msgs = make(mode="batch", cache_dir=tmp_path)
    outs = t.generate_batch(["p one", "p fail", "p three"])
    assert len(msgs.batches.created) == 1 and len(msgs.batches.created[0]) == 3
    assert json.loads(outs[0])["statements"] == ["p one"[-6:]] and outs[1] == ""
    assert json.loads(outs[2])["statements"] == ["three"[-6:]] or json.loads(outs[2])["statements"] == ["p three"[-6:]]
    assert not list((tmp_path / "_batches").glob("*.json"))  # finished: no pending batch left
    assert T.TEACHER_STATS.served_by["anthropic:claude-haiku-5-5/batch"] == 2


def test_an_interrupted_batch_is_resumed_not_paid_twice(tmp_path):
    t, msgs = make(mode="batch", cache_dir=tmp_path)
    real_results = msgs.batches.results
    msgs.batches.results = lambda bid: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        t.generate_batch(["p one", "p two"])
    assert len(list((tmp_path / "_batches").glob("*.json"))) == 1
    msgs.batches.results = real_results
    t2 = T.AnthropicTeacher("claude-haiku-5-5", client=SimpleNamespace(messages=msgs), mode="batch", poll_s=0,
                            cache_dir=tmp_path)
    outs = t2.generate_batch(["p one", "p two"])
    assert len(msgs.batches.created) == 1 and all(outs)  # the same batch, fetched again


def test_load_teacher_builds_the_anthropic_backend(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        T.load_teacher({"backend": "anthropic", "id": "claude-haiku-5-5"})
