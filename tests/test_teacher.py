"""models/teacher.py: prompt registry, the four backends, the response cache, and a structured round trip
on the stand-in (plan §1.6–1.7)."""

import importlib.util
import json
import sys
import types
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from models import teacher as T


class Retention(BaseModel):
    topic: str
    mentions_retention: bool
    retention_days: int | None


@pytest.fixture(autouse=True)
def fresh_stats():
    T.TEACHER_STATS.reset()
    yield
    T.TEACHER_STATS.reset()


# ------------------------------------------------------------------------------------ prompt registry
def write_prompt(directory, pid="p", version=1, body="Say {{ word }} about {{thing}}."):
    path = directory / f"{pid}.md"
    path.write_text(f"---\nid: {pid}\nversion: {version}\ndescription: test\n---\n{body}\n")
    return path


def test_prompt_header_hash_and_render(tmp_path):
    write_prompt(tmp_path)
    reg = T.PromptRegistry(tmp_path)
    p = reg.get("p")
    assert p.version == "1" and p.variables == {"word", "thing"} and len(p.hash) == 12
    assert p.render(word="hi", thing="x") == "Say hi about x.\n"
    with pytest.raises(T.PromptError, match="thing"):
        p.render(word="hi")


def test_prompt_hash_tracks_content_and_version(tmp_path):
    write_prompt(tmp_path)
    h1 = T.PromptRegistry(tmp_path).get("p").hash
    write_prompt(tmp_path, version=2)
    h2 = T.PromptRegistry(tmp_path).get("p").hash
    write_prompt(tmp_path, version=2, body="Say {{word}} about {{thing}}!")
    h3 = T.PromptRegistry(tmp_path).get("p").hash
    assert len({h1, h2, h3}) == 3


def test_prompt_header_is_required(tmp_path):
    (tmp_path / "bad.md").write_text("no header {{x}}")
    with pytest.raises(T.PromptError, match="front-matter"):
        T.PromptRegistry(tmp_path).get("bad")
    write_prompt(tmp_path, pid="other").rename(tmp_path / "renamed.md")
    with pytest.raises(T.PromptError, match="does not match"):
        T.PromptRegistry(tmp_path).get("renamed")


def test_repo_prompts_parse_and_usage_is_recorded():
    prompts = T.PROMPTS.all()
    assert "smoke_structured" in prompts
    T.PROMPTS.reset_usage()
    p = prompts["smoke_structured"]
    p.render(text="x")
    assert T.PROMPTS.usage() == {"smoke_structured": p.tag}


# ------------------------------------------------------------------------------------ OpenAI-compatible
GOOD = '{"topic": "retention", "mentions_retention": true, "retention_days": 30}'


def chat_response(content, provider=None):
    body = {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
    if provider:
        body["provider"] = provider  # OpenRouter adds the provider that served the call
    return body


def mock_client(handler):
    from openai import OpenAI

    return OpenAI(api_key="test", base_url="http://test/v1",
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_openai_backend_sends_json_schema_and_retries_truncation():
    requests, replies = [], ['{"topic": "reten', GOOD]

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response(replies[len(requests) - 1]))

    teacher = T.OpenAITeacher("served-model", client=mock_client(handler), max_tokens=64)
    out = teacher.generate_structured("prompt", Retention, system="sys")
    assert out == Retention(topic="retention", mentions_retention=True, retention_days=30)
    first, second = requests
    assert first["model"] == "served-model" and first["temperature"] == 0.0 and "provider" not in first
    assert first["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "prompt"}]
    assert first["response_format"]["type"] == "json_schema"
    assert first["response_format"]["json_schema"]["schema"] == Retention.model_json_schema()
    assert (first["max_completion_tokens"], second["max_completion_tokens"]) == (64, 128)
    assert T.TEACHER_STATS.summary() == {"calls": 2, "cache_hits": 0, "prompt_tokens": 22, "completion_tokens": 14,
                                         "served_by": {"openai:test:served-model": 2}}


def test_openrouter_provider_pin_is_sent_and_served_provider_recorded():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response(GOOD, provider="DeepInfra"))

    from experiments.config import load_config

    tcfg = load_config("configs/base.yaml", ["openrouter"])["models"]["teacher"]
    teacher = T.OpenAITeacher(tcfg["id"], base_url=tcfg["base_url"], provider=tcfg["provider"],
                              client=mock_client(handler))
    teacher.generate_structured("prompt", Retention)
    assert requests[0]["provider"] == {"order": ["deepinfra/fp8"], "allow_fallbacks": False, "data_collection": "deny"}
    served = T.TEACHER_STATS.summary()["served_by"]
    assert served == {"openai:openrouter.ai:qwen/qwen-2.5-72b-instruct@deepinfra/fp8/DeepInfra": 1}


def test_response_cache_makes_reruns_free(tmp_path):
    requests = []

    def handler(request):
        requests.append(1)
        return httpx.Response(200, json=chat_response(GOOD))

    teacher = T.OpenAITeacher("m", client=mock_client(handler), cache_dir=tmp_path / "cache")
    a = teacher.generate_structured("prompt", Retention)
    b = teacher.generate_structured("prompt", Retention)
    assert a == b and len(requests) == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
    teacher.generate_structured("another prompt", Retention)
    teacher.generate("prompt")  # unconstrained: a different cache key
    assert len(requests) == 3
    assert T.TEACHER_STATS.summary()["cache_hits"] == 1
    cached = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert cached["output"] == GOOD and cached["route"] == "openai:test:m"


def test_structured_failure_raises_with_raw_output():
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="nope"))]))))
    teacher = T.OpenAITeacher("m", client=client, max_retries=1)
    with pytest.raises(T.StructuredOutputError, match="nope"):
        teacher.generate_structured("p", Retention)


def test_missing_api_key_env_fails_loudly(monkeypatch):
    monkeypatch.delenv("NEGSPACE_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NEGSPACE_TEST_KEY"):
        T.load_teacher({"backend": "openai", "id": "m", "api_key_env": "NEGSPACE_TEST_KEY"})


# ------------------------------------------------------------------------------------ vLLM (faked)
def test_vllm_backend_call_shape(monkeypatch):
    """vLLM does not install on macOS; check our calls against the 0.31.0 API surface with a fake."""
    calls = {}

    class LLM:
        def __init__(self, *, model, revision, dtype, quantization, tensor_parallel_size,
                     gpu_memory_utilization, max_model_len, seed, structured_outputs_config):
            calls["init"] = dict(model=model, quantization=quantization, structured=structured_outputs_config)

        def chat(self, conversations, params, use_tqdm):
            calls["chat"] = (conversations, params)
            return [SimpleNamespace(prompt_token_ids=[1, 2, 3],
                                    outputs=[SimpleNamespace(text='{"topic": "t", "mentions_retention": false, '
                                                                  '"retention_days": null}', token_ids=[4, 5])])
                    for _ in conversations]

    class SamplingParams:
        def __init__(self, *, temperature, top_p, max_tokens, structured_outputs):
            self.max_tokens, self.structured_outputs = max_tokens, structured_outputs

    class StructuredOutputsParams:
        def __init__(self, *, json):
            self.json = json

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
    sampling = types.ModuleType("vllm.sampling_params")
    sampling.StructuredOutputsParams = StructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)

    from experiments.config import load_config

    tcfg = load_config("configs/base.yaml", ["gpu"])["models"]["teacher"]
    tcfg["cache"] = False
    teacher = T.load_teacher(tcfg)
    assert isinstance(teacher, T.VLLMTeacher)
    assert calls["init"] == {"model": "Qwen/Qwen2.5-72B-Instruct-AWQ", "quantization": "awq",
                             "structured": {"backend": "xgrammar"}}
    outs = teacher.generate_structured_batch(["a", "b"], Retention)
    assert [o.topic for o in outs] == ["t", "t"]
    convs, params = calls["chat"]
    assert len(convs) == 2 and params.structured_outputs.json == Retention.model_json_schema()
    assert T.TEACHER_STATS.summary()["served_by"] == {"vllm:Qwen/Qwen2.5-72B-Instruct-AWQ": 2}


# ------------------------------------------------------------------------------------ local models
def hub_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:
        return False


STAND_IN = "Qwen/Qwen2.5-1.5B-Instruct"
MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
PASSAGE = "We keep your account data for 90 days after you close your account, then delete it."


@pytest.fixture(scope="module")
def stand_in():
    if not hub_cached(STAND_IN):
        pytest.skip(f"{STAND_IN} is not in the HF cache; run: hf download {STAND_IN}")
    from experiments.config import load_config
    from experiments.run import resolve_device

    tcfg = load_config("configs/base.yaml", ["local", "smoke"])["models"]["teacher"]
    assert tcfg["id"] == STAND_IN and tcfg["backend"] == "transformers"
    tcfg["max_tokens"], tcfg["cache"] = 96, False
    return T.load_teacher(tcfg, device=resolve_device("auto"))


@pytest.mark.model
def test_structured_round_trip_on_stand_in(stand_in):
    prompt = T.PROMPTS.get("smoke_structured").render(text=PASSAGE)
    out = stand_in.generate_structured(prompt, Retention)
    assert isinstance(out, Retention)
    assert out.mentions_retention is True and out.retention_days == 90
    assert T.TEACHER_STATS.summary()["served_by"] == {f"transformers:{STAND_IN}": 1}


@pytest.fixture(scope="module")
def mlx_teacher():
    if importlib.util.find_spec("mlx_lm") is None:
        pytest.skip("mlx-lm is not installed (macOS only: pip install -e '.[mac]')")
    if not hub_cached(MLX_MODEL):
        pytest.skip(f"{MLX_MODEL} is not in the HF cache; run: hf download {MLX_MODEL}")
    from experiments.config import load_config

    tcfg = load_config("configs/base.yaml", ["local"])["models"]["teacher"]
    assert tcfg["id"] == MLX_MODEL and tcfg["backend"] == "mlx"
    tcfg["max_tokens"], tcfg["cache"] = 96, False
    return T.load_teacher(tcfg)


@pytest.mark.model
def test_structured_round_trip_on_mlx_local_teacher(mlx_teacher):
    prompt = T.PROMPTS.get("smoke_structured").render(text=PASSAGE)
    out = mlx_teacher.generate_structured(prompt, Retention)
    assert out.mentions_retention is True and out.retention_days == 90
    assert T.TEACHER_STATS.summary()["served_by"] == {f"mlx:{MLX_MODEL}": 1}


def test_openai_backend_round_robin_route_label_and_rejections():
    hits = {"a": 0, "b": 0}

    def handler_for(name, status=200):
        def handler(request):
            hits[name] += 1
            if status != 200:
                return httpx.Response(status, json={"error": {"message": "context exceeded"}})
            return httpx.Response(200, json=chat_response(GOOD))
        return handler

    teacher = T.OpenAITeacher("label", client=[mock_client(handler_for("a")), mock_client(handler_for("b"))],
                              served_as="llama.cpp-test/model-q4", concurrency=4)
    teacher.generate_structured_batch(["p1", "p2", "p3", "p4"], Retention)
    assert hits == {"a": 2, "b": 2} and teacher._route() == "openai:llama.cpp-test/model-q4"
    bad = T.OpenAITeacher("label", client=mock_client(handler_for("a", status=400)), max_retries=1)
    with pytest.raises(T.StructuredOutputError):  # a rejected request is a failed item, never a crash
        bad.generate_structured("p", Retention)
    assert T.TEACHER_STATS.summary()["served_by"].get("openai:test:label/rejected") == 2


def test_rate_limits_are_waited_out_not_fatal():
    """2026-10-17: the 72B run died on one 429 from OpenRouter's shared DeepInfra pool. Rate limits and transient
    server errors are now retried with back-off; only after rate_limit_retries attempts does the error surface."""
    from openai import OpenAI

    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(429, json={"error": {"message": "temporarily rate-limited upstream", "code": 429}})
        return httpx.Response(200, json=chat_response(GOOD))

    client = OpenAI(api_key="t", base_url="http://test/v1", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    teacher = T.OpenAITeacher("m", client=client, rate_limit_retries=3, rate_limit_wait_s=0)
    assert teacher.generate_structured("p", Retention).retention_days == 30 and len(calls) == 3

    calls.clear()
    stubborn = T.OpenAITeacher("m", client=client, rate_limit_retries=1, rate_limit_wait_s=0)
    with pytest.raises(Exception, match="429|rate"):
        stubborn.generate("p")
