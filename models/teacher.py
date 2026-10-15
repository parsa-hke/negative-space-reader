"""Teacher LLM clients with structured output, and the prompt registry (plan §0.2 #6, §1.6).

Four backends behind one `Teacher` interface, chosen by the compute overlay (plan §0.2 #6):
  - `VLLMTeacher`         in-process vLLM engine for batch jobs on the GPU server (schema induction,
                          labels); structured output via vLLM's xgrammar backend
                          (set models.teacher.structured_backend: outlines to fall back).
  - `OpenAITeacher`       any OpenAI-compatible endpoint: `vllm serve` on the GPU server, OpenRouter
                          (configs/openrouter.yaml, provider pinned via `provider`), or API models for
                          the baselines (plan §0.2 #17). Keys come from an env var. Records which
                          provider served each call.
  - `MLXTeacher`          Qwen2.5-7B-Instruct 4-bit under MLX on the Mac (configs/local.yaml) for prompt
                          iteration and coverage checks; structured output via Outlines' MLX adapter.
  - `TransformersTeacher` the 1.5B stand-in for smoke runs and tests (configs/smoke.yaml); structured
                          output via xgrammar's HF logits processor, Outlines as the fallback.

Every backend can cache responses on disk (`cache_dir`), keyed by model, backend, provider, messages,
schema and sampling settings: a rerun is free and byte-identical, and the cache is a releasable artefact.
`TEACHER_STATS` counts calls, cache hits and served providers; experiments/run.py records it per run.

    teacher = load_teacher(cfg["models"]["teacher"], device="mps")
    prompt = PROMPTS.get("smoke_structured")
    answer = teacher.generate_structured(prompt.render(text=...), MyPydanticModel)

Prompts live in prompts/<id>.md with a YAML front-matter header (`id`, `version`, `description`)
and a `{{placeholder}}` body. Each prompt's hash is the sha256 of the whole file (header included);
every prompt rendered during a run is recorded, and experiments/run.py writes the hashes to
metrics.json and runs/log.csv.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


# ============================================================================ prompt registry
class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    description: str
    template: str
    hash: str
    path: Path
    variables: frozenset[str] = field(default_factory=frozenset)

    @property
    def tag(self) -> str:
        return f"v{self.version}:{self.hash}"

    def render(self, **values: Any) -> str:
        missing = self.variables - values.keys()
        if missing:
            raise PromptError(f"prompt {self.id!r} is missing values for {sorted(missing)}")
        PROMPTS.record(self)
        return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), self.template)


def parse_prompt_file(path: Path) -> Prompt:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        raise PromptError(f"{path}: missing '---' YAML front-matter header")
    header = yaml.safe_load(m.group(1)) or {}
    for key in ("id", "version"):
        if key not in header:
            raise PromptError(f"{path}: header has no {key!r}")
    if header["id"] != path.stem:
        raise PromptError(f"{path}: header id {header['id']!r} does not match the file name")
    template = m.group(2).strip("\n") + "\n"
    return Prompt(
        id=header["id"],
        version=str(header["version"]),
        description=str(header.get("description", "")),
        template=template,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
        path=path,
        variables=frozenset(_PLACEHOLDER.findall(template)),
    )


class PromptRegistry:
    """Reads prompts/*.md; tracks which prompt versions were rendered since the last reset."""

    def __init__(self, directory: Path = PROMPTS_DIR):
        self.directory = Path(directory)
        self._used: dict[str, str] = {}

    def get(self, prompt_id: str) -> Prompt:
        path = self.directory / f"{prompt_id}.md"
        if not path.is_file():
            raise PromptError(f"no prompt {prompt_id!r} in {self.directory}")
        return parse_prompt_file(path)  # re-read every time: edits are never served stale

    def all(self) -> dict[str, Prompt]:
        return {p.stem: parse_prompt_file(p) for p in sorted(self.directory.glob("*.md"))}

    def record(self, prompt: Prompt) -> None:
        self._used[prompt.id] = prompt.tag

    def usage(self) -> dict[str, str]:
        return dict(self._used)

    def reset_usage(self) -> None:
        self._used.clear()


PROMPTS = PromptRegistry()


def schema_json(model: type[BaseModel]) -> str:
    """The JSON schema of a pydantic model, for prompts that show the expected output format."""
    return json.dumps(model.model_json_schema(), indent=2)


# ============================================================================ teachers
class TeacherStats:
    """Per-run counters: calls, cache hits, tokens where known, and the provider that served each call."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.served_by: dict[str, int] = {}

    def record(self, *, served_by: str, cached: bool, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.calls += 1
        self.cache_hits += cached
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.served_by[served_by] = self.served_by.get(served_by, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {"calls": self.calls, "cache_hits": self.cache_hits, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "served_by": dict(self.served_by)}


TEACHER_STATS = TeacherStats()


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, raw: str):
        super().__init__(f"{message}\n--- raw output ---\n{raw[:2000]}")
        self.raw = raw


class Teacher(ABC):
    """Common interface. Subclasses implement `_complete` (batched chat completion, optionally JSON-constrained)."""

    backend = "abstract"

    def __init__(self, model_id: str, *, temperature: float = 0.0, top_p: float = 1.0,
                 max_tokens: int = 1024, max_retries: int = 2, cache_dir: str | Path | None = None):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @abstractmethod
    def _complete(self, conversations: list[list[dict[str, str]]], *, json_schema: dict[str, Any] | None,
                  max_tokens: int) -> list[str]:
        """One completion string per conversation. Backends call `self._served(...)` once per completion."""

    # -- provenance and cache -----------------------------------------------------------------
    def _route(self) -> str:
        """Identifies the model build behind this teacher (part of the cache key and the stats)."""
        return f"{self.backend}:{self.model_id}"

    def _served(self, *, served_by: str | None = None, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        TEACHER_STATS.record(served_by=served_by or self._route(), cached=False,
                             prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    def _cache_key(self, conversation: list[dict[str, str]], json_schema: dict[str, Any] | None,
                   max_tokens: int) -> str:
        blob = json.dumps({"route": self._route(), "messages": conversation, "schema": json_schema,
                           "max_tokens": max_tokens, "temperature": self.temperature, "top_p": self.top_p},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _complete_cached(self, conversations: list[list[dict[str, str]]], *, json_schema: dict[str, Any] | None,
                         max_tokens: int) -> list[str]:
        if self.cache_dir is None:
            return self._complete(conversations, json_schema=json_schema, max_tokens=max_tokens)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        keys = [self._cache_key(c, json_schema, max_tokens) for c in conversations]
        outputs: list[str | None] = [None] * len(conversations)
        missing = []
        for i, key in enumerate(keys):
            path = self.cache_dir / f"{key}.json"
            if path.is_file():
                outputs[i] = json.loads(path.read_text(encoding="utf-8"))["output"]
                TEACHER_STATS.record(served_by="cache", cached=True)
            else:
                missing.append(i)
        if missing:
            fresh = self._complete([conversations[i] for i in missing], json_schema=json_schema, max_tokens=max_tokens)
            for i, out in zip(missing, fresh):
                outputs[i] = out
                (self.cache_dir / f"{keys[i]}.json").write_text(json.dumps(
                    {"route": self._route(), "messages": conversations[i], "schema": json_schema,
                     "max_tokens": max_tokens, "temperature": self.temperature, "output": out},
                    ensure_ascii=False), encoding="utf-8")
        return outputs  # type: ignore[return-value]

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]

    def generate(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> str:
        return self.generate_batch([prompt], system=system, max_tokens=max_tokens)[0]

    def generate_batch(self, prompts: list[str], *, system: str | None = None,
                       max_tokens: int | None = None) -> list[str]:
        convs = [self._messages(p, system) for p in prompts]
        return self._complete_cached(convs, json_schema=None, max_tokens=max_tokens or self.max_tokens)

    def generate_structured(self, prompt: str, output_model: type[T], *, system: str | None = None,
                            max_tokens: int | None = None) -> T:
        return self.generate_structured_batch([prompt], output_model, system=system, max_tokens=max_tokens)[0]

    def generate_structured_batch(self, prompts: list[str], output_model: type[T], *,
                                  system: str | None = None, max_tokens: int | None = None) -> list[T]:
        """Constrained generation + pydantic validation. Items that fail validation (e.g. truncated
        at max_tokens) are retried with a doubled token budget, up to max_retries times."""
        schema = output_model.model_json_schema()
        budget = max_tokens or self.max_tokens
        results: list[T | None] = [None] * len(prompts)
        raws: list[str] = [""] * len(prompts)
        pending = list(range(len(prompts)))
        for _attempt in range(self.max_retries + 1):
            convs = [self._messages(prompts[i], system) for i in pending]
            outputs = self._complete_cached(convs, json_schema=schema, max_tokens=budget)
            still = []
            for i, raw in zip(pending, outputs):
                raws[i] = raw
                try:
                    results[i] = output_model.model_validate_json(_strip_fences(raw))
                except ValidationError:
                    still.append(i)
            pending = still
            if not pending:
                break
            budget = min(budget * 2, 2 * (max_tokens or self.max_tokens))  # one doubling: truncation, not loops
        if pending:
            i = pending[0]
            raise StructuredOutputError(
                f"{len(pending)}/{len(prompts)} outputs failed {output_model.__name__} validation "
                f"after {self.max_retries + 1} attempts (model {self.model_id})", raws[i])
        return results  # type: ignore[return-value]


def _strip_fences(text: str) -> str:
    """Tolerate ```json fences from unconstrained API models; constrained output has none."""
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n(.*?)\n```$", text, re.DOTALL)
    return m.group(1) if m else text


# ---------------------------------------------------------------------------- transformers (stand-in)
class TransformersTeacher(Teacher):
    backend = "transformers"

    def __init__(self, model_id: str, *, device: str = "cpu", dtype: str = "bfloat16",
                 revision: str | None = None, structured_backend: str = "xgrammar",
                 structured_fallback: str | None = "outlines", **kw: Any):
        super().__init__(model_id, **kw)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype) if device != "cpu" else torch.float32
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch_dtype)
        self.model.to(device).eval()
        self.structured_backend = self._pick_backend(structured_backend, structured_fallback)
        self._grammar_cache: dict[str, Any] = {}

    @staticmethod
    def _pick_backend(primary: str, fallback: str | None) -> str:
        for name in (primary, fallback):
            if name is None:
                continue
            try:
                __import__(name)
                return name
            except ImportError:
                continue
        raise ImportError(f"neither {primary!r} nor {fallback!r} is installed")

    def _compiled_grammar(self, schema: dict[str, Any]) -> Any:
        import xgrammar as xgr

        key = json.dumps(schema, sort_keys=True)
        if key not in self._grammar_cache:
            if not self._grammar_cache:
                info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.model.config.vocab_size)
                self._compiler = xgr.GrammarCompiler(info)
            self._grammar_cache[key] = self._compiler.compile_json_schema(key)
        return self._grammar_cache[key]

    def _complete(self, conversations, *, json_schema, max_tokens):
        import torch

        texts = [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                 for c in conversations]
        if json_schema is not None and self.structured_backend == "outlines":
            outs = [self._complete_outlines(t, json_schema, max_tokens) for t in texts]
            for _ in outs:
                self._served()
            return outs
        processors = []
        if json_schema is not None:
            from xgrammar.contrib.hf import LogitsProcessor

            processors = [LogitsProcessor(self._compiled_grammar(json_schema))]  # one per generate() call
        batch = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**batch, max_new_tokens=max_tokens, logits_processor=processors,
                                      pad_token_id=self.tokenizer.pad_token_id, **self._sampling())
        new_tokens = out[:, batch["input_ids"].shape[1]:]
        n_prompt = int(batch["attention_mask"].sum(dim=1).float().mean().item())
        for row in new_tokens:
            self._served(prompt_tokens=n_prompt, completion_tokens=int((row != self.tokenizer.pad_token_id).sum()))
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def _complete_outlines(self, text: str, json_schema: dict[str, Any], max_tokens: int) -> str:
        import outlines
        from outlines.types import JsonSchema

        if not hasattr(self, "_outlines_model"):
            self._outlines_model = outlines.from_transformers(self.model, self.tokenizer)
        generator = outlines.Generator(self._outlines_model, JsonSchema(json_schema))
        return generator(text, max_new_tokens=max_tokens, **self._sampling())

    def _sampling(self) -> dict[str, Any]:
        """Override the checkpoint's generation_config (Qwen samples by default): greedy at temperature 0."""
        if self.temperature > 0:
            return {"do_sample": True, "temperature": self.temperature, "top_p": self.top_p}
        return {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}


# ---------------------------------------------------------------------------- MLX (Mac, local tier)
class MLXTeacher(Teacher):
    """Qwen2.5-7B-Instruct 4-bit under mlx-lm (configs/local.yaml). Measured 24 tok/s on the M5.
    Structured output through Outlines' MLX adapter (an outlines_core logits processor)."""

    backend = "mlx"

    def __init__(self, model_id: str, **kw: Any):
        super().__init__(model_id, **kw)
        from mlx_lm import load

        self.model, self.tokenizer = load(model_id)
        self._outlines_model = None

    def _sampler(self) -> Any:
        from mlx_lm.sample_utils import make_sampler

        # temp 0 is greedy in mlx-lm; top_p=0 disables nucleus sampling
        return make_sampler(temp=self.temperature, top_p=self.top_p if self.temperature > 0 else 0.0)

    def _complete(self, conversations, *, json_schema, max_tokens):
        from mlx_lm import generate

        outs = []
        for conv in conversations:
            text = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            if json_schema is None:
                out = generate(self.model, self.tokenizer, prompt=text, max_tokens=max_tokens, sampler=self._sampler())
            else:
                import outlines
                from outlines.types import JsonSchema

                if self._outlines_model is None:
                    self._outlines_model = outlines.from_mlxlm(self.model, self.tokenizer)
                generator = outlines.Generator(self._outlines_model, JsonSchema(json_schema))
                out = generator(text, max_tokens=max_tokens, sampler=self._sampler())
            self._served(prompt_tokens=len(self.tokenizer.encode(text)), completion_tokens=len(self.tokenizer.encode(out)))
            outs.append(out)
        return outs


# ---------------------------------------------------------------------------- vLLM (GPU server)
class VLLMTeacher(Teacher):
    """In-process vLLM engine (API checked against vllm 0.31.0: `structured_outputs_config` on the
    engine, `SamplingParams(structured_outputs=StructuredOutputsParams(json=...))` per request)."""

    backend = "vllm"

    def __init__(self, model_id: str, *, revision: str | None = None, dtype: str = "auto",
                 quantization: str | None = None, tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9, max_model_len: int | None = None,
                 structured_backend: str = "xgrammar", seed: int = 0, **kw: Any):
        super().__init__(model_id, **kw)
        from vllm import LLM

        self.llm = LLM(model=model_id, revision=revision, dtype=dtype, quantization=quantization,
                       tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       seed=seed, structured_outputs_config={"backend": structured_backend})

    def _complete(self, conversations, *, json_schema, max_tokens):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        structured = StructuredOutputsParams(json=json_schema) if json_schema is not None else None
        params = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=max_tokens,
                                structured_outputs=structured)
        outputs = self.llm.chat(conversations, params, use_tqdm=len(conversations) > 32)
        for o in outputs:
            self._served(prompt_tokens=len(o.prompt_token_ids or ()), completion_tokens=len(o.outputs[0].token_ids))
        return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------- OpenAI-compatible
class OpenAITeacher(Teacher):
    """Any OpenAI-compatible chat endpoint: `vllm serve` or an API model (baselines).
    response_format: json_schema (constrained, vLLM and OpenAI) | json_object | none (prompt-only;
    output is still validated). `provider` is OpenRouter's routing object (order, allow_fallbacks,
    data_collection), sent as extra_body; the provider that served each call is recorded."""

    backend = "openai"

    def __init__(self, model_id: str, *, base_url: str | list[str] | None = None, api_key_env: str | None = None,
                 response_format: str = "json_schema", request_timeout_s: float = 120,
                 concurrency: int = 8, provider: dict[str, Any] | None = None, served_as: str | None = None,
                 client: Any = None, **kw: Any):
        super().__init__(model_id, **kw)
        urls = base_url if isinstance(base_url, list) else [base_url]
        self.base_url = urls[0]
        self.provider = provider
        self.served_as = served_as  # e.g. "llama.cpp b11280 Qwen2.5-32B-Instruct q4_k_m": the build, not the port
        if client is None:
            from openai import OpenAI

            api_key = os.environ.get(api_key_env, "") if api_key_env else ""
            if api_key_env and not api_key:
                raise RuntimeError(f"environment variable {api_key_env} is not set")
            clients = [OpenAI(base_url=u, api_key=api_key or "EMPTY", timeout=request_timeout_s) for u in urls]
        else:
            clients = client if isinstance(client, list) else [client]
        self.clients = clients
        self.client = clients[0]
        self._next = 0
        self.response_format = response_format
        self.concurrency = concurrency

    def _route(self) -> str:
        if self.served_as:
            return f"openai:{self.served_as}"
        base = self.base_url or str(getattr(self.client, "base_url", "") or "api.openai.com")
        host = base.split("//")[-1].split("/")[0]
        pinned = ",".join(self.provider.get("order", [])) if self.provider else ""
        return f"openai:{host}:{self.model_id}" + (f"@{pinned}" if pinned else "")

    def _pick_client(self) -> Any:
        """Round-robin over the configured servers (e.g. one llama.cpp server per GPU)."""
        c = self.clients[self._next % len(self.clients)]
        self._next += 1
        return c

    def _one(self, messages: list[dict[str, str]], json_schema: dict[str, Any] | None, max_tokens: int) -> str:
        kwargs: dict[str, Any] = {}
        if self.provider:
            kwargs["extra_body"] = {"provider": self.provider}
        if json_schema is not None and self.response_format == "json_schema":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": str(json_schema.get("title", "output")), "schema": json_schema, "strict": False}}
        elif json_schema is not None and self.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        import openai

        try:
            resp = self._pick_client().chat.completions.create(
                model=self.model_id, messages=messages, temperature=self.temperature, top_p=self.top_p,
                max_completion_tokens=max_tokens, **kwargs)
        except openai.BadRequestError as e:  # e.g. prompt + budget beyond a slot's context: a failed item, not a crash
            print(f"[teacher] request rejected: {str(e)[:200]}", flush=True)
            self._served(served_by=f"{self._route()}/rejected")
            return ""
        usage = getattr(resp, "usage", None)
        extra = getattr(resp, "model_extra", None) or {}
        served = extra.get("provider") or getattr(resp, "provider", None)  # OpenRouter adds `provider`
        self._served(served_by=f"{self._route()}/{served}" if served else None,
                     prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                     completion_tokens=getattr(usage, "completion_tokens", 0) or 0)
        return resp.choices[0].message.content or ""

    def _complete(self, conversations, *, json_schema, max_tokens):
        if len(conversations) == 1:
            return [self._one(conversations[0], json_schema, max_tokens)]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(lambda c: self._one(c, json_schema, max_tokens), conversations))


# ---------------------------------------------------------------------------- factory
def load_teacher(tcfg: dict[str, Any], *, device: str | None = None, seed: int = 0) -> Teacher:
    """Build a teacher from the `models.teacher` config block."""
    common = {k: tcfg[k] for k in ("temperature", "top_p", "max_tokens", "max_retries") if k in tcfg}
    if tcfg.get("cache", False):
        common["cache_dir"] = tcfg.get("cache_dir", ".cache/negspace/teacher")
    backend = tcfg["backend"]
    if backend == "transformers":
        return TransformersTeacher(tcfg["id"], device=device or "cpu", dtype=tcfg.get("dtype", "bfloat16"),
                                   revision=tcfg.get("revision"),
                                   structured_backend=tcfg.get("structured_backend", "xgrammar"),
                                   structured_fallback=tcfg.get("structured_fallback"), **common)
    if backend == "mlx":
        return MLXTeacher(tcfg["id"], **common)
    if backend == "vllm":
        return VLLMTeacher(tcfg["id"], revision=tcfg.get("revision"), dtype=tcfg.get("dtype", "auto"),
                           quantization=tcfg.get("quantization"),
                           tensor_parallel_size=tcfg.get("tensor_parallel_size", 1),
                           gpu_memory_utilization=tcfg.get("gpu_memory_utilization", 0.9),
                           max_model_len=tcfg.get("max_model_len"),
                           structured_backend=tcfg.get("structured_backend", "xgrammar"), seed=seed, **common)
    if backend == "openai":
        return OpenAITeacher(tcfg["id"], base_url=tcfg.get("base_url"), api_key_env=tcfg.get("api_key_env"),
                             response_format=tcfg.get("response_format", "json_schema"),
                             request_timeout_s=tcfg.get("request_timeout_s", 120),
                             concurrency=tcfg.get("concurrency", 8), provider=tcfg.get("provider"),
                             served_as=tcfg.get("served_as"), **common)
    raise ValueError(f"unknown teacher backend {backend!r} (transformers | mlx | vllm | openai)")
