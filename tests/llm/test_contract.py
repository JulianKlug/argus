"""Contract suite on every configured backend (criteria 2, 3, 5, 6, 9). Synthetic prompts only.

Run: uv run pytest -m llm tests/llm/test_contract.py
"""

import json
import tomllib

import httpx
import pytest
from pydantic import BaseModel

from argus.llm import (
    CallContext, Capability, CapabilityError, ContextOverflowError, FinishReason, LlmClient, Message, Request,
    Role, ToolSpec,
)
from argus.llm import client as client_module
from argus.llm.config import DEFAULT_CONFIG
from argus.llm.sink import RECORD_FILE, read_records

OLLAMA = "http://127.0.0.1:11434"


def _ollama_up() -> bool:
    try:
        return httpx.get(f"{OLLAMA}/api/version", timeout=2, trust_env=False).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [pytest.mark.llm, pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")]

BACKENDS_TOML = tomllib.loads(DEFAULT_CONFIG.read_text())["backends"]
BACKENDS = sorted(BACKENDS_TOML)
CTX = CallContext("contract", "step")
MAX_TOKENS = 1024
MIN_TOOL_BACKENDS = 2

LAB_VALUES = {"lactate": "4.8 mmol/L", "creatinine": "1.9 mg/dL"}


class LabQuery(BaseModel):
    name: str


class Vital(BaseModel):
    name: str
    value: float


class Assessment(BaseModel):
    """Nested: object with a list of objects."""

    severity: str
    vitals: list[Vital]


def _config(tmp_path, backend: str, reasoning: str = "OFF", max_tokens: int = MAX_TOKENS, **backend_overrides):
    """Committed backends plus one test agent on `backend`."""
    lines = []
    for name, entry in BACKENDS_TOML.items():
        entry = entry | (backend_overrides if name == backend else {})
        lines.append(f"[backends.{name}]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in entry.items()]
    lines += [
        "[agents.contract]", f'backend = "{backend}"', f'reasoning = "{reasoning}"', f"max_tokens = {max_tokens}",
    ]
    path = tmp_path / "llm.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _client(tmp_path, backend: str, **kw) -> LlmClient:
    return LlmClient.for_agent("contract", _config(tmp_path, backend, **kw))


def _ask(text: str, system: str = "You are a terse ICU assistant.") -> Request:
    return Request((Message(Role.SYSTEM, system), Message(Role.USER, text)))


def _records(clean_env) -> list[dict]:
    return read_records(clean_env / RECORD_FILE)


@pytest.fixture(scope="module", params=BACKENDS)
def backend(request) -> str:
    """Module scope groups tests per backend: one model load each instead of a swap per test."""
    return request.param


_PROBES: dict = {}


@pytest.fixture(autouse=True)
def shared_probes(clean_env, monkeypatch):
    """Capability probes once per backend for the whole suite."""
    monkeypatch.setattr(client_module, "_probe_cache", _PROBES)


# ---------- criterion 2 ----------

def test_generation(tmp_path, backend):
    """protects=criterion 2 generation;
    fails_when=a plain request fails, finishes abnormally or returns nothing;
    no_claim=answer quality."""
    completion = _client(tmp_path, backend).generate(_ask("Name the organ that produces creatinine clearance problems when failing. One word."), CTX)
    assert completion.finish is FinishReason.STOP and completion.message.content.strip()


def test_nested_structured_output_or_refusal(tmp_path, backend, clean_env):
    """protects=criterion 2 nested structured output; unsupported -> CapabilityError with no request;
    fails_when=a JSON-schema backend returns an invalid object, or an unsupported one sends a request;
    no_claim=clinical correctness of the values."""
    client = _client(tmp_path, backend)
    request = _ask("Synthetic case: HR 128/min, MAP 58 mmHg, SpO2 91%. Give severity and the three vitals.")
    if Capability.JSON_SCHEMA not in client.capabilities.supported:
        with pytest.raises(CapabilityError):
            client.structured(request, Assessment, CTX)
        assert _records(clean_env) == []
        return

    value, completion = client.structured(request, Assessment, CTX)
    assert len(value.vitals) >= 1 and value.severity
    assert completion.finish is FinishReason.STOP


def test_reasoning_on_off_or_refusal(tmp_path, backend):
    """protects=criterion 2 reasoning on/off, reasoning text kept apart (L9);
    fails_when=ON yields no reasoning, OFF yields reasoning, or ON is accepted by a backend without it;
    no_claim=reasoning quality."""
    off = _client(tmp_path, backend)
    if Capability.REASONING not in off.capabilities.supported:
        with pytest.raises(CapabilityError):
            _client(tmp_path, backend, reasoning="ON")
        return

    question = _ask("Is a lactate of 4.8 mmol/L above the usual sepsis threshold of 2? Answer yes or no.")
    assert off.generate(question, CTX).reasoning is None
    on = _client(tmp_path, backend, reasoning="ON").generate(question, CTX)
    assert on.reasoning and on.finish is FinishReason.STOP
    assert on.reasoning not in on.message.content


def test_tools_or_refusal(tmp_path, backend, clean_env):
    """protects=criterion 2 tools: unsupported -> CapabilityError, no request;
    fails_when=a backend without working tool calls accepts a tool request;
    no_claim=covered by the round-trip test for supporting backends."""
    client = _client(tmp_path, backend)
    if Capability.TOOLS in client.capabilities.supported:
        return
    with pytest.raises(CapabilityError):
        client.generate(Request(_ask("x").messages, (ToolSpec("get_lab", "lab", LabQuery),)), CTX)
    assert _records(clean_env) == []


# ---------- criterion 3 ----------

def test_enough_tool_backends(tmp_path):
    """protects=criterion 3 precondition: ≥2 tool-capable backends (two model families, L1);
    fails_when=fewer than two backends pass the live tool probe;
    no_claim=which families."""
    supported = [b for b in BACKENDS if Capability.TOOLS in _client(tmp_path, b).capabilities.supported]
    assert len(supported) >= MIN_TOOL_BACKENDS, f"tool-capable backends: {supported}"


def test_tool_round_trip(tmp_path, backend):
    """protects=criterion 3: assistant turn with ≥2 tool calls → results linked to calls → answer using both;
    fails_when=fewer than two calls, a result cannot be linked, or the answer ignores a result;
    no_claim=tool choice in harder prompts."""
    client = _client(tmp_path, backend)
    if Capability.TOOLS not in client.capabilities.supported:
        pytest.skip(f"{backend}: TOOLS not supported (test_tools_or_refusal covers it)")

    tools = (ToolSpec("get_lab", "Latest value of one lab test", LabQuery),)
    request = Request(
        _ask("Call get_lab for lactate and for creatinine, both in this one turn, then report both values.").messages,
        tools,
    )
    first = client.generate(request, CTX)
    assert first.finish is FinishReason.TOOL_CALLS
    names = sorted(c.arguments["name"].lower() for c in first.message.tool_calls)
    assert names == ["creatinine", "lactate"]

    results = tuple(
        Message(Role.TOOL, LAB_VALUES[c.arguments["name"].lower()], tool_call_id=c.id, tool_name=c.name)
        for c in first.message.tool_calls
    )
    answer = client.generate(Request(request.messages + (first.message,) + results, tools), CTX)
    assert answer.finish is FinishReason.STOP
    assert "4.8" in answer.message.content and "1.9" in answer.message.content


# ---------- criterion 5 ----------

def _long_request(words: int, with_tools: bool) -> Request:
    """Long synthetic summary + multi-turn history (+ tools)."""
    summary = " ".join(f"hour{i % 24} HR {80 + i % 50} MAP {60 + i % 30};" for i in range(words))
    messages = [Message(Role.SYSTEM, "You review synthetic ICU summaries."), Message(Role.USER, summary)]
    for i in range(3):
        messages += [Message(Role.ASSISTANT, f"Noted part {i}."), Message(Role.USER, f"Continue with part {i + 1}.")]
    tools = (ToolSpec("get_lab", "Latest value of one lab test", LabQuery),) if with_tools else ()
    return Request(tuple(messages), tools)


def test_budget_refuses_over_context_and_accepts_exact_count(tmp_path, backend, clean_env):
    """protects=criterion 5: over-budget full request -> ContextOverflowError with no inference; an accepted
    long request is evaluated at exactly the pre-counted length;
    fails_when=an oversize request reaches inference, or prompt_tokens differs from the pre-count;
    no_claim=behaviour at the GPU limit (docs/backends/gpu_budget.md)."""
    client = _client(tmp_path, backend, max_tokens=16)
    with_tools = Capability.TOOLS in client.capabilities.supported

    # ~10 tokens per synthetic entry: twice the context.
    too_big = _long_request(words=client.capabilities.context_tokens // 5, with_tools=with_tools)
    assert client.count_request_tokens(too_big, Assessment) + 16 > client.capabilities.context_tokens
    with pytest.raises(ContextOverflowError):
        client.generate(too_big, CTX)
    refused = _records(clean_env)[-1]
    assert refused["wire_request"] is None and refused["raw_response"] is None

    accepted = _long_request(words=600, with_tools=with_tools)
    completion = client.generate(accepted, CTX)
    record = _records(clean_env)[-1]
    assert record["pre_count"] == completion.usage.prompt_tokens > 4000


def test_forced_truncation_detected(tmp_path, backend, clean_env, monkeypatch):
    """protects=criterion 5 forced silent truncation (server context below config);
    native Ollama: `truncate: false` refusal; /v1: post-call count mismatch;
    fails_when=an answer to a shortened prompt is returned;
    no_claim=other truncation mechanisms."""
    adapter = BACKENDS_TOML[backend]["adapter"]
    if adapter == "openai_compat":
        # /v1 cannot set num_ctx: config claims more than the server default (32k on this GPU).
        client = _client(tmp_path, backend, max_tokens=16, context_tokens=65536)
        request = _long_request(words=3500, with_tools=False)
        assert client.count_request_tokens(request) > 32768
    else:
        # Native: pretend the pre-count is tiny so the request reaches a server whose num_ctx is too small.
        client = _client(tmp_path, backend, max_tokens=16)
        request = _long_request(words=3500, with_tools=False)
        monkeypatch.setattr(client._backend, "count_request_tokens", lambda call: 1)

    with pytest.raises(ContextOverflowError):
        client.generate(request, CTX)
    assert _records(clean_env)[-1]["error"] == "ContextOverflowError"


# ---------- criterion 6 (quick; full 50-request check: scripts/measure_backends.py tokenizer) ----------

def test_pre_count_matches_server_count(tmp_path, backend, clean_env):
    """protects=criterion 6 smoke: pre-count equals prompt_eval_count on varied requests (unicode, history, tools);
    fails_when=any accepted call's count differs (the client would raise ContextOverflowError);
    no_claim=the 50-request evidence (measurement script)."""
    client = _client(tmp_path, backend, max_tokens=4)
    tools = (ToolSpec("get_lab", "Latest value of one <lab> & test", LabQuery),) if Capability.TOOLS in client.capabilities.supported else ()
    texts = ["SpO₂ 92 % · FiO₂ 0.4 · PEEP 8 cmH₂O", "Ünïcödé 日本語 😀 \"quoted\" {json: 1}\n\n\ttab", "1234567890 3.14159 -42 1e-9"]
    for i, text in enumerate(texts):
        messages = _ask(text).messages + ((Message(Role.ASSISTANT, "ok"), Message(Role.USER, "again")) if i else ())
        client.generate(Request(messages, tools), CTX)
    assert all(r["pre_count"] == r["usage"]["prompt_tokens"] for r in _records(clean_env))


# ---------- criterion 9 ----------

def test_two_fresh_calls_identical(tmp_path, backend):
    """protects=criterion 9: same agent, request, seed, temperature 0 -> identical answers;
    fails_when=answers differ (then variability must be measured and reported: scripts/measure_backends.py);
    no_claim=bitwise GPU determinism across loads or batch sizes."""
    request = _ask("List three causes of lactic acidosis, comma-separated.")
    answers = [_client(tmp_path, backend).generate(request, CTX).message.content for _ in range(2)]
    assert answers[0] == answers[1]
