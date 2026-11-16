"""LlmClient behaviour against a fake Ollama: capabilities, budgeting, retries, tools, records."""

import json
import logging
import socket
import time

import httpx
import pytest
from pydantic import BaseModel

from argus.llm import (
    BackendUnavailableError, CallContext, Capability, CapabilityError, ContextOverflowError, FinishReason,
    LlmClient, Message, PolicyViolationError, Request, Role, StructuredOutputError, ToolCallError, ToolSpec,
)
from argus.llm.backends import BackendCall, WireTool, connect
from argus.llm.config import load_agent
from argus.llm.sink import RECORD_FILE, message_from_dict, read_records
from argus.llm.types import Reasoning, Settings
from llm_fakes import DIGEST, SENTINEL, VERSION, Reply, agent_toml, backend_toml

CTX = CallContext("run-1", "step-1", parent_record_id="parent-0", labels={"arm": "single", "window": "w1"})
ADAPTERS = ["ollama", "openai_compat"]


class Lab(BaseModel):
    name: str
    hours: int


class Window(BaseModel):
    start: str
    end: str


class Finding(BaseModel):
    """Nested schema: list of objects inside an object."""

    risk: float
    windows: list[Window]


LAB_TOOL = ToolSpec("get_lab", "Fetch one lab series", Lab)
VITALS_TOOL = ToolSpec("get_window", "Fetch a time window", Window)


def _client(write_config, fake, adapter="ollama", backend_kw=None, **agent_kw) -> LlmClient:
    path = write_config(backend_toml(adapter=adapter, **(backend_kw or {})), agent_toml(**agent_kw))
    return LlmClient.for_agent("agent", path, transport=httpx.MockTransport(fake.handle))


def _request(*texts: str, tools=()) -> Request:
    return Request(tuple(Message(Role.USER, t) for t in texts), tools)


def _records(run_dir) -> list[dict]:
    return read_records(run_dir / RECORD_FILE)


# ---------- capabilities (criteria 2, 12; L6, L10) ----------

def test_tools_unsupported_raise_capability_error_without_request(write_config, fake):
    """protects=criterion 2 unsupported capability;
    fails_when=a tool request reaches a backend without TOOLS;
    no_claim=real server capability reporting (contract suite)."""
    fake.capabilities = ("completion",)
    client = _client(write_config, fake)
    with pytest.raises(CapabilityError):
        client.generate(_request("hi", tools=(LAB_TOOL,)), CTX)
    assert fake.agent_requests() == []


def test_reported_tools_dropped_when_calls_do_not_parse(write_config, fake):
    """protects=L6 live probe (mistral-small3.2 on Ollama 0.34.1 reports tools but its calls are dropped);
    fails_when=TOOLS is trusted from /api/show alone;
    no_claim=probe covers every tool-call shape."""
    fake.tools_parse = False
    client = _client(write_config, fake)
    assert Capability.TOOLS not in client.capabilities.supported


def test_json_schema_unsupported_raises_capability_error_without_request(write_config, fake):
    """protects=criterion 2 structured output on a backend whose JSON-schema probe fails;
    fails_when=structured() sends a request anyway;
    no_claim=grammar quality."""
    fake.json_schema_ok = False
    client = _client(write_config, fake)
    with pytest.raises(CapabilityError):
        client.structured(_request("hi"), Finding, CTX)
    assert fake.agent_requests() == []


def test_config_restricts_never_grants(write_config, fake):
    """protects=L6 intersection;
    fails_when=a config allowlist adds a capability the server lacks, or a listed restriction is ignored;
    no_claim=none."""
    fake.capabilities = ("completion",)
    client = _client(write_config, fake, backend_kw={"capabilities": ["TOOLS", "SEED", "JSON_SCHEMA"]})
    assert Capability.TOOLS not in client.capabilities.supported

    fake.capabilities = ("completion", "tools")
    client = _client(write_config, fake, backend_kw={"capabilities": ["SEED"]})
    assert client.capabilities.supported == {Capability.SEED}


def test_required_capability_missing_fails_at_construction(write_config, fake):
    """protects=criteria 12 + L10: a swap to a backend lacking a `requires` capability fails at construction;
    fails_when=the client is built and degrades later;
    no_claim=none."""
    fake.capabilities = ("completion",)
    with pytest.raises(CapabilityError):
        _client(write_config, fake, requires=["TOOLS"])
    with pytest.raises(CapabilityError):
        _client(write_config, fake, reasoning="ON")


def test_backend_swap_is_config_only(write_config, fake):
    """protects=criterion 12: swapping backend changes only config;
    fails_when=the same agent code needs changes to run on the other adapter;
    no_claim=answer equivalence across models."""
    texts = []
    for adapter in ADAPTERS:
        fake.replies.append(Reply(content="ok"))
        completion = _client(write_config, fake, adapter=adapter).generate(_request("hi"), CTX)
        texts.append(completion.message.content)
    assert texts == ["ok", "ok"]
    paths = [r.url.path for r in fake.inference_requests()]
    assert "/api/chat" in paths and "/v1/chat/completions" in paths


def test_context_above_nominal_refused(write_config, fake):
    """protects=config context cannot exceed the model's own context;
    fails_when=an impossible context_tokens is accepted;
    no_claim=GPU fit (measured, docs/backends/gpu_budget.md)."""
    with pytest.raises(CapabilityError):
        _client(write_config, fake, backend_kw={"context_tokens": 10**7})


def test_reasoning_switch_always_explicit(write_config, fake):
    """protects=L9/L10: a thinking model gets an explicit think value (Qwen3.8 defaults to xhigh thinking);
    fails_when=reasoning OFF omits the switch, or a model without the switch is sent one;
    no_claim=model honours the switch."""
    fake.replies += [Reply(content="a"), Reply(content="b")]
    _client(write_config, fake).generate(_request("hi"), CTX)
    assert fake.agent_requests()[-1]["think"] is False

    fake.capabilities = ("completion", "tools")
    _client(write_config, fake).generate(_request("hi"), CTX)
    assert "think" not in fake.agent_requests()[-1]


# ---------- budgeting (criterion 5, L3) ----------

def _full_request(history_turns: int, filler: int) -> Request:
    """Long summary + multi-turn history + tool definitions."""
    messages = [Message(Role.SYSTEM, "You review ICU care."), Message(Role.USER, "summary " + "x" * filler)]
    for i in range(history_turns):
        messages += [Message(Role.ASSISTANT, f"note {i}"), Message(Role.USER, f"follow-up {i}")]
    return Request(tuple(messages), (LAB_TOOL, VITALS_TOOL))


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_over_budget_request_refused_without_inference(write_config, fake, clean_env, adapter):
    """protects=criterion 5: count_request_tokens + max_tokens > context_tokens -> ContextOverflowError, nothing sent;
    fails_when=an oversize request reaches inference, or the budget ignores history/tools/schema/max_tokens;
    no_claim=server-side truncation (next tests)."""
    client = _client(write_config, fake, adapter=adapter)
    request = _full_request(history_turns=3, filler=0)
    base = client.count_request_tokens(request, Finding)
    room = client.capabilities.context_tokens - 256 - base   # 256 = max_tokens

    fits = _full_request(history_turns=3, filler=room)
    assert client.count_request_tokens(fits, Finding) + 256 == client.capabilities.context_tokens
    too_big = _full_request(history_turns=3, filler=room + 1)
    with pytest.raises(ContextOverflowError):
        client.generate(too_big, CTX)
    assert fake.agent_requests() == []

    # The refusal is recorded (request text in the run dir only), with no wire body.
    record = _records(clean_env)[-1]
    assert record["error"] == "ContextOverflowError" and record["wire_request"] is None


def test_count_request_tokens_includes_schema_tools_and_history(write_config, fake):
    """protects=count_request_tokens covers the full input;
    fails_when=adding tools or history leaves the count unchanged;
    no_claim=exactness on real servers (criterion 6, contract suite)."""
    client = _client(write_config, fake)
    bare = client.count_request_tokens(_request("hi"))
    assert client.count_request_tokens(_request("hi", tools=(LAB_TOOL,))) > bare
    assert client.count_request_tokens(_request("hi", "again")) > bare
    assert client.count_text_tokens("héllo") == len("héllo".encode())


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_silent_truncation_detected_after_call(write_config, fake, clean_env, adapter):
    """protects=criterion 5 forced silent truncation: evaluated prompt shorter than pre-count -> ContextOverflowError;
    fails_when=a truncated prompt's answer is returned to the agent;
    no_claim=which part was truncated."""
    client = _client(write_config, fake, adapter=adapter)
    fake.prompt_eval_delta = -1027
    fake.replies.append(Reply(content="answer from a truncated prompt"))
    with pytest.raises(ContextOverflowError) as error:
        client.generate(_request("hi"), CTX)
    record = _records(clean_env)[-1]
    assert record["error"] == "ContextOverflowError" and record["record_id"] == error.value.record_id
    assert record["raw_response"] is not None


def test_v1_silent_truncation_not_masked_by_truncating_render(write_config, fake):
    """protects=criterion 5 on /v1 (regression): Ollama's /v1 drops old messages to fit its default
    context, in render-only mode too, so a /v1-rendered pre-count agrees with the truncated evaluation;
    fails_when=an answer to a prompt that lost the summary is returned;
    no_claim=remote providers' truncation behaviour."""
    fake.v1_num_ctx = 2000
    client = _client(write_config, fake, adapter="openai_compat", backend_kw={"context_tokens": 65536})
    fake.replies.append(Reply(content="answer without the summary"))
    request = Request((Message(Role.SYSTEM, "sys"), Message(Role.USER, "summary " + "x" * 5000), Message(Role.USER, "question")))
    with pytest.raises(ContextOverflowError):
        client.generate(request, CTX)


def test_server_context_refusal_maps_to_overflow(write_config, fake, monkeypatch):
    """protects=Ollama `truncate: false` refusal (HTTP 400) becomes ContextOverflowError;
    fails_when=it surfaces as a generic error or leaks the server body;
    no_claim=pre-count accuracy (bypassed here)."""
    client = _client(write_config, fake)
    monkeypatch.setattr(client._backend, "count_request_tokens", lambda call: 1)
    with pytest.raises(ContextOverflowError) as error:
        client.generate(_request("y" * 5000), CTX)
    assert SENTINEL not in str(error.value)


# ---------- structured output (criterion 4, L4) ----------

def test_structured_retry_carries_validation_errors_then_valid(write_config, fake, clean_env):
    """protects=criterion 4 retry path: invalid first answer -> retry with errors -> valid;
    fails_when=no retry, retry without the invalid answer and error list, or input values echoed in feedback;
    no_claim=model uses the feedback well."""
    client = _client(write_config, fake)
    fake.replies += [
        Reply(content=json.dumps({"risk": SENTINEL, "windows": []})),
        Reply(content=json.dumps({"risk": 0.8, "windows": [{"start": "08:00", "end": "14:00"}]})),
    ]
    value, completion = client.structured(_request("assess"), Finding, CTX)
    assert value.windows[0].end == "14:00"

    second = fake.agent_requests()[-1]["messages"]
    assert second[-2] == {"role": "assistant", "content": json.dumps({"risk": SENTINEL, "windows": []})}
    assert "risk" in second[-1]["content"] and SENTINEL not in second[-1]["content"]

    records = _records(clean_env)
    assert [r["attempt"] for r in records] == [1, 2]
    assert records[1]["record_id"] == completion.record_id


def test_structured_invalid_twice_raises(write_config, fake, clean_env):
    """protects=criterion 4: one retry only, then StructuredOutputError; every attempt recorded;
    fails_when=a third attempt is made or an invalid value is returned;
    no_claim=none."""
    client = _client(write_config, fake)
    fake.replies += [Reply(content="not json"), Reply(content='{"risk": "high"}')]
    with pytest.raises(StructuredOutputError):
        client.structured(_request("assess"), Finding, CTX)
    assert len(fake.agent_requests()) == 2
    assert len(_records(clean_env)) == 2


def test_structured_length_finish_not_retried(write_config, fake, clean_env):
    """protects=criterion 4: finish LENGTH -> no retry;
    fails_when=a cut-off answer triggers a second, equally long attempt;
    no_claim=none."""
    client = _client(write_config, fake)
    fake.replies.append(Reply(content='{"risk": 0.', done_reason="length"))
    with pytest.raises(StructuredOutputError):
        client.structured(_request("assess"), Finding, CTX)
    assert len(fake.agent_requests()) == 1
    assert _records(clean_env)[0]["finish"] == "length"


def test_structured_sends_inlined_nested_schema(write_config, fake):
    """protects=nested schemas reach the server without dangling $ref;
    fails_when=the wire schema contains $ref/$defs (Ollama keeps only some keys per tool property);
    no_claim=grammar support on the server."""
    client = _client(write_config, fake)
    fake.replies.append(Reply(content=json.dumps({"risk": 0.1, "windows": []})))
    client.structured(_request("assess"), Finding, CTX)
    wire = json.dumps(fake.agent_requests()[-1]["format"])
    assert "$ref" not in wire and "$defs" not in wire and '"start"' in wire


# ---------- tool calls (criterion 3, L5) ----------

def test_tool_round_trip_links_results_to_calls(write_config, fake):
    """protects=criterion 3 shape: ≥2 calls in one turn, results linked by id, follow-up answer;
    fails_when=ids or names are lost between the assistant turn and the tool results;
    no_claim=model quality (contract suite)."""
    client = _client(write_config, fake)
    fake.replies += [
        Reply(tool_calls=[("get_lab", {"name": "lactate", "hours": 6}), ("get_window", {"start": "08:00", "end": "14:00"})]),
        Reply(content="lactate 4.8 within 08:00-14:00"),
    ]
    request = _request("check", tools=(LAB_TOOL, VITALS_TOOL))
    first = client.generate(request, CTX)
    assert first.finish is FinishReason.TOOL_CALLS and len(first.message.tool_calls) == 2

    results = tuple(
        Message(Role.TOOL, f"result {c.name}", tool_call_id=c.id, tool_name=c.name) for c in first.message.tool_calls
    )
    follow = Request(request.messages + (first.message,) + results, request.tools)
    second = client.generate(follow, CTX)
    assert second.finish is FinishReason.STOP

    wire = fake.agent_requests()[-1]["messages"]
    calls = wire[1]["tool_calls"]
    assert [m["tool_call_id"] for m in wire[2:]] == [c["id"] for c in calls]
    assert [m["tool_name"] for m in wire[2:]] == ["get_lab", "get_window"]


@pytest.mark.parametrize("bad", [("drop_table", {"name": "x"}), ("get_lab", {"name": "lactate", "hours": "six"})])
def test_invalid_tool_call_retried_then_valid(write_config, fake, bad):
    """protects=criterion 3: unknown tool / schema-failing arguments -> retry with per-call feedback;
    fails_when=an invalid call is returned to the caller, or the retry lacks feedback linked to the call;
    no_claim=none."""
    client = _client(write_config, fake)
    fake.replies += [Reply(tool_calls=[bad]), Reply(tool_calls=[("get_lab", {"name": "lactate", "hours": 6})])]
    completion = client.generate(_request("check", tools=(LAB_TOOL,)), CTX)
    assert completion.message.tool_calls[0].arguments == {"name": "lactate", "hours": 6}

    retry = fake.agent_requests()[-1]["messages"]
    assert retry[-1]["role"] == "tool" and retry[-1]["tool_call_id"] == retry[-2]["tool_calls"][0]["id"]
    assert "rejected" in retry[-1]["content"]


def test_invalid_tool_calls_twice_raise_tool_call_error(write_config, fake, clean_env):
    """protects=criterion 3: never returned to the caller after the retry;
    fails_when=generate returns invalid calls or retries more than once;
    no_claim=none."""
    client = _client(write_config, fake)
    fake.replies += [Reply(tool_calls=[("nope", {})]), Reply(tool_calls=[("get_lab", {"hours": 1})])]
    with pytest.raises(ToolCallError):
        client.generate(_request("check", tools=(LAB_TOOL,)), CTX)
    assert len(_records(clean_env)) == 2


def test_dropped_tool_call_turn_is_invalid(write_config, fake):
    """protects=L5 against servers that swallow malformed calls (empty content, no calls, finish stop);
    fails_when=an empty turn is returned as a valid answer;
    no_claim=detection of partially dropped calls."""
    client = _client(write_config, fake)
    fake.replies += [Reply(), Reply()]
    with pytest.raises(ToolCallError):
        client.generate(_request("check", tools=(LAB_TOOL,)), CTX)


def test_structured_refuses_tools(write_config, fake):
    """protects=structured() and tools are separate paths;
    fails_when=tools are silently ignored in structured();
    no_claim=none."""
    with pytest.raises(ValueError):
        _client(write_config, fake).structured(_request("x", tools=(LAB_TOOL,)), Finding, CTX)


# ---------- records (criteria 10, 11) ----------

@pytest.mark.parametrize("adapter", ADAPTERS)
def test_record_per_attempt_with_provenance(write_config, fake, clean_env, adapter):
    """protects=criterion 11: one record per attempt with digest, server version, settings, usage, context ids;
    reasoning apart from the answer; metrics the backend does not report are null;
    fails_when=a field is missing, reasoning leaks into the message, or an unreported metric is invented;
    no_claim=metric accuracy."""
    client = _client(write_config, fake, adapter=adapter)
    fake.replies.append(Reply(content="answer", thinking="private chain"))
    completion = client.generate(_request("hi"), CTX)

    (record,) = _records(clean_env)
    assert record["digest"] == DIGEST and record["server_version"] == VERSION
    assert record["settings"]["seed"] == 7 and record["settings"]["temperature"] == 0.0
    assert (record["run_id"], record["step_id"], record["parent_record_id"]) == ("run-1", "step-1", "parent-0")
    assert record["labels"] == {"arm": "single", "window": "w1"}
    assert record["reasoning"] == "private chain" and record["message"]["content"] == "answer"
    assert record["usage"]["reasoning_tokens"] is None and record["usage"]["queue_s"] is None
    assert record["pre_count"] == record["usage"]["prompt_tokens"]
    assert completion.reasoning == "private chain" and "private" not in completion.message.content

    if adapter == "ollama":
        assert completion.usage.gpu_s == pytest.approx(0.005)
        assert completion.usage.cost == pytest.approx(0.01) and completion.usage.currency == "GPU_S"
    else:
        assert record["usage"]["gpu_s"] is None and record["usage"]["cost"] is None


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_records_rebuild_wire_request_byte_identical(write_config, fake, clean_env, adapter, tmp_path):
    """protects=criterion 10: records alone rebuild every request (messages, tool calls + results, settings)
    byte-identical to what was sent, without inference;
    fails_when=any field needed to rebuild the body is missing or re-encoded differently;
    no_claim=server-side replay of the answer (fresh runs measure variability, criterion 9)."""
    client = _client(write_config, fake, adapter=adapter)
    fake.replies += [Reply(tool_calls=[("get_lab", {"name": "lactate", "hours": 6})]), Reply(content="done ✓")]
    request = _request("check ünïcode", tools=(LAB_TOOL,))
    first = client.generate(request, CTX)
    results = (Message(Role.TOOL, "4.8", tool_call_id=first.message.tool_calls[0].id, tool_name="get_lab"),)
    client.generate(Request(request.messages + (first.message,) + results, request.tools), CTX)

    sent = [r.content for r in fake.inference_requests() if json.loads(r.content)["messages"][0]["content"].startswith("check")]
    config = load_agent("agent", tmp_path / "llm.toml").backend
    backend = connect(config, lambda status, name: None, httpx.MockTransport(fake.handle)).backend
    rebuilt = []
    for record in _records(clean_env):
        settings = record["settings"]
        call = BackendCall(
            messages=tuple(message_from_dict(m) for m in record["request"]["messages"]),
            tools=tuple(WireTool(**t) for t in record["request"]["tools"]),
            schema=record["request"]["schema"],
            settings=Settings(settings["temperature"], settings["seed"], settings["max_tokens"], Reasoning(settings["reasoning"])),
            reasoning=None if settings["reasoning_sent"] is None else Reasoning(settings["reasoning_sent"]),
            context_tokens=settings["context_tokens"],
        )
        rebuilt.append(backend.encode(call))
        assert rebuilt[-1].decode() == record["wire_request"]
    assert rebuilt == sent


# ---------- failure hygiene (criteria 13, 15) ----------

def test_failures_never_expose_request_content(write_config, fake, caplog, monkeypatch):
    """protects=criterion 13 / L11: error messages and logs carry ids, never request content;
    fails_when=the sentinel appears in any exception text or log record of the failure paths;
    no_claim=third-party library logs outside argus loggers."""
    caplog.set_level(logging.DEBUG)
    errors = []

    def expect(exc_type, fn):
        with pytest.raises(exc_type) as info:
            fn()
        errors.append(info.value)

    client = _client(write_config, fake, max_tokens=4000)
    expect(ContextOverflowError, lambda: client.generate(_request(SENTINEL * 20), CTX))

    client = _client(write_config, fake)
    fake.replies += [Reply(content=SENTINEL), Reply(content=json.dumps({"risk": SENTINEL}))]
    expect(StructuredOutputError, lambda: client.structured(_request(SENTINEL), Finding, CTX))

    fake.replies += [Reply(tool_calls=[(SENTINEL, {"x": SENTINEL})])] * 2
    expect(ToolCallError, lambda: client.generate(_request(SENTINEL, tools=(LAB_TOOL,)), CTX))

    fake.prompt_eval_delta = -1
    fake.replies.append(Reply(content=SENTINEL))
    expect(ContextOverflowError, lambda: client.generate(_request(SENTINEL), CTX))
    fake.prompt_eval_delta = 0

    monkeypatch.setattr(client._backend, "count_request_tokens", lambda call: 1)
    expect(ContextOverflowError, lambda: client.generate(_request(SENTINEL * 400), CTX))

    fake.capabilities = ("completion",)
    expect(CapabilityError, lambda: _client(write_config, fake).generate(_request(SENTINEL, tools=(LAB_TOOL,)), CTX))
    monkeypatch.setenv("HTTPS_PROXY", SENTINEL)
    expect(PolicyViolationError, lambda: _client(write_config, fake))

    texts = [str(e) for e in errors] + [repr(e) for e in errors] + [caplog.text]
    assert all(SENTINEL not in t for t in texts)
    assert len(errors) == 7


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_backend_down_raises_unavailable_naming_backend(write_config, adapter):
    """protects=criterion 15: Ollama down -> BackendUnavailableError naming the backend, within the timeout;
    fails_when=a raw httpx error escapes, the backend is not named, or the call hangs past connect_timeout_s;
    no_claim=behaviour when the server accepts but stalls mid-answer (request_timeout_s)."""
    url = f"http://127.0.0.1:{_closed_port()}"
    backend_kw = {"base_url": url if adapter == "ollama" else f"{url}/v1", "connect_timeout_s": 2.0}
    if adapter == "openai_compat":
        backend_kw["control_url"] = url
    path = write_config(backend_toml(adapter=adapter, **backend_kw), agent_toml())

    started = time.monotonic()
    with pytest.raises(BackendUnavailableError, match="backend fake"):
        LlmClient.for_agent("agent", path)
    assert time.monotonic() - started < 2.0


def test_backend_timeout_raises_unavailable(write_config, fake):
    """protects=criterion 15 timeout path;
    fails_when=httpx.TimeoutException escapes untyped;
    no_claim=timing (fake transport)."""
    def stall(request):
        raise httpx.ReadTimeout("stalled", request=request)

    path = write_config(backend_toml(), agent_toml())
    with pytest.raises(BackendUnavailableError, match="backend fake"):
        LlmClient.for_agent("agent", path, transport=httpx.MockTransport(stall))

