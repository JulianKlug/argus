"""Adapter mapping in both directions: recorded real responses (tests/llm/fixtures) and wire bodies."""

import json
from pathlib import Path

import httpx
import pytest

from argus.llm.backends import BackendCall, WireTool, connect
from argus.llm.backends.ollama import _decode as decode_ollama
from argus.llm.backends.openai_compat import _decode as decode_openai
from argus.llm.config import _backend
from argus.llm.errors import BackendMismatchError
from argus.llm.types import FinishReason, Message, Reasoning, Role, Settings, ToolCall

FIXTURES = Path(__file__).parent / "fixtures"
SETTINGS = Settings(temperature=0.0, seed=7, max_tokens=512, reasoning=Reasoning.ON)
LAB = WireTool("get_lab", "Latest value of one lab test", {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _backend_for(adapter: str):
    config = _backend("b", {
        "adapter": adapter, "base_url": "http://127.0.0.1:11434" + ("/v1" if adapter == "openai_compat" else ""),
        "control_url": "http://127.0.0.1:11434", "model": "m:1b", "digest": "a" * 64, "policy": "LOCAL", "context_tokens": 8192,
    })
    return connect(config, lambda status, name: None, httpx.MockTransport(lambda r: httpx.Response(500))).backend


def _round_trip_call(reasoning: Reasoning | None) -> BackendCall:
    """User → assistant with 2 tool calls → 2 linked tool results."""
    calls = (ToolCall("call_a", "get_lab", {"name": "lactate"}), ToolCall("call_b", "get_lab", {"name": "créatinine"}))
    return BackendCall(
        messages=(
            Message(Role.SYSTEM, "sys"),
            Message(Role.USER, "check"),
            Message(Role.ASSISTANT, "", tool_calls=calls),
            Message(Role.TOOL, "4.8", tool_call_id="call_a", tool_name="get_lab"),
            Message(Role.TOOL, "1.9", tool_call_id="call_b", tool_name="get_lab"),
        ),
        tools=(LAB,), schema=None, settings=SETTINGS, reasoning=reasoning, context_tokens=8192,
    )


def test_ollama_decodes_recorded_parallel_tool_calls():
    """protects=Ollama native → ToolCall mapping on a real qwen3.8 response (2 calls, ids, thinking);
    fails_when=ids, names, arguments or finish are lost, or thinking leaks into content;
    no_claim=other models' output formats."""
    result = decode_ollama(_fixture("ollama_chat_tool_calls.json"))
    assert result.finish is FinishReason.TOOL_CALLS
    assert [(c.name, c.arguments) for c in result.message.tool_calls] == [("get_lab", {"name": "lactate"}), ("get_lab", {"name": "creatinine"})]
    assert len({c.id for c in result.message.tool_calls}) == 2 and all(c.id.startswith("call_") for c in result.message.tool_calls)
    assert result.reasoning and result.reasoning not in result.message.content
    assert result.prompt_tokens > 0 and result.gpu_s > 0 and result.load_s is not None
    assert result.malformed_tool_calls == frozenset()


def test_ollama_decodes_recorded_length_finish():
    """protects=LENGTH detection (drives the no-retry rule) and reasoning kept apart;
    fails_when=a cut-off answer is reported as STOP;
    no_claim=none."""
    result = decode_ollama(_fixture("ollama_chat_length.json"))
    assert result.finish is FinishReason.LENGTH
    assert result.message.content == "" and result.reasoning


def test_openai_decodes_recorded_tool_calls_and_text():
    """protects=OpenAI-compatible → ToolCall mapping (arguments are JSON strings) and usage;
    fails_when=arguments stay strings, ids drop, or unreported metrics are invented;
    no_claim=remote providers' extensions."""
    result = decode_openai(_fixture("openai_chat_tool_calls.json"))
    assert result.finish is FinishReason.TOOL_CALLS
    assert [c.arguments for c in result.message.tool_calls] == [{"name": "lactate"}, {"name": "creatinine"}]
    assert result.reasoning and result.load_s is None and result.gpu_s is None and result.reasoning_tokens is None

    text = decode_openai(_fixture("openai_chat_text.json"))
    assert text.finish is FinishReason.STOP and text.message.content == "Yes."


def test_malformed_arguments_flagged_not_dropped():
    """protects=L5 input: a call whose arguments are not a JSON object is flagged for the retry path;
    fails_when=it is silently dropped or passed on with string arguments;
    no_claim=none."""
    data = _fixture("openai_chat_tool_calls.json")
    data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"name": "lact'
    result = decode_openai(data)
    assert result.malformed_tool_calls == {result.message.tool_calls[0].id}


@pytest.mark.parametrize("decode,data", [
    (decode_ollama, {"done_reason": "unload", "message": {}, "prompt_eval_count": 1}),
    (decode_openai, {"choices": [{"message": {}, "finish_reason": "content_filter"}], "usage": {"prompt_tokens": 1}}),
])
def test_unknown_finish_refused(decode, data):
    """protects=finish mapping is closed;
    fails_when=an unknown finish reason is passed on as STOP;
    no_claim=none."""
    with pytest.raises(BackendMismatchError):
        decode(data)


def test_ollama_encodes_tool_round_trip():
    """protects=Ollama native wire form of a tool exchange: ids on calls, tool_name + tool_call_id on results;
    fails_when=the native template could not link results to calls (needs tool_name);
    no_claim=template rendering (server side)."""
    body = json.loads(_backend_for("ollama").encode(_round_trip_call(Reasoning.OFF)))
    assistant, first, second = body["messages"][2:]
    assert [c["id"] for c in assistant["tool_calls"]] == ["call_a", "call_b"]
    assert assistant["tool_calls"][1]["function"] == {"index": 1, "name": "get_lab", "arguments": {"name": "créatinine"}}
    assert (first["tool_call_id"], first["tool_name"]) == ("call_a", "get_lab")
    assert (second["tool_call_id"], second["content"]) == ("call_b", "1.9")
    assert body["think"] is False and body["truncate"] is False and body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_predict": 512, "num_ctx": 8192, "seed": 7}


def test_openai_encodes_tool_round_trip():
    """protects=OpenAI wire form: arguments as JSON strings, results linked by tool_call_id;
    fails_when=arguments are objects or results lose their call id;
    no_claim=provider acceptance of `seed`."""
    body = json.loads(_backend_for("openai_compat").encode(_round_trip_call(Reasoning.ON)))
    assistant, first, _ = body["messages"][2:]
    assert assistant["tool_calls"][0] == {"id": "call_a", "type": "function", "function": {"name": "get_lab", "arguments": '{"name": "lactate"}'}}
    assert (first["role"], first["tool_call_id"], first["name"]) == ("tool", "call_a", "get_lab")
    assert body["reasoning_effort"] == "medium" and body["max_tokens"] == 512 and body["seed"] == 7


def test_encoding_is_deterministic_utf8():
    """protects=criterion 10 precondition: encode is a pure function of the call;
    fails_when=two encodes of the same call differ or non-ASCII is escaped inconsistently;
    no_claim=none."""
    for adapter in ("ollama", "openai_compat"):
        backend = _backend_for(adapter)
        call = _round_trip_call(None)
        assert backend.encode(call) == backend.encode(call)
        assert "créatinine".encode() in backend.encode(call)
