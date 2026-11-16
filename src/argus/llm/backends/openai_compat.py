"""OpenAI-compatible /chat/completions adapter.

Today it serves Ollama's /v1. Metadata, digest and the pre-count come from the native Ollama API
(`control_url`): /v1 cannot disable truncation, and its render-only mode truncates as well, so a
/v1 pre-count would agree with a silently truncated evaluation. The native render
(`truncate: false`) counts the full prompt; a truncated /v1 evaluation then fails the post-check.

A remote zero-retention backend will need its own count API: without one the backend is refused
(no exact count, L3).
"""

import json

from argus.llm.backends.base import BackendCall, BackendResult, ServerInfo, WireTool
from argus.llm.backends.http import HttpSession, HttpStatusError
from argus.llm.backends.ollama import OllamaBackend, dumps, map_status_error
from argus.llm.errors import BackendMismatchError, CapabilityError
from argus.llm.types import FinishReason, Message, Reasoning, Role, ToolCall

CHAT_PATH = "/chat/completions"
FINISH_REASONS = {"stop": FinishReason.STOP, "length": FinishReason.LENGTH, "tool_calls": FinishReason.TOOL_CALLS}
REASONING_EFFORT = {Reasoning.ON: "medium", Reasoning.OFF: "none"}
SCHEMA_NAME = "answer"


class OpenAiCompatBackend:
    def __init__(self, session: HttpSession, model: str, native: OllamaBackend | None) -> None:
        self._session = session
        self._model = model
        self._native = native

    def server_info(self) -> ServerInfo:
        if self._native is None:
            raise CapabilityError(f"model {self._model}: no control API, no digest or exact token count")
        return self._native.server_info()

    def encode(self, call: BackendCall) -> bytes:
        return dumps(self._body(call))

    def complete(self, body: bytes) -> BackendResult:
        try:
            data = self._session.post_json(CHAT_PATH, body)
        except HttpStatusError as e:
            raise map_status_error(e, self._model) from None
        return _decode(data)

    def count_request_tokens(self, call: BackendCall) -> int:
        return self._native.count_request_tokens(call)

    def count_text_tokens(self, text: str) -> int:
        return self._native.count_text_tokens(text)

    def _body(self, call: BackendCall) -> dict:
        settings = call.settings
        body = {"model": self._model, "messages": [_encode_message(m) for m in call.messages]}
        if call.tools:
            body["tools"] = [_encode_tool(t) for t in call.tools]
        if call.schema is not None:
            body["response_format"] = _response_format(call.schema)
        if call.reasoning is not None:
            body["reasoning_effort"] = REASONING_EFFORT[call.reasoning]
        body |= {"temperature": settings.temperature, "max_tokens": settings.max_tokens, "stream": False}
        if settings.seed is not None:
            body["seed"] = settings.seed
        return body


def _response_format(schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": SCHEMA_NAME, "schema": schema, "strict": True}}


def _encode_message(message: Message) -> dict:
    out = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
            for c in message.tool_calls
        ]
    if message.role is Role.TOOL:
        out["tool_call_id"] = message.tool_call_id
        out["name"] = message.tool_name
    return out


def _encode_tool(tool: WireTool) -> dict:
    return {"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters}}


def _decode(data: dict) -> BackendResult:
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError):
        raise BackendMismatchError("response without choices") from None

    message = choice.get("message") or {}
    calls, malformed = [], set()
    for i, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function", {})
        call_id = raw.get("id") or f"call_{i}"
        try:
            arguments = json.loads(function.get("arguments") or "")
        except ValueError:
            arguments = None
        if not isinstance(arguments, dict):
            malformed.add(call_id)
            arguments = {}
        calls.append(ToolCall(call_id, function.get("name", ""), arguments))

    finish = FINISH_REASONS.get(choice.get("finish_reason"))
    if finish is None:
        raise BackendMismatchError(f"unexpected finish_reason {choice.get('finish_reason')!r}")
    if calls and finish is FinishReason.STOP:
        finish = FinishReason.TOOL_CALLS

    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return BackendResult(
        message=Message(Role.ASSISTANT, message.get("content") or "", tool_calls=tuple(calls)),
        reasoning=message.get("reasoning") or message.get("reasoning_content") or None,
        finish=finish,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage.get("completion_tokens", 0),
        reasoning_tokens=details.get("reasoning_tokens"),
        queue_s=None,
        load_s=None,
        gpu_s=None,
        raw_response=data,
        malformed_tool_calls=frozenset(malformed),
    )
