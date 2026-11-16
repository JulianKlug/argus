"""Ollama native /api/chat adapter, plus the Ollama control API shared with openai_compat.

Request budgeting (L3) on Ollama:

    request ──► /api/chat {_debug_render_only}  ──► rendered prompt ──► GGUF tokenizer ──► pre_count
    request ──► /api/chat {truncate: false}     ──► prompt_eval_count  (must equal pre_count)

`truncate: false` turns Ollama's silent truncation into HTTP 400; the client's post-check still
catches any other path that shortens the prompt.
"""

import json
from pathlib import Path

from argus.llm.backends.base import BackendCall, BackendResult, ServerInfo, WireTool
from argus.llm.backends.http import HttpSession, HttpStatusError
from argus.llm.backends.tokenizer import GgufTokenizer, load_gguf_tokenizer
from argus.llm.errors import BackendMismatchError, CapabilityError, ContextOverflowError
from argus.llm.types import Capability, FinishReason, Message, Reasoning, Role, ToolCall

NS_PER_S = 1e9
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
CONTEXT_ERROR_MARKERS = ("exceed_context_size_error", "exceeds the available context size", "exceeds the context length")

# Ollama capability names -> ours. Seed is accepted by every Ollama model.
REPORTED_CAPABILITIES = {"tools": Capability.TOOLS, "thinking": Capability.REASONING}
ALWAYS_SUPPORTED = frozenset({Capability.SEED})

DONE_REASONS = {"stop": FinishReason.STOP, "length": FinishReason.LENGTH}


def dumps(body: dict) -> bytes:
    """Canonical wire encoding: compact, UTF-8, key order as built."""
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


class OllamaControl:
    """Ollama native metadata API: version, digest, capabilities, model blob, tokenizer."""

    def __init__(self, session: HttpSession, model: str) -> None:
        self._session = session
        self._model = model
        self._blob: Path | None = None

    def server_info(self) -> ServerInfo:
        version = self._session.get_json("/api/version")["version"]
        tags = self._session.get_json("/api/tags")["models"]
        entry = next((m for m in tags if self._model in (m.get("name"), m.get("model"))), None)
        if entry is None:
            raise BackendMismatchError(f"model {self._model} not on server")

        show = self._session.post_json("/api/show", dumps({"model": self._model}))
        reported = frozenset(REPORTED_CAPABILITIES[c] for c in show.get("capabilities", []) if c in REPORTED_CAPABILITIES)
        info = show.get("model_info", {})
        self._blob = _model_blob(show.get("modelfile", ""))
        return ServerInfo(
            version=version,
            digest=entry.get("digest"),
            remote_host=show.get("remote_host") or entry.get("remote_host"),
            capabilities=reported | ALWAYS_SUPPORTED,
            context_length=info.get(f"{info.get('general.architecture')}.context_length"),
        )

    def tokenizer(self) -> GgufTokenizer:
        """Tokenizer of the model blob found by `server_info`; fail closed without one."""
        if self._blob is None or not self._blob.is_file():
            raise CapabilityError(f"model {self._model}: GGUF blob not readable, no exact token count")
        return load_gguf_tokenizer(self._blob)


class OllamaBackend:
    def __init__(self, session: HttpSession, model: str) -> None:
        self._session = session
        self._model = model
        self._control = OllamaControl(session, model)
        self._tokenizer: GgufTokenizer | None = None

    def server_info(self) -> ServerInfo:
        return self._control.server_info()

    def encode(self, call: BackendCall) -> bytes:
        return dumps(self._body(call))

    def complete(self, body: bytes) -> BackendResult:
        try:
            data = self._session.post_json("/api/chat", body)
        except HttpStatusError as e:
            raise map_status_error(e, self._model) from None
        return _decode(data)

    def count_request_tokens(self, call: BackendCall) -> int:
        body = self._body(call) | {"_debug_render_only": True}
        try:
            data = self._session.post_json("/api/chat", dumps(body))
        except HttpStatusError as e:
            raise map_status_error(e, self._model) from None
        return self.tokenizer().count_prompt(data["_debug_info"]["rendered_template"])

    def count_text_tokens(self, text: str) -> int:
        return self.tokenizer().count(text)

    def tokenizer(self) -> GgufTokenizer:
        if self._tokenizer is None:
            self._tokenizer = self._control.tokenizer()
        return self._tokenizer

    def _body(self, call: BackendCall) -> dict:
        settings = call.settings
        options = {"temperature": settings.temperature, "num_predict": settings.max_tokens, "num_ctx": call.context_tokens}
        if settings.seed is not None:
            options["seed"] = settings.seed

        body = {"model": self._model, "messages": [_encode_message(m) for m in call.messages]}
        if call.tools:
            body["tools"] = [_encode_tool(t) for t in call.tools]
        if call.schema is not None:
            body["format"] = call.schema
        if call.reasoning is not None:
            body["think"] = call.reasoning is Reasoning.ON
        body |= {"stream": False, "truncate": False, "options": options}
        return body


def _encode_message(message: Message) -> dict:
    out = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "function": {"index": i, "name": c.name, "arguments": c.arguments}}
            for i, c in enumerate(message.tool_calls)
        ]
    if message.role is Role.TOOL:
        out["tool_name"] = message.tool_name
        out["tool_call_id"] = message.tool_call_id
    return out


def _encode_tool(tool: WireTool) -> dict:
    return {"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters}}


def _decode(data: dict) -> BackendResult:
    message = data.get("message") or {}
    calls, malformed = [], set()
    for i, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function", {})
        call_id = raw.get("id") or f"call_{i}"
        arguments = function.get("arguments")
        if not isinstance(arguments, dict):
            malformed.add(call_id)
            arguments = {}
        calls.append(ToolCall(call_id, function.get("name", ""), arguments))

    finish = DONE_REASONS.get(data.get("done_reason"))
    if finish is None:
        raise BackendMismatchError(f"unexpected done_reason {data.get('done_reason')!r}")
    if calls and finish is FinishReason.STOP:
        finish = FinishReason.TOOL_CALLS

    durations = [data.get("prompt_eval_duration"), data.get("eval_duration")]
    return BackendResult(
        message=Message(Role.ASSISTANT, message.get("content", ""), tool_calls=tuple(calls)),
        reasoning=message.get("thinking") or None,
        finish=finish,
        prompt_tokens=data["prompt_eval_count"],
        completion_tokens=data.get("eval_count", 0),
        reasoning_tokens=None,  # Ollama does not report it separately
        queue_s=None,
        load_s=_seconds(data.get("load_duration")),
        gpu_s=None if None in durations else sum(durations) / NS_PER_S,
        raw_response=data,
        malformed_tool_calls=frozenset(malformed),
    )


def map_status_error(error: HttpStatusError, model: str) -> Exception:
    """Map HTTP failures without echoing the body (it may contain request text)."""
    if error.status == HTTP_BAD_REQUEST and any(m in error.body for m in CONTEXT_ERROR_MARKERS):
        return ContextOverflowError(f"model {model}: server refused request longer than its context")
    if error.status == HTTP_NOT_FOUND:
        return BackendMismatchError(f"model {model} not found on server")
    return BackendMismatchError(f"model {model}: server rejected request (HTTP {error.status})")


def _model_blob(modelfile: str) -> Path | None:
    """First `FROM /path` line of `ollama show --modelfile`: the weights blob (second is the projector)."""
    for line in modelfile.splitlines():
        if line.startswith("FROM /"):
            return Path(line.removeprefix("FROM ").strip())
    return None


def _seconds(ns: int | None) -> float | None:
    return None if ns is None else ns / NS_PER_S
