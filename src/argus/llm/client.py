"""LlmClient: the only way agents reach a model.

    for_agent ─► config ─► policy (static) ─► connect ─► policy (server) ─► capabilities
    generate / structured
        └─ attempt: pre_count ─► budget check ─► send ─► post-check ─► CallRecord
             └─ invalid tool calls / schema-invalid answer ─► one corrective retry (L4, L5)
"""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from argus.llm import policy
from argus.llm.backends import Backend, BackendCall, BackendResult, ServerInfo, WireTool, connect
from argus.llm.config import AgentConfig, load_agent
from argus.llm.errors import (
    BackendMismatchError, CapabilityError, ContextOverflowError, LlmError, StructuredOutputError, ToolCallError,
)
from argus.llm.sink import CallRecord, CallSink, message_to_dict
from argus.llm.types import (
    CallContext, Capabilities, Capability, Completion, FinishReason, Message, Reasoning, Request, Role,
    Settings, ToolSpec, Usage,
)

T = TypeVar("T", bound=BaseModel)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # first answer + one corrective retry

STRUCTURED_FEEDBACK = (
    "Your previous answer does not match the required JSON schema:\n{errors}\n"
    "Answer again with only a JSON object that matches the schema."
)
TOOL_FEEDBACK_INVALID = "Tool call rejected, not executed: {errors}. Correct it and call again."
TOOL_FEEDBACK_SIBLING = "Not executed: another call in this turn was invalid. Repeat all calls of this turn."
TOOL_FEEDBACK_UNPARSED = "Your previous turn produced neither an answer nor a parsable tool call. Answer or call a tool."

# Live capability probes (L6): synthetic prompts through the normal adapter path, no patient data.
PROBE_MAX_TOKENS = 256
PROBE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
PROBE_SCHEMA_PROMPT = 'Answer with the JSON object {"ok": true}.'
PROBE_TOOL = WireTool(
    "get_lab", "Latest value of one lab test",
    {"type": "object", "properties": {"name": {"type": "string", "description": "lab test name"}}, "required": ["name"]},
)
PROBE_TOOL_PROMPT = "Call the get_lab tool with name=lactate. Do not answer before calling it."

# Probe results per (url, model, digest, capability): one live probe per process.
_probe_cache: dict[tuple[str, str, str, Capability], bool] = {}


class LlmClient:
    """Use `for_agent`; the constructor takes already vetted parts."""

    def __init__(self, agent: AgentConfig, backend: Backend, server: ServerInfo, capabilities: Capabilities, sink: CallSink) -> None:
        self._agent = agent
        self._backend = backend
        self._server = server
        self._sink = sink
        self.capabilities = capabilities

    @classmethod
    def for_agent(cls, agent: str, config: Path | None = None, *, transport=None) -> "LlmClient":
        """Fail closed: any policy or capability problem raises before an inference request.
        `transport` replaces the network in tests (fake HTTP transport)."""
        agent_config = load_agent(agent, config)
        backend_config = agent_config.backend

        # Static checks: zero requests.
        policy.check_environment(os.environ)
        policy.check_endpoint(backend_config, date.today())
        sink = CallSink(policy.check_run_dir(os.environ))
        policy.check_record_file(sink.path)
        connection = connect(backend_config, policy.check_response_status, transport)
        policy.check_transport(connection.trust_env, connection.follow_redirects)

        # Server metadata: zero inference requests.
        server = connection.backend.server_info()
        policy.check_server(backend_config, server.digest, server.remote_host)

        capabilities = _capabilities(agent_config, connection.backend, server)
        return cls(agent_config, connection.backend, server, capabilities, sink)

    def generate(self, request: Request, ctx: CallContext) -> Completion:
        """Answer or tool calls. Returned tool calls name an offered tool and pass its schema (L5)."""
        if request.tools and Capability.TOOLS not in self.capabilities.supported:
            raise CapabilityError(f"backend {self._agent.backend.name}: tools not supported")

        specs = {t.name: t for t in request.tools}
        tools = tuple(_wire_tool(t) for t in request.tools)
        messages = request.messages
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result, completion = self._attempt(messages, tools, None, ctx, attempt)
            errors = _tool_call_errors(result, specs)

            # Finished normally with nothing parsed: the server dropped a malformed tool call.
            unparsed = (
                bool(specs) and result.finish is FinishReason.STOP
                and not result.message.tool_calls and not result.message.content.strip()
            )
            if not errors and not unparsed:
                return completion

            log.warning("llm record %s: invalid tool turn", completion.record_id)
            if result.finish is FinishReason.LENGTH or attempt == MAX_ATTEMPTS:
                raise ToolCallError("no valid tool calls after retry", completion.record_id)
            feedback = (Message(Role.USER, TOOL_FEEDBACK_UNPARSED),) if unparsed else _tool_feedback(result.message, errors)
            messages = messages + (result.message,) + feedback

        raise AssertionError("unreachable")

    def structured(self, request: Request, schema: type[T], ctx: CallContext) -> tuple[T, Completion]:
        """Schema-constrained answer validated by pydantic; one corrective retry (L4)."""
        if request.tools:
            raise ValueError("structured() takes no tools; use generate()")
        if Capability.JSON_SCHEMA not in self.capabilities.supported:
            raise CapabilityError(f"backend {self._agent.backend.name}: JSON schema not supported")

        wire_schema = inline_schema(schema)
        messages = request.messages
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result, completion = self._attempt(messages, (), wire_schema, ctx, attempt)
            try:
                return schema.model_validate_json(result.message.content), completion
            except ValidationError as e:
                log.warning("llm record %s: schema-invalid answer", completion.record_id)
                if result.finish is FinishReason.LENGTH or attempt == MAX_ATTEMPTS:
                    raise StructuredOutputError("no schema-valid answer", completion.record_id) from None
                feedback = STRUCTURED_FEEDBACK.format(errors=_validation_summary(e))
                messages = messages + (result.message, Message(Role.USER, feedback))

        raise AssertionError("unreachable")

    def count_text_tokens(self, text: str) -> int:
        return self._backend.count_text_tokens(text)

    def count_request_tokens(self, request: Request, schema: type | None = None) -> int:
        """Full prompt as the server renders it: template, history, tools, schema."""
        tools = tuple(_wire_tool(t) for t in request.tools)
        return self._backend.count_request_tokens(self._call(request.messages, tools, None if schema is None else inline_schema(schema)))

    def _call(self, messages: tuple[Message, ...], tools: tuple[WireTool, ...], schema: dict | None) -> BackendCall:
        # A backend with a reasoning switch always gets an explicit value: its default may be ON.
        has_switch = Capability.REASONING in self._server.capabilities
        return BackendCall(
            messages=messages,
            tools=tools,
            schema=schema,
            settings=self._agent.settings,
            reasoning=self._agent.settings.reasoning if has_switch else None,
            context_tokens=self.capabilities.context_tokens,
        )

    def _attempt(self, messages, tools, schema, ctx: CallContext, attempt: int) -> tuple[BackendResult, Completion]:
        """One budgeted, recorded call (L3, L11)."""
        call = self._call(messages, tools, schema)
        record = _RecordDraft(self._agent, self._server, call, ctx, attempt)
        backend = self._agent.backend

        try:
            record.pre_count = self._backend.count_request_tokens(call)
            if record.pre_count + call.settings.max_tokens > call.context_tokens:
                raise ContextOverflowError(f"backend {backend.name}: request exceeds context budget")

            record.wire = self._backend.encode(call)
            started = time.monotonic()
            result = self._backend.complete(record.wire)
            latency = time.monotonic() - started
            record.result = result

            # Silent truncation: the server evaluated a different prompt than the one budgeted.
            if abs(result.prompt_tokens - record.pre_count) > backend.count_tolerance:
                raise ContextOverflowError(f"backend {backend.name}: evaluated prompt length differs from pre-count")
        except Exception as e:
            self._sink.write(record.finish(type(e).__name__, None))
            log.error("llm record %s failed: %s", record.record_id, type(e).__name__)
            if isinstance(e, LlmError):
                raise type(e)(str(e), record.record_id) from None
            raise

        usage = self._usage(result, latency)
        self._sink.write(record.finish(None, usage))
        log.info("llm record %s: %s, %d+%d tokens", record.record_id, result.finish.value, usage.prompt_tokens, usage.completion_tokens)
        return result, Completion(result.message, result.reasoning, result.finish, usage, record.record_id)

    def _usage(self, result: BackendResult, latency_s: float) -> Usage:
        backend = self._agent.backend
        priced = result.gpu_s is not None and backend.gpu_cost_per_s is not None
        return Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            reasoning_tokens=result.reasoning_tokens,
            latency_s=latency_s,
            queue_s=result.queue_s,
            load_s=result.load_s,
            gpu_s=result.gpu_s,
            cost=result.gpu_s * backend.gpu_cost_per_s if priced else None,
            currency=backend.currency if priced else None,
        )


class _RecordDraft:
    """Collects one attempt's fields; `finish` freezes them into a CallRecord."""

    def __init__(self, agent: AgentConfig, server: ServerInfo, call: BackendCall, ctx: CallContext, attempt: int) -> None:
        self.record_id = uuid.uuid4().hex
        self.started_at = datetime.now(UTC).isoformat()
        self.pre_count: int | None = None
        self.wire: bytes | None = None
        self.result: BackendResult | None = None
        self._agent = agent
        self._server = server
        self._call = call
        self._ctx = ctx
        self._attempt = attempt

    def finish(self, error: str | None, usage: Usage | None) -> CallRecord:
        agent, server, call, result = self._agent, self._server, self._call, self.result
        settings = asdict(call.settings) | {
            "reasoning": call.settings.reasoning.value,
            "reasoning_sent": None if call.reasoning is None else call.reasoning.value,
            "context_tokens": call.context_tokens,
        }
        request = {
            "messages": [message_to_dict(m) for m in call.messages],
            "tools": [asdict(t) for t in call.tools],
            "schema": call.schema,
        }
        return CallRecord(
            record_id=self.record_id,
            run_id=self._ctx.run_id,
            step_id=self._ctx.step_id,
            parent_record_id=self._ctx.parent_record_id,
            labels=dict(self._ctx.labels),
            agent=agent.name,
            backend=agent.backend.name,
            model=agent.backend.model,
            digest=agent.backend.digest,
            server_version=server.version,
            settings=settings,
            request=request,
            wire_request=None if self.wire is None else self.wire.decode(),
            pre_count=self.pre_count,
            raw_response=None if result is None else result.raw_response,
            reasoning=None if result is None else result.reasoning,
            message=None if result is None else message_to_dict(result.message),
            finish=None if result is None else result.finish.value,
            usage=None if usage is None else asdict(usage),
            error=error,
            attempt=self._attempt,
            started_at=self.started_at,
        )


def _capabilities(agent: AgentConfig, backend: Backend, server: ServerInfo) -> Capabilities:
    """Probed ∩ configured (L6); agent requirements must hold now, never degrade later (L10)."""
    config = agent.backend
    if server.context_length is not None and config.context_tokens > server.context_length:
        raise CapabilityError(f"backend {config.name}: context_tokens above the model's context")

    # No exact count, no backend (L3): load the tokenizer now.
    backend.count_text_tokens("")

    # Reported tools are kept only if a live call parses; JSON schema is never reported, only probed.
    allowed = config.capabilities
    supported = set(server.capabilities if allowed is None else server.capabilities & allowed)
    candidates = {Capability.JSON_SCHEMA} | (supported & {Capability.TOOLS})
    for capability in candidates:
        if allowed is not None and capability not in allowed:
            continue
        key = (config.base_url, config.model, config.digest, capability)
        if key not in _probe_cache:
            _probe_cache[key] = _probe(backend, server, config.context_tokens, capability)
        if _probe_cache[key]:
            supported.add(capability)
        else:
            supported.discard(capability)

    needed = set(agent.requires)
    if agent.settings.reasoning is Reasoning.ON:
        needed.add(Capability.REASONING)
    if agent.settings.seed is not None:
        needed.add(Capability.SEED)
    missing = needed - supported
    if missing:
        raise CapabilityError(f"backend {config.name} lacks {sorted(c.value for c in missing)} for agent {agent.name}")

    return Capabilities(context_tokens=config.context_tokens, supported=frozenset(supported))


def _probe(backend: Backend, server: ServerInfo, context_tokens: int, capability: Capability) -> bool:
    """One synthetic call; True iff the answer comes back in the expected shape."""
    tool = capability is Capability.TOOLS
    call = BackendCall(
        messages=(Message(Role.USER, PROBE_TOOL_PROMPT if tool else PROBE_SCHEMA_PROMPT),),
        tools=(PROBE_TOOL,) if tool else (),
        schema=None if tool else PROBE_SCHEMA,
        settings=Settings(temperature=0.0, seed=None, max_tokens=PROBE_MAX_TOKENS, reasoning=Reasoning.OFF),
        reasoning=Reasoning.OFF if Capability.REASONING in server.capabilities else None,
        context_tokens=context_tokens,
    )
    try:
        result = backend.complete(backend.encode(call))
    except BackendMismatchError:
        return False  # server refused the request shape

    if tool:
        calls = result.message.tool_calls
        return any(c.name == PROBE_TOOL.name and c.id not in result.malformed_tool_calls for c in calls)
    try:
        return isinstance(json.loads(result.message.content).get("ok"), bool)
    except (ValueError, AttributeError):
        return False


def _wire_tool(spec: ToolSpec) -> WireTool:
    return WireTool(spec.name, spec.description, inline_schema(spec.parameters))


def inline_schema(model: type[BaseModel]) -> dict:
    """JSON schema with local $refs inlined: Ollama keeps only a subset of keys per tool property,
    and a dangling $ref would hide nested fields from the model. Recursive models are refused."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node, seen: frozenset[str]):
        if isinstance(node, list):
            return [resolve(n, seen) for n in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if ref is None:
            return {k: resolve(v, seen) for k, v in node.items()}

        name = ref.removeprefix("#/$defs/")
        if name in seen:
            raise ValueError(f"recursive schema {name!r} cannot be inlined")
        rest = {k: v for k, v in node.items() if k != "$ref"}
        return resolve(defs[name], seen | {name}) | resolve(rest, seen)

    return resolve(schema, frozenset())


def _tool_call_errors(result: BackendResult, specs: dict[str, ToolSpec]) -> dict[str, str]:
    """call id -> reason, for every invalid call of the turn. Reasons never echo argument values."""
    errors = {}
    ids = [c.id for c in result.message.tool_calls]
    for call in result.message.tool_calls:
        if ids.count(call.id) > 1:
            errors[call.id] = "duplicate call id"
        elif call.id in result.malformed_tool_calls:
            errors[call.id] = "arguments are not a JSON object"
        elif call.name not in specs:
            errors[call.id] = f"unknown tool {call.name!r}; offered: {sorted(specs)}"
        else:
            try:
                specs[call.name].parameters.model_validate(call.arguments)
            except ValidationError as e:
                errors[call.id] = _validation_summary(e)
    return errors


def _tool_feedback(message: Message, errors: dict[str, str]) -> tuple[Message, ...]:
    """One TOOL message per call of the rejected turn, linked by id."""
    return tuple(
        Message(
            Role.TOOL,
            TOOL_FEEDBACK_INVALID.format(errors=errors[c.id]) if c.id in errors else TOOL_FEEDBACK_SIBLING,
            tool_call_id=c.id,
            tool_name=c.name,
        )
        for c in message.tool_calls
    )


def _validation_summary(error: ValidationError) -> str:
    """`loc: msg` lines without input values (they may be patient text)."""
    items = error.errors(include_url=False, include_context=False, include_input=False)
    return "\n".join(f"- {'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in items)
