"""Provider-independent LLM types: what agents send and receive."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class DataPolicy(Enum):
    LOCAL = "LOCAL"
    ZERO_RETENTION = "ZERO_RETENTION"


class FinishReason(Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"


class Capability(Enum):
    JSON_SCHEMA = "JSON_SCHEMA"
    TOOLS = "TOOLS"
    REASONING = "REASONING"
    SEED = "SEED"


class Reasoning(Enum):
    ON = "ON"
    OFF = "OFF"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: type[BaseModel]


@dataclass(frozen=True)
class Request:
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()


@dataclass(frozen=True)
class CallContext:
    run_id: str
    step_id: str
    parent_record_id: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    latency_s: float
    queue_s: float | None
    load_s: float | None
    gpu_s: float | None
    cost: float | None
    currency: str | None


@dataclass(frozen=True)
class Capabilities:
    context_tokens: int
    supported: frozenset[Capability]


@dataclass(frozen=True)
class Completion:
    message: Message
    reasoning: str | None
    finish: FinishReason
    usage: Usage
    record_id: str


@dataclass(frozen=True)
class Settings:
    """Sampling settings of one agent; recorded with every call."""

    temperature: float
    seed: int | None
    max_tokens: int
    reasoning: Reasoning
