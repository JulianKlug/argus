"""Backend protocol: what LlmClient needs from any provider adapter.

    client ── encode(call) ──► bytes ── complete(bytes) ──► BackendResult
           ── count_request_tokens(call), count_text_tokens(text)
           ── server_info()

The client records the exact bytes it hands to `complete`, so records equal what was sent.
"""

from dataclasses import dataclass
from typing import Protocol

from argus.llm.types import Capability, FinishReason, Message, Reasoning, Settings


@dataclass(frozen=True)
class WireTool:
    """Tool as offered on the wire: JSON schema, no pydantic class."""

    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class BackendCall:
    messages: tuple[Message, ...]
    tools: tuple[WireTool, ...]
    schema: dict | None
    settings: Settings
    reasoning: Reasoning | None     # None: backend has no reasoning switch, send nothing
    context_tokens: int


@dataclass(frozen=True)
class ServerInfo:
    version: str
    digest: str | None
    remote_host: str | None
    capabilities: frozenset[Capability]     # reported by the server, before the live probe
    context_length: int | None              # model's nominal context


@dataclass(frozen=True)
class BackendResult:
    message: Message
    reasoning: str | None
    finish: FinishReason
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    queue_s: float | None
    load_s: float | None
    gpu_s: float | None
    raw_response: dict
    malformed_tool_calls: frozenset[str]    # call ids whose arguments were not a JSON object


class Backend(Protocol):
    def server_info(self) -> ServerInfo: ...

    def encode(self, call: BackendCall) -> bytes: ...

    def complete(self, body: bytes) -> BackendResult: ...

    def count_request_tokens(self, call: BackendCall) -> int: ...

    def count_text_tokens(self, text: str) -> int: ...
