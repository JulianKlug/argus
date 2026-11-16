"""Call records: one JSONL line per attempt in $ARGUS_RUN_DIR/llm_calls.jsonl (owner-only).

A record holds everything needed to rebuild the call without inference: the domain request,
the exact wire body sent, settings, the raw response and usage.
"""

import json
import os
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path

from argus.llm.types import Message, Role, ToolCall

RECORD_FILE = "llm_calls.jsonl"


@dataclass(frozen=True)
class CallRecord:
    record_id: str
    run_id: str
    step_id: str
    parent_record_id: str | None
    labels: dict[str, str]
    agent: str
    backend: str
    model: str
    digest: str
    server_version: str
    settings: dict
    request: dict           # messages, tools, schema as sent (domain form)
    wire_request: str       # exact body bytes sent, UTF-8
    pre_count: int | None
    raw_response: dict | None
    reasoning: str | None
    message: dict | None
    finish: str | None
    usage: dict | None
    error: str | None       # error class name only, never content
    rejection: str | None   # validation failure of the answer (no input values); None = accepted
    attempt: int
    started_at: str


class CallSink:
    """Appends to a descriptor vetted by policy (`open_record_file`); never reopens by path."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        weakref.finalize(self, os.close, fd)

    def write(self, record: CallRecord) -> None:
        # Key order is kept: schema property order reaches the prompt (criterion 10).
        data = (json.dumps(asdict(record), ensure_ascii=False) + "\n").encode()
        while data:
            data = data[os.write(self._fd, data):]  # os.write may write fewer bytes than given


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def message_to_dict(message: Message) -> dict:
    return {
        "role": message.role.value,
        "content": message.content,
        "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
    }


def message_from_dict(data: dict) -> Message:
    return Message(
        role=Role(data["role"]),
        content=data["content"],
        tool_calls=tuple(ToolCall(c["id"], c["name"], c["arguments"]) for c in data["tool_calls"]),
        tool_call_id=data["tool_call_id"],
        tool_name=data["tool_name"],
    )
