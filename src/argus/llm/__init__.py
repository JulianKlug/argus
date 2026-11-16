"""Provider-independent LLM interface: the only way agents reach a model.

Spec: docs/spec_llm_backbone.md.
"""

from argus.llm.client import LlmClient
from argus.llm.errors import (
    BackendMismatchError, BackendUnavailableError, CapabilityError, ContextOverflowError, LlmError,
    PolicyViolationError, StructuredOutputError, ToolCallError,
)
from argus.llm.types import (
    CallContext, Capabilities, Capability, Completion, DataPolicy, FinishReason, Message, Reasoning, Request,
    Role, Settings, ToolCall, ToolSpec, Usage,
)

__all__ = [
    "BackendMismatchError",
    "BackendUnavailableError",
    "CallContext",
    "Capabilities",
    "Capability",
    "CapabilityError",
    "Completion",
    "ContextOverflowError",
    "DataPolicy",
    "FinishReason",
    "LlmClient",
    "LlmError",
    "Message",
    "PolicyViolationError",
    "Reasoning",
    "Request",
    "Role",
    "Settings",
    "StructuredOutputError",
    "ToolCall",
    "ToolCallError",
    "ToolSpec",
    "Usage",
]
