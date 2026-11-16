"""LLM errors. Messages carry backend names and record ids, never request content (L11)."""


class LlmError(Exception):
    """Base for every LLM interface failure."""

    def __init__(self, message: str, record_id: str | None = None) -> None:
        super().__init__(message if record_id is None else f"{message} (record {record_id})")
        self.record_id = record_id


class BackendUnavailableError(LlmError):
    """Backend unreachable or timed out."""


class CapabilityError(LlmError):
    """Request or agent needs a capability the backend lacks; nothing was sent."""


class ContextOverflowError(LlmError):
    """Request does not fit the context, or the server evaluated a different prompt length."""


class StructuredOutputError(LlmError):
    """No schema-valid answer after the retry."""


class ToolCallError(LlmError):
    """No valid tool calls after the retry."""


class PolicyViolationError(LlmError):
    """Data governance check failed; nothing was sent."""


class BackendMismatchError(LlmError):
    """Server reports a model, digest or response shape other than the pinned one."""
