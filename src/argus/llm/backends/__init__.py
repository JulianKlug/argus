"""Provider adapters. Only this package speaks HTTP or knows provider URL paths."""

import os
from collections.abc import Callable
from dataclasses import dataclass

from argus.llm.backends.base import Backend, BackendCall, BackendResult, ServerInfo, WireTool
from argus.llm.backends.http import HttpSession
from argus.llm.backends.ollama import OllamaBackend
from argus.llm.backends.openai_compat import OpenAiCompatBackend
from argus.llm.config import Adapter, BackendConfig

__all__ = ["Backend", "BackendCall", "BackendResult", "Connection", "ServerInfo", "WireTool", "connect"]


@dataclass(frozen=True)
class Connection:
    backend: Backend
    trust_env: bool           # HTTP client settings, vetted by policy
    follow_redirects: bool


def connect(config: BackendConfig, guard: Callable[[int, str], None], transport=None) -> Connection:
    """Adapter for `config`. `guard` vets every HTTP status; `transport` is a test seam (fake HTTP)."""

    def session(url: str, headers: dict[str, str] | None = None) -> HttpSession:
        return HttpSession(
            config.name, url, config.connect_timeout_s, config.request_timeout_s, guard, transport, headers,
        )

    if config.adapter is Adapter.OLLAMA:
        main = session(config.base_url)
        return Connection(OllamaBackend(main, config.model), main.trust_env, main.follow_redirects)

    headers = None
    if config.api_key_env:
        key = os.environ.get(config.api_key_env)
        if not key:
            raise ValueError(f"backend {config.name}: {config.api_key_env} not set")
        headers = {"Authorization": f"Bearer {key}"}

    main = session(config.base_url, headers)
    native = None if config.control_url is None else OllamaBackend(session(config.control_url), config.model)
    return Connection(OpenAiCompatBackend(main, config.model, native), main.trust_env, main.follow_redirects)
