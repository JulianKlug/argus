"""Load and validate config/llm.toml: backends (pinned) and per-agent settings."""

import re
import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from argus.llm.types import Capability, DataPolicy, Reasoning, Settings

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "llm.toml"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 7
DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_REQUEST_TIMEOUT_S = 600.0
DIGEST = re.compile(r"[0-9a-f]{64}")


class Adapter(Enum):
    OLLAMA = "ollama"
    OPENAI_COMPAT = "openai_compat"


@dataclass(frozen=True)
class BackendConfig:
    name: str
    adapter: Adapter
    base_url: str
    model: str
    digest: str
    policy: DataPolicy
    context_tokens: int
    count_tolerance: int
    connect_timeout_s: float
    request_timeout_s: float
    control_url: str | None             # Ollama native API behind an openai_compat endpoint
    capabilities: frozenset[Capability] | None  # allowlist; restricts probed set, never grants
    gpu_cost_per_s: float | None
    currency: str | None
    verification: Path | None           # ZERO_RETENTION note
    api_key_env: str | None


@dataclass(frozen=True)
class AgentConfig:
    name: str
    backend: BackendConfig
    requires: frozenset[Capability]
    settings: Settings


_BACKEND_KEYS = {
    "adapter", "base_url", "model", "digest", "policy", "context_tokens", "count_tolerance",
    "connect_timeout_s", "request_timeout_s", "control_url", "capabilities", "gpu_cost_per_s",
    "currency", "verification", "api_key_env",
}
_AGENT_KEYS = {"backend", "requires", "temperature", "seed", "max_tokens", "reasoning"}


def load_agent(agent: str, path: Path | None = None) -> AgentConfig:
    """Agent `agent` with its backend; ValueError on any malformed or unknown entry."""
    raw = tomllib.loads((path or DEFAULT_CONFIG).read_text())
    agents = raw.get("agents", {})
    if agent not in agents:
        raise ValueError(f"agent {agent!r} not in LLM config")

    entry = agents[agent]
    _reject_unknown(entry, _AGENT_KEYS, f"agents.{agent}")

    backends = raw.get("backends", {})
    if entry.get("backend") not in backends:
        raise ValueError(f"agents.{agent}: unknown backend {entry.get('backend')!r}")

    settings = Settings(
        temperature=float(entry.get("temperature", DEFAULT_TEMPERATURE)),
        seed=entry.get("seed", DEFAULT_SEED),
        max_tokens=_positive_int(entry.get("max_tokens"), f"agents.{agent}.max_tokens"),
        reasoning=Reasoning(entry.get("reasoning", Reasoning.OFF.value)),
    )
    return AgentConfig(
        name=agent,
        backend=_backend(entry["backend"], backends[entry["backend"]]),
        requires=frozenset(Capability(c) for c in entry.get("requires", [])),
        settings=settings,
    )


def _backend(name: str, entry: dict) -> BackendConfig:
    _reject_unknown(entry, _BACKEND_KEYS, f"backends.{name}")

    digest = entry.get("digest", "")
    if not DIGEST.fullmatch(digest):
        raise ValueError(f"backends.{name}.digest must be the full 64-hex sha256")

    # Cost needs a unit: a number without currency is not comparable across arms.
    if ("gpu_cost_per_s" in entry) != ("currency" in entry):
        raise ValueError(f"backends.{name}: gpu_cost_per_s and currency go together")

    capabilities = entry.get("capabilities")
    verification = entry.get("verification")
    return BackendConfig(
        name=name,
        adapter=Adapter(entry["adapter"]),
        base_url=entry["base_url"].rstrip("/"),
        model=entry["model"],
        digest=digest,
        policy=DataPolicy(entry["policy"]),
        context_tokens=_positive_int(entry.get("context_tokens"), f"backends.{name}.context_tokens"),
        count_tolerance=int(entry.get("count_tolerance", 0)),
        connect_timeout_s=float(entry.get("connect_timeout_s", DEFAULT_CONNECT_TIMEOUT_S)),
        request_timeout_s=float(entry.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)),
        control_url=entry["control_url"].rstrip("/") if "control_url" in entry else None,
        capabilities=None if capabilities is None else frozenset(Capability(c) for c in capabilities),
        gpu_cost_per_s=entry.get("gpu_cost_per_s"),
        currency=entry.get("currency"),
        verification=None if verification is None else REPO_ROOT / verification,
        api_key_env=entry.get("api_key_env"),
    )


def _reject_unknown(entry: dict, allowed: set[str], where: str) -> None:
    """A typo must not silently drop a restriction."""
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")


def _positive_int(value, where: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value
