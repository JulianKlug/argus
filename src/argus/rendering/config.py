"""Load and validate config/rendering.toml. Strict: unknown keys or bad values raise."""

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path

from argus.data import Domain
from argus.data.catalog import POE_KEY_SEPARATOR
from argus.data.domains import Table
from argus.llm.config import DEFAULT_CONFIG as LLM_CONFIG
from argus.rendering.errors import RenderingConfigError

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "rendering.toml"

# Tables a core variable may come from: both carry itemid, valuenum and valueuom.
CORE_TABLES = frozenset({Table.CHARTEVENTS.source_name, Table.LABEVENTS.source_name})

_TOP_KEYS = {"format_version", "tokenizers", "demographics", "budget", "selection", "core_variables"}
_BUDGET_KEYS = {"total", "current_state", "trends", "notes", "past_admissions"}
_SELECTION_KEYS = {
    "recent_lookback_hours", "trend_lookback_hours", "trend_min_points", "past_dx_per_admission",
    "recent_domain_priority", "note_section_priority", "contentless_orders", "documented_delay_minutes",
}
_CORE_KEYS = {"table", "itemid", "label"}


class Demographics(Enum):
    MINIMAL = "minimal"  # age, sex
    FULL = "full"  # + race, language, insurance, marital status


@dataclass(frozen=True)
class CoreVariable:
    table: str
    itemid: int
    label: str


@dataclass(frozen=True)
class Budget:
    total: int
    current_state: int
    trends: int
    notes: int
    past_admissions: int


@dataclass(frozen=True)
class RenderingConfig:
    format_version: int
    tokenizers: tuple[str, ...]
    demographics: Demographics
    budget: Budget
    recent_lookback: timedelta
    trend_lookback: timedelta
    trend_min_points: int
    documented_delay: timedelta  # show availability time when known this much after event time
    past_dx_per_admission: int
    recent_domain_priority: tuple[Domain, ...]
    note_section_priority: tuple[str, ...]
    contentless_orders: tuple[str, ...]  # poe 'order_type|order_subtype' skipped without detail
    core_variables: tuple[CoreVariable, ...]

    def digest(self) -> str:
        """SHA256 of the canonical JSON form; identifies the experimental condition."""
        canonical = json.dumps(asdict(self), sort_keys=True, default=_canonical, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_config(path: Path | None = None, backends: frozenset[str] | None = None) -> RenderingConfig:
    """Parse and validate; `backends` defaults to the backend IDs of config/llm.toml."""
    try:
        raw = tomllib.loads((path or DEFAULT_CONFIG).read_text())
    except tomllib.TOMLDecodeError as e:
        raise RenderingConfigError(f"rendering config: {e}") from None

    known_backends = backends if backends is not None else frozenset(tomllib.loads(LLM_CONFIG.read_text())["backends"])
    return parse_config(raw, known_backends)


def parse_config(raw: dict, backends: frozenset[str]) -> RenderingConfig:
    _reject_unknown(raw, _TOP_KEYS, "rendering")
    budget_raw = _table(raw, "budget")
    selection = _table(raw, "selection")
    _reject_unknown(budget_raw, _BUDGET_KEYS, "budget")
    _reject_unknown(selection, _SELECTION_KEYS, "selection")

    budget = Budget(**{key: _positive(budget_raw.get(key), f"budget.{key}") for key in _BUDGET_KEYS})
    quotas = budget.current_state + budget.trends + budget.notes + budget.past_admissions
    if quotas > budget.total:
        raise RenderingConfigError("budget: quotas sum above total")

    return RenderingConfig(
        format_version=_positive(raw.get("format_version"), "format_version"),
        tokenizers=_tokenizers(raw.get("tokenizers"), backends),
        demographics=_enum(Demographics, raw.get("demographics"), "demographics"),
        budget=budget,
        recent_lookback=timedelta(hours=_positive(selection.get("recent_lookback_hours"), "recent_lookback_hours")),
        trend_lookback=timedelta(hours=_positive(selection.get("trend_lookback_hours"), "trend_lookback_hours")),
        trend_min_points=_positive(selection.get("trend_min_points"), "trend_min_points"),
        documented_delay=timedelta(minutes=_positive(selection.get("documented_delay_minutes"), "documented_delay_minutes")),
        past_dx_per_admission=_positive(selection.get("past_dx_per_admission"), "past_dx_per_admission"),
        recent_domain_priority=_domains(selection.get("recent_domain_priority")),
        note_section_priority=_sections(selection.get("note_section_priority")),
        contentless_orders=_order_keys(selection.get("contentless_orders")),
        core_variables=_core_variables(raw.get("core_variables")),
    )


# ── validation helpers ────────────────────────────────────────────────────


def _tokenizers(value, backends: frozenset[str]) -> tuple[str, ...]:
    names = _unique_list(value, "tokenizers", str)
    unknown = set(names) - backends
    if unknown:
        raise RenderingConfigError(f"tokenizers: unknown backend IDs {sorted(unknown)}")
    return tuple(names)


def _domains(value) -> tuple[Domain, ...]:
    """Notes are owned by the Notes section, never listed as recent evidence."""
    domains = tuple(_enum(Domain, name, "recent_domain_priority") for name in _unique_list(value, "recent_domain_priority", str))
    if Domain.NOTES in domains:
        raise RenderingConfigError("recent_domain_priority: notes belong to the Notes section")
    return domains


def _sections(value) -> tuple[str, ...]:
    names = _unique_list(value, "note_section_priority", str)
    if any(not name.strip() or name != name.upper() for name in names):
        raise RenderingConfigError("note_section_priority: names must be non-empty upper case")
    return tuple(names)


def _order_keys(value) -> tuple[str, ...]:
    """'order_type|order_subtype', both parts non-empty, e.g. 'General Care|Activity'."""
    keys = _unique_list(value, "contentless_orders", str)
    for key in keys:
        parts = key.split(POE_KEY_SEPARATOR)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise RenderingConfigError(f"contentless_orders: {key!r} is not 'order_type{POE_KEY_SEPARATOR}order_subtype'")
    return tuple(keys)


def _core_variables(value) -> tuple[CoreVariable, ...]:
    """Unique (table, itemid) and labels; tables limited to CORE_TABLES."""
    if not isinstance(value, list) or not value:
        raise RenderingConfigError("core_variables: non-empty list required")

    variables = []
    for index, entry in enumerate(value):
        where = f"core_variables[{index}]"
        if not isinstance(entry, dict):
            raise RenderingConfigError(f"{where}: table required")
        _reject_unknown(entry, _CORE_KEYS, where)
        if entry.get("table") not in CORE_TABLES:
            raise RenderingConfigError(f"{where}.table must be one of {sorted(CORE_TABLES)}")
        label = entry.get("label")
        if not isinstance(label, str) or not label.strip():
            raise RenderingConfigError(f"{where}.label required")
        variables.append(CoreVariable(entry["table"], _positive(entry.get("itemid"), f"{where}.itemid"), label))

    keys = [(v.table, v.itemid) for v in variables]
    labels = [v.label for v in variables]
    if len(set(keys)) != len(keys) or len(set(labels)) != len(labels):
        raise RenderingConfigError("core_variables: duplicate variable or label")
    return tuple(variables)


def _table(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise RenderingConfigError(f"[{key}] table required")
    return value


def _unique_list(value, where: str, kind: type) -> list:
    if not isinstance(value, list) or not value or not all(isinstance(v, kind) for v in value):
        raise RenderingConfigError(f"{where}: non-empty list of {kind.__name__} required")
    if len(set(value)) != len(value):
        raise RenderingConfigError(f"{where}: duplicates")
    return value


def _enum(kind: type[Enum], value, where: str):
    try:
        return kind(value)
    except ValueError:
        raise RenderingConfigError(f"{where}: unknown value {value!r}") from None


def _positive(value, where: str) -> int:
    # bool is an int subclass: `true` must not pass as 1.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RenderingConfigError(f"{where} must be a positive integer")
    return value


def _reject_unknown(entry: dict, allowed: set[str], where: str) -> None:
    """A typo must not silently fall back to a default."""
    unknown = set(entry) - allowed
    if unknown:
        raise RenderingConfigError(f"{where}: unknown keys {sorted(unknown)}")


def _canonical(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, timedelta):
        return value.total_seconds()
    raise TypeError(type(value).__name__)
