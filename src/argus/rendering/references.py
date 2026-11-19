"""Opaque prompt IDs for evidence and the map back to source rows.

    record    E + base32(sha256("{table}:{row}"))[:8]                                   e.g. E3HP0RT2C
    decision  D + base32(sha256("{table}:{row}:{category}:{action}:{event_time}"))[:8]
    trend     T + per-window index in render order                                     e.g. T1

Record and decision IDs are stable across windows for a fixed store; a collision raises.
"""

import base64
import hashlib
import re
from datetime import datetime

import pandas as pd

from argus.data import DecisionAction, DecisionCategory
from argus.rendering.errors import RenderingError
from argus.rendering.types import EvidenceRef, RefKind, SourceRow

ID_LENGTH = 8  # base32 chars: 40 bits
RECORD_PREFIX = "E"
DECISION_PREFIX = "D"
TREND_PREFIX = "T"
_BRACKETED = re.compile(r"\[([A-Z0-9]+)\]")


def record_id(source: SourceRow) -> str:
    return RECORD_PREFIX + _digest(f"{source.table}:{source.row}")


def decision_id(source: SourceRow, category: DecisionCategory, action: DecisionAction, event_time: datetime) -> str:
    key = f"{source.table}:{source.row}:{category.value}:{action.value}:{pd.Timestamp(event_time).isoformat()}"
    return DECISION_PREFIX + _digest(key)


def _digest(key: str) -> str:
    return base64.b32encode(hashlib.sha256(key.encode()).digest()).decode()[:ID_LENGTH]


class References:
    """Registry of every ID issued while rendering one window; one ID -> one key."""

    def __init__(self) -> None:
        self._refs: dict[str, EvidenceRef] = {}
        self._trends = 0

    def record(self, source: SourceRow) -> str:
        return self._register(record_id(source), EvidenceRef(RefKind.RECORD, (source,)))

    def decision(self, source: SourceRow, category: DecisionCategory, action: DecisionAction, event_time: datetime) -> str:
        ref = EvidenceRef(RefKind.DECISION, (source,), category, action)
        return self._register(decision_id(source, category, action, event_time), ref)

    def trend(self, sources: tuple[SourceRow, ...]) -> str:
        """Next window-local trend ID; callers issue them in render order."""
        if not sources:
            raise ValueError("trend needs at least one source")
        self._trends += 1
        return self._register(f"{TREND_PREFIX}{self._trends}", EvidenceRef(RefKind.TREND, tuple(sorted(sources))))

    def used_in(self, text: str) -> dict[str, EvidenceRef]:
        """Refs whose bracketed ID occurs in `text`, sorted by ID."""
        cited = set(_BRACKETED.findall(text))
        return {ref_id: ref for ref_id, ref in sorted(self._refs.items()) if ref_id in cited}

    def _register(self, ref_id: str, ref: EvidenceRef) -> str:
        """Same key twice is fine (citations); a different key under the same ID is a collision."""
        known = self._refs.setdefault(ref_id, ref)
        if known != ref:
            raise RenderingError(f"reference ID collision ({ref.kind.value})")
        return ref_id
