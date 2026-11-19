"""Public value objects of the renderer."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from argus.data import DecisionAction, DecisionCategory, Domain


class Section(Enum):
    """Rendered sections, in text order."""

    HEADER = "header"
    DECISIONS = "decisions"
    CURRENT_STATE = "current_state"
    TRENDS = "trends"
    RECENT_EVIDENCE = "recent_evidence"
    NOTES = "notes"
    PAST_ADMISSIONS = "past_admissions"
    COVERAGE = "coverage"


MANDATORY_SECTIONS = (Section.HEADER, Section.DECISIONS, Section.COVERAGE)
OPTIONAL_SECTIONS = tuple(s for s in Section if s not in MANDATORY_SECTIONS)


class RefKind(Enum):
    RECORD = "record"  # one source row
    DECISION = "decision"  # one D2 decision occurrence
    TREND = "trend"  # statistics derived from several rows


@dataclass(frozen=True, order=True)
class SourceRow:
    table: str  # Table.source_name
    row: int  # source_row


@dataclass(frozen=True)
class EvidenceRef:
    kind: RefKind
    sources: tuple[SourceRow, ...]  # exactly 1 for RECORD and DECISION; >= 1 for TREND
    category: DecisionCategory | None = None  # DECISION only
    action: DecisionAction | None = None  # DECISION only
    time: datetime | None = None  # DECISION only: event time, so occurrences of one row are distinct keys


@dataclass(frozen=True)
class RenderedWindow:
    text: str
    refs: Mapping[str, EvidenceRef]  # prompt ID -> reference
    section_tokens: Mapping[Section, Mapping[str, int]]  # section -> backend id -> tokens
    total_tokens: Mapping[str, int]  # backend id -> tokens of `text`
    omissions: Mapping[Section, int]  # optional section -> items not rendered for budget
    rendered_items: Mapping[Section, int]  # optional section -> items rendered (measurement)
    domain_omissions: Mapping[Domain, int]  # recent evidence: domain -> items not rendered for budget
    config_digest: str  # SHA256 of canonical config
    content_hash: str  # SHA256 of UTF-8 `text`
