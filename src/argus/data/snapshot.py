"""Snapshot value objects returned to agents."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from argus.data.domains import DecisionCategory, Domain


@dataclass(frozen=True)
class Demographics:
    sex: str
    age_at_t: int  # anchor_age + (at.year - anchor_year)
    race: str | None
    language: str | None
    insurance: str | None
    marital_status: str | None


@dataclass(frozen=True)
class CurrentEncounter:
    hadm_id: int
    admittime: datetime
    admission_type: str
    admission_location: str | None
    edregtime: datetime | None
    stay_id: int
    icu_intime: datetime
    first_careunit: str


@dataclass(frozen=True)
class ClinicalRecord:
    """Records of one period (current or past admissions), one DataFrame per domain."""

    _frames: Mapping[Domain, pd.DataFrame]

    def get(self, domain: Domain) -> pd.DataFrame:
        """Copy, so callers cannot alter the snapshot; empty frame keeps the schema."""
        return self._frames[domain].copy()


@dataclass(frozen=True)
class PatientSnapshot:
    subject_id: int
    at: datetime
    demographics: Demographics
    current_encounter: CurrentEncounter
    current: ClinicalRecord  # event_time >= current_encounter.admittime
    past: ClinicalRecord  # event_time <  current_encounter.admittime


@dataclass(frozen=True)
class DecisionWindow:
    """Decisions taken in (start, at], with the snapshot at `at` as evidence."""

    snapshot: PatientSnapshot  # also contains the decision rows
    start: datetime  # at - window, exclusive
    _frames: Mapping[DecisionCategory, pd.DataFrame]

    def get(self, category: DecisionCategory) -> pd.DataFrame:
        """Copy; one row per decision, `(source_table, source_row)` joins back to the snapshot."""
        return self._frames[category].copy()
