"""Assemble a PatientSnapshot from per-table visible rows. No time logic here."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from argus.data.availability import Encounter, Timeline
from argus.data.catalog import DecisionCatalog
from argus.data.decisions import SOURCE_TABLES, decisions
from argus.data.domains import TABLE_DOMAIN, Domain
from argus.data.snapshot import ClinicalRecord, CurrentEncounter, DecisionWindow, Demographics, PatientSnapshot
from argus.data.store import MimicStore

DEFAULT_WINDOW = timedelta(hours=6)


def _optional(value):
    """pandas missing value -> None."""
    return None if pd.isna(value) else value


def _to_datetime(value) -> datetime | None:
    return None if pd.isna(value) else pd.Timestamp(value).to_pydatetime()


class SnapshotEngine:
    """Leak-free view of one patient at one instant."""

    def __init__(self, store_dir: Path) -> None:
        self._store = MimicStore(store_dir)
        self._catalog = DecisionCatalog()

    def get_snapshot(self, subject_id: int, at: datetime) -> PatientSnapshot:
        return self._snapshot(self._timeline(subject_id, at), subject_id, at)

    def get_window(self, subject_id: int, at: datetime, window: timedelta = DEFAULT_WINDOW) -> DecisionWindow:
        """Decisions taken in (at - window, at] plus the snapshot at `at`."""
        if window <= timedelta(0):
            raise ValueError("`window` must be positive")

        timeline = self._timeline(subject_id, at)
        snapshot = self._snapshot(timeline, subject_id, at)
        start = pd.Timestamp(at - window)
        frames = {table: timeline.visible(table) for table in SOURCE_TABLES}
        return DecisionWindow(snapshot=snapshot, start=start.to_pydatetime(), _frames=decisions(frames, self._catalog, start))

    def _timeline(self, subject_id: int, at: datetime) -> Timeline:
        if at.tzinfo is not None:
            raise ValueError("`at` must be naive (MIMIC shifted time)")

        return Timeline(self._store, subject_id, pd.Timestamp(at))

    def _snapshot(self, timeline: Timeline, subject_id: int, at: datetime) -> PatientSnapshot:
        patient = timeline.patient()
        encounter = timeline.current_encounter()

        # Domain frame = concatenation of its tables' visible rows, then split by period.
        current, past = {}, {}
        admittime = encounter.admission["admittime"]
        for domain in Domain:
            frame = self._domain_frame(timeline, domain)
            current[domain], past[domain] = timeline.split(frame, admittime)

        return PatientSnapshot(
            subject_id=subject_id,
            at=at,
            demographics=self._demographics(timeline, patient, encounter),
            current_encounter=self._current_encounter(encounter),
            current=ClinicalRecord(current),
            past=ClinicalRecord(past),
        )

    def _domain_frame(self, timeline: Timeline, domain: Domain) -> pd.DataFrame:
        frames = [timeline.visible(t) for t, d in TABLE_DOMAIN.items() if d is domain]
        return pd.concat(frames, ignore_index=True)

    def _demographics(self, timeline: Timeline, patient: pd.Series, encounter: Encounter) -> Demographics:
        admission = encounter.admission
        return Demographics(
            sex=patient["gender"],
            age_at_t=timeline.age_at(patient),
            race=_optional(admission["race"]),
            language=_optional(admission["language"]),
            insurance=_optional(admission["insurance"]),
            marital_status=_optional(admission["marital_status"]),
        )

    def _current_encounter(self, encounter: Encounter) -> CurrentEncounter:
        admission, icustay = encounter.admission, encounter.icustay
        return CurrentEncounter(
            hadm_id=int(admission["hadm_id"]),
            admittime=_to_datetime(admission["admittime"]),
            admission_type=admission["admission_type"],
            admission_location=_optional(admission["admission_location"]),
            edregtime=_to_datetime(admission["edregtime"]),
            stay_id=int(icustay["stay_id"]),
            icu_intime=_to_datetime(icustay["intime"]),
            first_careunit=icustay["first_careunit"],
        )
