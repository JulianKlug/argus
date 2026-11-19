"""Synthetic DecisionWindows for renderer tests: no store, no real patient data.

    make_window(current={Domain: [rows]}, past=..., decisions={DecisionCategory: [rows]})
        └─ frames with COMMON_COLUMNS (+ DECISION_COLUMNS), as D1/D2 would return them
"""

import tomllib
from datetime import timedelta

import pandas as pd

from argus.data import (
    ClinicalRecord, CurrentEncounter, DecisionAction, DecisionCategory, DecisionWindow, Demographics, Domain,
    PatientSnapshot,
)
from argus.data.domains import COMMON_COLUMNS, DECISION_COLUMNS
from argus.rendering.config import DEFAULT_CONFIG, RenderingConfig, parse_config

AT = pd.Timestamp("2180-03-14 12:00")
WINDOW = timedelta(hours=6)
START = AT - WINDOW
ADMITTED = AT - timedelta(days=2, hours=5, minutes=10)
ICU_IN = AT - timedelta(days=1, hours=8, minutes=40)
SUBJECT = 1
CURRENT_HADM = 20
PAST_HADM = 10
STAY = 200
TOKENIZERS = ("chars", "words", "bytes3")
HEART_RATE = 220045
LACTATE = 50813

# Columns some renderer paths read on every frame of a domain, even when empty.
_DOMAIN_COLUMNS = {
    Domain.CHARTED_OBSERVATIONS: ("itemid", "value", "valuenum", "valueuom"),
    Domain.LABS: ("itemid", "value", "valuenum", "valueuom"),
    Domain.NOTES: ("text",),
    Domain.ENCOUNTERS: ("admittime", "dischtime"),
}

COUNTERS = {
    "chars": len,
    "words": lambda text: len(text.split()),
    "bytes3": lambda text: len(text.encode()) // 3 + 1,
}


def row(table: str, source_row: int, time: pd.Timestamp, label: str | None = None, **extra) -> dict:
    """One record with common columns; available at `time` unless overridden."""
    return {
        "source_table": table, "source_row": source_row, "hadm_id": CURRENT_HADM, "stay_id": STAY,
        "event_time": time, "available_time": time, "is_pending": False, "is_truncated": False,
        "label": label, **extra,
    }


def decision(table: str, source_row: int, time: pd.Timestamp, action: DecisionAction, label: str, **extra) -> dict:
    links = {"subcategory": None, "order_table": None, "order_row": pd.NA} | extra
    return row(table, source_row, time, label, action=action, **links)


def frame(rows: list[dict], columns: tuple[str, ...] = ()) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    for column in (*COMMON_COLUMNS, *columns):
        if column not in result:
            result[column] = pd.Series(dtype=object)
    result["event_time"] = pd.to_datetime(result["event_time"])
    result["available_time"] = pd.to_datetime(result["available_time"])
    result["is_pending"] = result["is_pending"].astype(bool)
    result["is_truncated"] = result["is_truncated"].astype(bool)
    result["source_row"] = result["source_row"].astype("int64")
    result["hadm_id"] = result["hadm_id"].astype("Int64")
    return result


def make_window(
    current: dict[Domain, list[dict]] | None = None,
    past: dict[Domain, list[dict]] | None = None,
    decisions: dict[DecisionCategory, list[dict]] | None = None,
    race: str | None = "RACE-SENTINEL",
) -> DecisionWindow:
    current, past, decisions = current or {}, past or {}, decisions or {}

    def record(rows: dict[Domain, list[dict]]) -> ClinicalRecord:
        return ClinicalRecord({d: frame(rows.get(d, []), _DOMAIN_COLUMNS.get(d, ())) for d in Domain})

    snapshot = PatientSnapshot(
        subject_id=SUBJECT,
        at=AT.to_pydatetime(),
        demographics=Demographics(
            sex="F", age_at_t=67, care_years=(2014, 2016), race=race, language="ENGLISH",
            insurance="Medicare", marital_status="MARRIED",
        ),
        current_encounter=CurrentEncounter(
            hadm_id=CURRENT_HADM, admittime=ADMITTED.to_pydatetime(), admission_type="EW EMER.",
            admission_location="EMERGENCY ROOM", edregtime=None, stay_id=STAY, icu_intime=ICU_IN.to_pydatetime(),
            first_careunit="Medical Intensive Care Unit (MICU)",
        ),
        current=record(current),
        past=record(past),
    )
    frames = {c: frame(decisions.get(c, []), DECISION_COLUMNS) for c in DecisionCategory}
    return DecisionWindow(snapshot=snapshot, start=START.to_pydatetime(), _frames=frames)


def config(**overrides) -> RenderingConfig:
    """Shipped config with test tokenizers; `overrides` patch [budget] / [selection] / top-level keys."""
    raw = tomllib.loads(DEFAULT_CONFIG.read_text())
    raw["tokenizers"] = list(TOKENIZERS)
    for key, value in overrides.items():
        section = "budget" if key in raw["budget"] else "selection" if key in raw["selection"] else None
        (raw[section] if section else raw)[key] = value
    return parse_config(raw, frozenset(TOKENIZERS))
