"""Integration: SnapshotEngine over the synthetic Parquet fixture store (criteria 3-9, 14)."""

from datetime import timezone

import pandas as pd
import pytest

import argus.data
from argus.data import (
    AmbiguousEncounterError, Domain, NotInIcuStayError, SnapshotEngine, SubjectNotFoundError,
)
from argus.data.domains import (
    AVAILABLE_TIME, COMMON_COLUMNS, EVENT_TIME, HADM_ID, IS_PENDING, IS_TRUNCATED, SOURCE_ROW, SOURCE_TABLE,
    TABLE_DOMAIN,
)
from conftest import (
    AT, CURRENT_HADM, INFUSION_AMOUNT, INFUSION_AT, MINUTE, PENDING_LAB_STORED, PENDING_LAB_VALUE, SUBJECT,
    SUBJECT_AMBIGUOUS, SUBJECT_NO_ICU, UNKNOWN_SUBJECT, Kind,
)

CODED_TABLES = ("diagnoses_icd", "drgcodes", "procedures_icd", "hcpcsevents", "discharge")
CURRENT_DISCHARGE_NOTE_ROW = 2  # note D3 in conftest._notes


@pytest.fixture(scope="module")
def engine(fixture_store) -> SnapshotEngine:
    return SnapshotEngine(fixture_store.path)


def _all_rows(snapshot) -> pd.DataFrame:
    """Every returned row, current and past, all domains."""
    frames = [record.get(domain) for record in (snapshot.current, snapshot.past) for domain in Domain]
    return pd.concat(frames, ignore_index=True)


def _keys(frame: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(frame[SOURCE_TABLE], frame[SOURCE_ROW]))


def _row(snapshot, domain: Domain, table: str, source_row: int) -> pd.Series:
    frame = pd.concat([snapshot.current.get(domain), snapshot.past.get(domain)])
    match = frame[(frame[SOURCE_TABLE] == table) & (frame[SOURCE_ROW] == source_row)]
    assert len(match) == 1, (table, source_row)
    return match.iloc[0]


def test_errors_for_unknown_subject_outside_icu_and_aware_time(engine):
    """protects=criterion 3 input validation;
    fails_when=snapshot returned for unknown subject, non-ICU `at`, ambiguous admission or tz-aware `at`;
    no_claim=error messages."""
    with pytest.raises(SubjectNotFoundError):
        engine.get_snapshot(UNKNOWN_SUBJECT, AT.to_pydatetime())
    with pytest.raises(NotInIcuStayError):
        engine.get_snapshot(SUBJECT, pd.Timestamp("2180-01-16").to_pydatetime())  # after ICU outtime
    with pytest.raises(NotInIcuStayError):
        engine.get_snapshot(SUBJECT_NO_ICU, AT.to_pydatetime())
    with pytest.raises(AmbiguousEncounterError):
        engine.get_snapshot(SUBJECT_AMBIGUOUS, AT.to_pydatetime())
    with pytest.raises(ValueError):
        engine.get_snapshot(SUBJECT, AT.to_pydatetime().replace(tzinfo=timezone.utc))


def test_fixture_plants_sentinel_for_every_rule_table(fixture_store):
    """protects=criterion 4 coverage: the sentinel check below spans every table in the rules table;
    fails_when=a table is added to TABLE_DOMAIN without a sentinel in the fixture;
    no_claim=sentinel realism."""
    planted = {table for table, _ in fixture_store.data.marked[Kind.SENTINEL]}
    assert {t.source_name for t in TABLE_DOMAIN} <= planted


def test_no_sentinel_returned(engine, fixture_store):
    """protects=criterion 4: nothing available after `at` reaches agents;
    fails_when=any table rule admits a row available at `at` + 1 min (or banned outright);
    no_claim=leaks through fields of visible rows (see redaction tests)."""
    returned = _keys(_all_rows(engine.get_snapshot(SUBJECT, AT.to_pydatetime())))
    assert returned & fixture_store.data.marked[Kind.SENTINEL] == set()


def test_boundary_rows_returned(engine, fixture_store):
    """protects=criterion 5 inclusive cutoff available_time == at;
    fails_when=strict `<` used anywhere;
    no_claim=sub-minute precision."""
    snapshot = engine.get_snapshot(SUBJECT, AT.to_pydatetime())
    rows = _all_rows(snapshot)

    assert fixture_store.data.marked[Kind.BOUNDARY] <= _keys(rows)
    assert (rows[AVAILABLE_TIME] <= AT).all() and (rows[EVENT_TIME] <= AT).all()


def test_pending_lab_resolves_after_storetime(engine):
    """protects=criterion 6 lab drawn before `at`, resulted after;
    fails_when=value visible while pending, or row/value missing once stored;
    no_claim=other result fields (unit-tested)."""
    lab_row = 2  # labevent_id 3 in conftest._labs

    pending = _row(engine.get_snapshot(SUBJECT, AT.to_pydatetime()), Domain.LABS, "labevents", lab_row)
    assert pending[IS_PENDING] and pd.isna(pending["valuenum"])

    later = (PENDING_LAB_STORED + MINUTE).to_pydatetime()
    resulted = _row(engine.get_snapshot(SUBJECT, later), Domain.LABS, "labevents", lab_row)
    assert not resulted[IS_PENDING] and resulted["valuenum"] == PENDING_LAB_VALUE


def test_running_infusion_prorated(engine):
    """protects=criterion 7 infusion 10:00-16:00, 600 mL, at 12:00;
    fails_when=amount != 200, or endtime/statusdescription visible, or not flagged truncated;
    no_claim=ingredientevents (same rule, unit-tested)."""
    row = _row(engine.get_snapshot(SUBJECT, INFUSION_AT.to_pydatetime()), Domain.INPUTS, "inputevents", 0)
    assert row["amount"] == pytest.approx(INFUSION_AMOUNT / 3)
    assert pd.isna(row["endtime"]) and pd.isna(row["statusdescription"]) and row[IS_TRUNCATED]


def test_current_admission_outcome_and_codes_withheld(engine):
    """protects=criterion 8 no hindsight on current admission outcome, stay end, death or coding;
    fails_when=any outcome field set, `dod` column anywhere, or coded/discharge rows of current hadm;
    no_claim=outcome inferable from clinical content (e.g. comfort-care orders)."""
    snapshot = engine.get_snapshot(SUBJECT, AT.to_pydatetime())
    encounters = snapshot.current.get(Domain.ENCOUNTERS)

    admission = encounters[(encounters[SOURCE_TABLE] == "admissions") & (encounters[HADM_ID] == CURRENT_HADM)].iloc[0]
    for column in ("dischtime", "deathtime", "discharge_location", "hospital_expire_flag"):
        assert pd.isna(admission[column]), column
    stay = encounters[encounters[SOURCE_TABLE] == "icustays"].iloc[0]
    for column in ("outtime", "los", "last_careunit"):
        assert pd.isna(stay[column]), column

    rows = _all_rows(snapshot)
    assert "dod" not in rows.columns
    coded = rows[rows[SOURCE_TABLE].isin(CODED_TABLES)]
    assert not (coded[HADM_ID] == CURRENT_HADM).any()
    assert len(coded) > 0  # prior admission codes still present


def test_current_discharge_note_banned_despite_early_storetime(engine):
    """protects=criterion 9 D3 hard ban;
    fails_when=discharge note of current admission returned when stored before `at`;
    no_claim=discharge_detail (covered by sentinel test)."""
    notes = _all_rows(engine.get_snapshot(SUBJECT, AT.to_pydatetime()))
    discharge = notes[notes[SOURCE_TABLE] == "discharge"]
    assert CURRENT_DISCHARGE_NOTE_ROW not in discharge[SOURCE_ROW].tolist()
    assert len(discharge) == 1  # prior admission note D1


def test_snapshot_header_and_split(engine):
    """protects=demographics/current encounter assembly and current/past partition;
    fails_when=header fields drawn from wrong admission, or prior rows in `current`;
    no_claim=real-data split (criterion 12)."""
    snapshot = engine.get_snapshot(SUBJECT, AT.to_pydatetime())

    assert snapshot.demographics.sex == "F" and snapshot.demographics.age_at_t == 62
    assert snapshot.demographics.marital_status == "WIDOWED"  # current, not prior admission
    assert snapshot.current_encounter.hadm_id == CURRENT_HADM and snapshot.current_encounter.stay_id == 1000
    admittime = pd.Timestamp(snapshot.current_encounter.admittime)
    for domain in Domain:
        assert (snapshot.current.get(domain)[EVENT_TIME] >= admittime).all()
        assert (snapshot.past.get(domain)[EVENT_TIME] < admittime).all()


def test_empty_domain_keeps_schema(engine):
    """protects=ClinicalRecord.get returns typed empty frame with common columns;
    fails_when=missing domain raises or loses columns;
    no_claim=column dtypes across datasets."""
    frame = engine.get_snapshot(SUBJECT, AT.to_pydatetime()).past.get(Domain.MICROBIOLOGY)
    assert frame.empty
    assert list(frame.columns[: len(COMMON_COLUMNS)]) == list(COMMON_COLUMNS)


def test_public_api_is_spec_names_only():
    """protects=criterion 14 encapsulation (+ decision window spec criterion 11);
    fails_when=driver/policy/catalog internals exported;
    no_claim=import-time side effects."""
    assert sorted(argus.data.__all__) == sorted([
        "SnapshotEngine", "PatientSnapshot", "ClinicalRecord", "Demographics", "CurrentEncounter", "Domain",
        "SnapshotError", "SubjectNotFoundError", "NotInIcuStayError", "AmbiguousEncounterError",
        "DecisionWindow", "DecisionCategory", "DecisionAction",
    ])
