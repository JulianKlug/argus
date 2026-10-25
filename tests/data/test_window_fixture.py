"""Integration: SnapshotEngine.get_window over the fixture store and the shipped catalog."""

from datetime import timedelta, timezone

import pandas as pd
import pytest

from argus.data import DecisionAction, DecisionCategory, Domain, SnapshotEngine
from argus.data.domains import ACTION, EVENT_TIME, SOURCE_ROW, SOURCE_TABLE
from conftest import AT, HOUR, MINUTE, SUBJECT, WINDOW, WINDOW_START, Kind

A = DecisionAction
EXPECTED = {
    DecisionCategory.MEDICATION: {
        ("prescriptions", 0, A.ORDER, AT),  # boundary row: poe ordertime == AT
        ("inputevents", 3, A.START, WINDOW_START + HOUR),
        ("inputevents", 4, A.CHANGE, WINDOW_START + 2 * HOUR),
        ("inputevents", 4, A.STOP, WINDOW_START + 3 * HOUR),
    },
    DecisionCategory.DEVICE_SETTING: {
        ("chartevents", 5, A.CONTINUE, WINDOW_START + HOUR),
        ("chartevents", 6, A.CHANGE, WINDOW_START + 2 * HOUR),
        ("procedureevents", 0, A.START, AT),  # ventilation started at AT, still running
    },
    DecisionCategory.TEST_ORDER: {
        ("labevents", 0, A.ORDER, AT - HOUR),
        ("labevents", 2, A.ORDER, AT - 30 * MINUTE),  # result pending
        ("microbiologyevents", 1, A.ORDER, AT - HOUR),  # two pending isolates, one specimen
    },
    DecisionCategory.PROCEDURE: set(),  # fixture has no lines, airway or restraints
    DecisionCategory.CONSULT: set(),
    DecisionCategory.PATIENT_MOVEMENT: set(),
}


@pytest.fixture(scope="module")
def engine(fixture_store) -> SnapshotEngine:
    return SnapshotEngine(fixture_store.path)


def _decisions(frame: pd.DataFrame) -> set[tuple]:
    return set(zip(frame[SOURCE_TABLE], frame[SOURCE_ROW], frame[ACTION], frame[EVENT_TIME]))


def _snapshot_keys(snapshot) -> set[tuple[str, int]]:
    frames = [record.get(d) for record in (snapshot.current, snapshot.past) for d in Domain]
    rows = pd.concat(frames, ignore_index=True)
    return set(zip(rows[SOURCE_TABLE], rows[SOURCE_ROW]))


def test_window_lists_expected_decisions(engine):
    """protects=end-to-end decision list for the fixture subject (store -> availability -> decisions);
    fails_when=a planted decision is missing, a non-decision (emar, diluent, poe medication New,
    earlier lab, sentinel) appears, or an action/time differs;
    no_claim=real MIMIC coverage (real_data tests)."""
    window = engine.get_window(SUBJECT, AT.to_pydatetime(), WINDOW)

    assert window.start == WINDOW_START.to_pydatetime()
    for category, expected in EXPECTED.items():
        assert _decisions(window.get(category)) == expected, category


def test_decisions_are_visible_rows_inside_window(engine, fixture_store):
    """protects=spec criterion 1 + 7: decisions only from rows the snapshot shows, start < event_time <= at,
    never emar or sentinel rows;
    fails_when=decision derived from a row hidden at `at`, outside the window, or from eMAR;
    no_claim=clinical correctness of the categories."""
    window = engine.get_window(SUBJECT, AT.to_pydatetime(), WINDOW)
    visible = _snapshot_keys(window.snapshot)
    sentinels = fixture_store.data.marked[Kind.SENTINEL]

    for category in DecisionCategory:
        frame = window.get(category)
        keys = set(zip(frame[SOURCE_TABLE], frame[SOURCE_ROW]))
        assert keys <= visible, category
        assert not keys & sentinels, category
        assert "emar" not in set(frame[SOURCE_TABLE]), category
        assert ((frame[EVENT_TIME] > WINDOW_START) & (frame[EVENT_TIME] <= AT)).all(), category


def test_window_snapshot_equals_get_snapshot(engine):
    """protects=get_window evidence is exactly get_snapshot at the same `at`;
    fails_when=window path builds the snapshot differently (extra or missing rows);
    no_claim=field values beyond row identity."""
    at = AT.to_pydatetime()
    assert _snapshot_keys(engine.get_window(SUBJECT, at).snapshot) == _snapshot_keys(engine.get_snapshot(SUBJECT, at))


def test_window_rejects_non_positive_window_and_aware_time(engine):
    """protects=input validation of get_window;
    fails_when=zero/negative window or tz-aware `at` accepted;
    no_claim=error messages."""
    with pytest.raises(ValueError):
        engine.get_window(SUBJECT, AT.to_pydatetime(), timedelta(0))
    with pytest.raises(ValueError):
        engine.get_window(SUBJECT, AT.to_pydatetime().replace(tzinfo=timezone.utc))
