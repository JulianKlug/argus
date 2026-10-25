"""Unit tests: one availability rule per table, on in-memory frames (no Parquet)."""

import pandas as pd
import pytest

import build_store
from argus.data.availability import Timeline, _eod, _prorate
from argus.data.domains import (
    AVAILABLE_TIME, EVENT_TIME, IS_PENDING, IS_TRUNCATED, SOURCE_ROW, Table,
)
from argus.data.errors import AmbiguousEncounterError, NotInIcuStayError
from argus.data.store import DROPPED_COLUMNS
from conftest import COLUMNS

AT = pd.Timestamp("2180-01-10 12:00")
MINUTE = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
SUBJECT = 1
PRIOR_HADM = 100
CURRENT_HADM = 200
STAY = 1000
ADMITTIME = pd.Timestamp("2180-01-08 08:00")


def _frame(table: Table, rows: list[dict]) -> pd.DataFrame:
    """Rows typed like MimicStore output: build_store type maps, `_row`, `label`."""
    columns = [c for c in COLUMNS[table.value] if c not in DROPPED_COLUMNS]
    frame = pd.DataFrame(rows, columns=columns)
    types = build_store.TABLE_TYPES.get(table.value, {})
    for column in columns:
        dtype = types.get(column)
        if column in build_store.TIMESTAMP_COLUMNS:
            frame[column] = pd.to_datetime(frame[column]).astype("datetime64[us]")
        elif dtype == "DOUBLE" or column in build_store.DOUBLE_COLUMNS:
            frame[column] = frame[column].astype("Float64")
        elif column in build_store.BIGINT_COLUMNS:
            frame[column] = frame[column].astype("Int64")
    frame.insert(0, "_row", pd.array(range(len(frame)), dtype="Int64"))
    frame["label"] = "x"
    return frame


def _admission(hadm_id: int, admittime, dischtime, **extra) -> dict:
    return dict(subject_id=SUBJECT, hadm_id=hadm_id, admittime=admittime, dischtime=dischtime,
                admission_type="URGENT", **extra)


DEFAULT_ROWS = {
    Table.PATIENTS: [dict(subject_id=SUBJECT, gender="F", anchor_age=60, anchor_year=2178,
                          anchor_year_group="2014 - 2016", dod=AT + DAY)],
    Table.ADMISSIONS: [
        _admission(PRIOR_HADM, pd.Timestamp("2179-06-01"), pd.Timestamp("2179-06-10"),
                   discharge_location="HOME", hospital_expire_flag=0),
        _admission(CURRENT_HADM, ADMITTIME, AT + 5 * DAY, deathtime=AT + 5 * DAY, discharge_location="DIED",
                   hospital_expire_flag=1, edregtime=ADMITTIME - HOUR, edouttime=ADMITTIME + HOUR),
    ],
    Table.ICUSTAYS: [dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, stay_id=STAY, first_careunit="MICU",
                          last_careunit="SICU", intime=AT - DAY, outtime=AT + DAY, los=2.0)],
}


class _StubStore:
    """MimicStore stand-in serving hand-made frames; absent tables are empty."""

    def __init__(self, rows: dict[Table, list[dict]]) -> None:
        self._frames = {table: _frame(table, table_rows) for table, table_rows in rows.items()}

    def rows(self, table: Table, subject_id: int) -> pd.DataFrame:
        if table not in self._frames:
            return _frame(table, [])
        return self._frames[table].copy()


def _timeline(at: pd.Timestamp = AT, **tables: list[dict]) -> Timeline:
    """Timeline over DEFAULT_ROWS overridden by `tables` (keyed by Table name), encounter resolved."""
    rows = DEFAULT_ROWS | {Table[name]: table_rows for name, table_rows in tables.items()}
    timeline = Timeline(_StubStore(rows), SUBJECT, at)
    timeline.current_encounter()
    return timeline


def _rows(timeline: Timeline, table: Table) -> list[int]:
    return timeline.visible(table)[SOURCE_ROW].tolist()


STAY_IDS = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, stay_id=STAY)


# ── charted: max(charttime, storetime) ──────────────────────────────────────


@pytest.mark.parametrize("table", [Table.CHARTEVENTS, Table.DATETIMEEVENTS, Table.OUTPUTEVENTS])
def test_charted_rows_follow_later_of_charttime_and_storetime(table):
    """protects=charted row visible only once both charted and stored;
    fails_when=row filtered on charttime or storetime alone, or boundary excluded;
    no_claim=correct storetime semantics in MIMIC itself."""
    rows = [
        dict(charttime=AT - HOUR, storetime=AT, **STAY_IDS),  # boundary: stored exactly at
        dict(charttime=AT - HOUR, storetime=AT + MINUTE, **STAY_IDS),  # stored after at
        dict(charttime=AT + MINUTE, storetime=AT - HOUR, **STAY_IDS),  # storetime < charttime
        dict(charttime=AT - HOUR, storetime=None, **STAY_IDS),  # null storetime -> charttime
    ]
    timeline = _timeline(**{table.name: rows})

    visible = timeline.visible(table)

    assert visible[SOURCE_ROW].tolist() == [0, 3]
    assert visible[AVAILABLE_TIME].tolist() == [AT, AT - HOUR]


def test_emar_follows_later_of_charttime_and_storetime():
    """protects=administration hidden until documented;
    fails_when=emar filtered on charttime only;
    no_claim=emar_detail linkage (see inherit test)."""
    rows = [
        dict(emar_id="a", charttime=AT - MINUTE, storetime=AT, subject_id=SUBJECT, hadm_id=CURRENT_HADM),
        dict(emar_id="b", charttime=AT - MINUTE, storetime=AT + MINUTE, subject_id=SUBJECT, hadm_id=CURRENT_HADM),
    ]
    assert _rows(_timeline(EMAR=rows), Table.EMAR) == [0]


def test_detail_rows_inherit_parent_visibility():
    """protects=detail rows (emar_detail) carry no independent time and follow their parent;
    fails_when=detail of a hidden parent leaks, or detail of a visible parent is dropped;
    no_claim=all four detail tables (same code path, covered in fixture tests)."""
    emar = [
        dict(emar_id="a", charttime=AT - HOUR, storetime=AT - HOUR, subject_id=SUBJECT, hadm_id=CURRENT_HADM),
        dict(emar_id="b", charttime=AT - HOUR, storetime=AT + MINUTE, subject_id=SUBJECT, hadm_id=CURRENT_HADM),
    ]
    detail = [dict(subject_id=SUBJECT, emar_id="a", emar_seq=1), dict(subject_id=SUBJECT, emar_id="b", emar_seq=1)]
    timeline = _timeline(EMAR=emar, EMAR_DETAIL=detail)

    visible = timeline.visible(Table.EMAR_DETAIL)

    assert visible["emar_id"].tolist() == ["a"]
    assert visible[AVAILABLE_TIME].tolist() == [AT - HOUR]
    assert visible["hadm_id"].tolist() == [CURRENT_HADM]


# ── labs and microbiology: pending results ──────────────────────────────────


LAB = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, itemid=50813, value="4.2", valuenum=4.2, valueuom="mmol/L",
           ref_range_lower="0.5", ref_range_upper="2", flag="abnormal", comments="c")


def test_lab_drawn_not_resulted_is_pending_with_result_withheld():
    """protects=D12 drawn-but-not-resulted labs visible without values;
    fails_when=value/flag/comments leak before storetime, or row hidden entirely;
    no_claim=lab result timing beyond storetime."""
    timeline = _timeline(LABEVENTS=[dict(labevent_id=1, charttime=AT - 30 * MINUTE, storetime=AT + 30 * MINUTE, **LAB)])

    row = timeline.visible(Table.LABEVENTS).iloc[0]

    assert row[IS_PENDING]
    assert row[AVAILABLE_TIME] == AT - 30 * MINUTE
    for column in ("value", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "comments"):
        assert pd.isna(row[column]), column


def test_lab_resulted_at_boundary_shows_value():
    """protects=inclusive cutoff for lab results;
    fails_when=result stored exactly at `at` treated as pending;
    no_claim=rounding of MIMIC storetimes."""
    timeline = _timeline(LABEVENTS=[dict(labevent_id=1, charttime=AT - HOUR, storetime=AT, **LAB)])

    row = timeline.visible(Table.LABEVENTS).iloc[0]

    assert not row[IS_PENDING]
    assert row["valuenum"] == 4.2
    assert row[AVAILABLE_TIME] == AT


def test_lab_not_yet_drawn_is_hidden():
    """protects=future specimen invisible;
    fails_when=row visible from storetime alone;
    no_claim=order (poe) visibility of the same lab."""
    timeline = _timeline(LABEVENTS=[dict(labevent_id=1, charttime=AT + MINUTE, storetime=AT - HOUR, **LAB)])
    assert _rows(timeline, Table.LABEVENTS) == []


MICRO = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, test_itemid=90201, test_name="Blood Culture",
             org_itemid=80002, org_name="E. COLI", ab_name="CEFTRIAXONE", interpretation="S", comments="final")


def test_micro_pending_result_withheld_and_deduplicated():
    """protects=pending cultures show once per specimen test with no organism;
    fails_when=org/ab/interpretation leak, or one pending row per isolate;
    no_claim=dedup of resulted rows (kept as-is)."""
    rows = [dict(microevent_id=i, micro_specimen_id=10, charttime=AT - HOUR, storetime=AT + DAY, **MICRO)
            for i in (1, 2)]

    visible = _timeline(MICROBIOLOGYEVENTS=rows).visible(Table.MICROBIOLOGYEVENTS)

    assert len(visible) == 1
    row = visible.iloc[0]
    assert row[IS_PENDING]
    for column in ("org_itemid", "org_name", "ab_name", "interpretation", "comments"):
        assert pd.isna(row[column]), column


def test_micro_without_any_store_time_stays_pending():
    """protects=culture with null storetime and storedate never reveals results;
    fails_when=null store time treated as available;
    no_claim=how often MIMIC lacks both."""
    rows = [dict(microevent_id=1, micro_specimen_id=10, charttime=AT - 30 * DAY, **MICRO)]
    row = _timeline(MICROBIOLOGYEVENTS=rows).visible(Table.MICROBIOLOGYEVENTS).iloc[0]
    assert row[IS_PENDING] and pd.isna(row["org_name"])


def test_micro_date_only_times_use_end_of_day():
    """protects=date-only collection/store dates available at date + 24h;
    fails_when=midnight of the date used (up to 24h early);
    no_claim=true intra-day timing."""
    day = AT.normalize()
    rows = [
        dict(microevent_id=1, micro_specimen_id=10, chartdate=day - DAY, storedate=day - DAY, **MICRO),  # all EOD <= at
        dict(microevent_id=2, micro_specimen_id=11, chartdate=day, **MICRO),  # collected at day + 24h > at
    ]
    visible = _timeline(MICROBIOLOGYEVENTS=rows).visible(Table.MICROBIOLOGYEVENTS)

    assert visible[SOURCE_ROW].tolist() == [0]
    assert visible[EVENT_TIME].tolist() == [day]
    assert not visible[IS_PENDING].iloc[0]


# ── medication orders ───────────────────────────────────────────────────────


RX = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, drug="Heparin")
POE = dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, order_type="Medications", order_status="Inactive")


def test_prescription_waits_for_poe_order_and_withholds_future_stop():
    """protects=prescription visible at max(starttime, poe.ordertime), stoptime withheld;
    fails_when=backdated starttime exposes a later order, or planned stop leaks;
    no_claim=prescription edits after `at` (not in MIMIC)."""
    poe = [dict(poe_id="p1", poe_seq=1, ordertime=AT - DAY, **POE),
           dict(poe_id="p2", poe_seq=2, ordertime=AT + MINUTE, **POE)]
    rx = [dict(pharmacy_id=1, poe_id="p1", starttime=AT - HOUR, stoptime=AT + DAY, **RX),
          dict(pharmacy_id=2, poe_id="p2", starttime=AT - HOUR, stoptime=AT + DAY, **RX),
          dict(pharmacy_id=3, poe_id=None, starttime=AT, stoptime=AT - MINUTE, **RX)]

    visible = _timeline(POE=poe, PRESCRIPTIONS=rx).visible(Table.PRESCRIPTIONS)

    assert visible[SOURCE_ROW].tolist() == [0, 2]
    assert pd.isna(visible["stoptime"].iloc[0]) and visible[IS_TRUNCATED].iloc[0]
    assert visible["stoptime"].iloc[1] == AT - MINUTE and not visible[IS_TRUNCATED].iloc[1]


def test_prescription_and_poe_visible_when_subject_has_no_poe_orders():
    """protects=subjects without any poe row (485 in MIMIC-IV 3.1) still get prescriptions and an empty poe;
    fails_when=ordertime lookup over an empty poe table raises (pandas 3 Series.map on empty datetime mapper);
    no_claim=ordering of prescriptions without poe."""
    rx = [dict(pharmacy_id=1, poe_id=None, starttime=AT - HOUR, stoptime=AT - MINUTE, **RX)]
    timeline = _timeline(PRESCRIPTIONS=rx)

    assert timeline.visible(Table.PRESCRIPTIONS)[SOURCE_ROW].tolist() == [0]
    assert timeline.visible(Table.POE).empty


def test_pharmacy_withholds_future_times_and_unsettled_status():
    """protects=pharmacy row from entertime; stop/verify/expiry and status hidden until they occur;
    fails_when=status 'Discontinued' or future stoptime leaks;
    no_claim=status history (MIMIC stores only final status)."""
    rows = [
        dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, pharmacy_id=1, entertime=AT, starttime=AT + HOUR,
             stoptime=AT + DAY, verifiedtime=AT + MINUTE, expirationdate=AT + DAY, status="Discontinued"),
        dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, pharmacy_id=2, entertime=AT - DAY, starttime=AT - DAY,
             stoptime=AT - HOUR, verifiedtime=AT - DAY, status="Discontinued"),
        dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, pharmacy_id=3, entertime=AT + MINUTE, starttime=AT - HOUR),
    ]
    visible = _timeline(PHARMACY=rows).visible(Table.PHARMACY)

    assert visible[SOURCE_ROW].tolist() == [0, 1]
    running, stopped = visible.iloc[0], visible.iloc[1]
    for column in ("stoptime", "verifiedtime", "expirationdate", "status"):
        assert pd.isna(running[column]), column
    assert running[IS_TRUNCATED] and running[AVAILABLE_TIME] == AT
    assert stopped["status"] == "Discontinued" and not stopped[IS_TRUNCATED]


def test_poe_drops_status_and_withholds_future_discontinuation():
    """protects=orders visible from ordertime, final status never shown, discontinuation only once ordered;
    fails_when=order_status present, or discontinued_by points to a future order;
    no_claim=poe_detail field content."""
    poe = [dict(poe_id="p1", poe_seq=1, ordertime=AT - DAY, discontinued_by_poe_id="p2", **POE),
           dict(poe_id="p2", poe_seq=2, ordertime=AT + MINUTE, discontinue_of_poe_id="p1", **POE),
           dict(poe_id="p3", poe_seq=3, ordertime=AT - DAY, discontinued_by_poe_id="p4", **POE),
           dict(poe_id="p4", poe_seq=4, ordertime=AT, discontinue_of_poe_id="p3", **POE)]

    visible = _timeline(POE=poe).visible(Table.POE)

    assert "order_status" not in visible
    assert visible["poe_id"].tolist() == ["p1", "p3", "p4"]
    assert pd.isna(visible["discontinued_by_poe_id"].iloc[0]) and visible[IS_TRUNCATED].iloc[0]
    assert visible["discontinued_by_poe_id"].iloc[1] == "p4" and not visible[IS_TRUNCATED].iloc[1]


# ── intervals ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("table", [Table.INPUTEVENTS, Table.INGREDIENTEVENTS])
def test_running_infusion_is_prorated_and_end_withheld(table):
    """protects=D5 infusion clipped at `at`: amount delivered so far, end and status unknown;
    fails_when=full amount, endtime or 'FinishedRunning' visible mid-infusion;
    no_claim=titration within the row (linear assumption)."""
    rows = [
        dict(starttime=AT - 2 * HOUR, endtime=AT + 4 * HOUR, amount=600.0, statusdescription="FinishedRunning", **STAY_IDS),
        dict(starttime=AT - 2 * HOUR, endtime=AT - HOUR, amount=100.0, statusdescription="FinishedRunning", **STAY_IDS),
        dict(starttime=AT + MINUTE, endtime=AT + HOUR, amount=50.0, **STAY_IDS),
    ]
    visible = _timeline(**{table.name: rows}).visible(table)

    assert visible[SOURCE_ROW].tolist() == [0, 1]
    running, done = visible.iloc[0], visible.iloc[1]
    assert running["amount"] == pytest.approx(200.0)
    assert pd.isna(running["endtime"]) and pd.isna(running["statusdescription"]) and running[IS_TRUNCATED]
    assert done["amount"] == 100.0 and done["endtime"] == AT - HOUR and not done[IS_TRUNCATED]


def test_running_procedure_value_is_prorated():
    """protects=procedureevents value clipped like infusions;
    fails_when=full planned duration visible;
    no_claim=value semantics per itemid."""
    rows = [dict(starttime=AT - HOUR, endtime=AT + 3 * HOUR, value=240.0, statusdescription="FinishedRunning",
                 itemid=225792, **STAY_IDS)]
    row = _timeline(PROCEDUREEVENTS=rows).visible(Table.PROCEDUREEVENTS).iloc[0]
    assert row["value"] == pytest.approx(60.0)
    assert pd.isna(row["endtime"]) and row[IS_TRUNCATED]


def test_prorate_linear_share():
    """protects=proration arithmetic;
    fails_when=fraction uses wrong reference times;
    no_claim=clinical validity of linear delivery."""
    start, end = pd.Series([pd.Timestamp("2180-01-10 10:00")]), pd.Series([pd.Timestamp("2180-01-10 16:00")])
    result = _prorate(pd.Series([600.0], dtype="Float64"), start, end, pd.Timestamp("2180-01-10 12:00"))
    assert result.iloc[0] == pytest.approx(200.0)


def test_eod_adds_24_hours():
    """protects=date-only rule `date + 24h`;
    fails_when=date used at midnight;
    no_claim=timezone handling (MIMIC times are naive)."""
    assert _eod(pd.Series([pd.Timestamp("2180-01-10")])).iloc[0] == pd.Timestamp("2180-01-11")


# ── notes, outpatient, coded data ───────────────────────────────────────────


def test_discharge_notes_only_from_other_admissions():
    """protects=D3 current-admission discharge summary hard-banned, prior ones available once stored;
    fails_when=current note visible via early storetime, or null-hadm note assumed safe;
    no_claim=hindsight inside radiology reports."""
    note = dict(subject_id=SUBJECT, note_type="DS", note_seq=1, text="t")
    rows = [dict(note_id="a", hadm_id=PRIOR_HADM, charttime=AT - 30 * DAY, storetime=AT - 29 * DAY, **note),
            dict(note_id="b", hadm_id=CURRENT_HADM, charttime=AT - HOUR, storetime=AT - HOUR, **note),
            dict(note_id="c", hadm_id=None, charttime=AT - 30 * DAY, storetime=AT - 29 * DAY, **note),
            dict(note_id="d", hadm_id=PRIOR_HADM, charttime=AT - 30 * DAY, storetime=AT + MINUTE, **note)]
    assert _timeline(DISCHARGE=rows).visible(Table.DISCHARGE)["note_id"].tolist() == ["a"]


def test_omr_available_end_of_chartdate():
    """protects=outpatient measurements dated `d` available at d + 24h;
    fails_when=same-day omr visible at noon;
    no_claim=omr intra-day timing."""
    rows = [dict(subject_id=SUBJECT, chartdate=AT.normalize(), seq_num=1, result_name="BMI", result_value="30"),
            dict(subject_id=SUBJECT, chartdate=AT.normalize() - DAY, seq_num=1, result_name="BMI", result_value="29")]
    visible = _timeline(OMR=rows).visible(Table.OMR)
    assert visible[SOURCE_ROW].tolist() == [1]
    assert visible[AVAILABLE_TIME].iloc[0] == AT.normalize()


@pytest.mark.parametrize("table", [Table.DIAGNOSES_ICD, Table.DRGCODES])
def test_codes_available_at_owning_discharge(table):
    """protects=billing codes only for discharged admissions, timed at dischtime;
    fails_when=current admission codes leak, or codes timed earlier than discharge;
    no_claim=actual coding delay after discharge."""
    rows = [dict(subject_id=SUBJECT, hadm_id=PRIOR_HADM, seq_num=1, icd_code="0389", icd_version=9,
                 drg_code="871", description="x"),
            dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, seq_num=1, icd_code="0389", icd_version=9,
                 drg_code="871", description="x")]
    rows = [{k: v for k, v in r.items() if k in COLUMNS[table.value]} for r in rows]
    visible = _timeline(**{table.name: rows}).visible(table)

    assert visible["hadm_id"].tolist() == [PRIOR_HADM]
    assert visible[EVENT_TIME].iloc[0] == visible[AVAILABLE_TIME].iloc[0] == pd.Timestamp("2179-06-10")


def test_codes_hidden_until_discharge_boundary():
    """protects=inclusive cutoff on dischtime for coded tables;
    fails_when=codes of an admission discharged 1 min after `at` appear;
    no_claim=overlapping admissions (rejected upstream as ambiguous)."""
    admissions = DEFAULT_ROWS[Table.ADMISSIONS] + [
        _admission(300, pd.Timestamp("2179-01-01"), AT + MINUTE), _admission(301, pd.Timestamp("2179-01-01"), AT)]
    codes = [dict(subject_id=SUBJECT, hadm_id=h, seq_num=1, icd_code="0389", icd_version=9) for h in (300, 301)]
    timeline = Timeline(_StubStore(DEFAULT_ROWS | {Table.ADMISSIONS: admissions, Table.DIAGNOSES_ICD: codes}),
                        SUBJECT, AT)
    # Encounter not resolved: 300 overlaps `at`; only the coded rule is under test.
    assert timeline.visible(Table.DIAGNOSES_ICD)["hadm_id"].tolist() == [301]


@pytest.mark.parametrize("table", [Table.PROCEDURES_ICD, Table.HCPCSEVENTS])
def test_coded_procedures_event_at_chartdate_available_at_discharge(table):
    """protects=procedure codes keep their clinical date but appear only after discharge;
    fails_when=available_time = chartdate (pre-discharge leak);
    no_claim=chartdate accuracy."""
    rows = [dict(subject_id=SUBJECT, hadm_id=h, seq_num=1, chartdate=pd.Timestamp("2179-06-03"))
            for h in (PRIOR_HADM, CURRENT_HADM)]
    visible = _timeline(**{table.name: rows}).visible(table)
    assert visible["hadm_id"].tolist() == [PRIOR_HADM]
    assert visible[EVENT_TIME].iloc[0] == pd.Timestamp("2179-06-03")
    assert visible[AVAILABLE_TIME].iloc[0] == pd.Timestamp("2179-06-10")


# ── encounters ──────────────────────────────────────────────────────────────


def test_admission_outcome_withheld_for_current_only():
    """protects=current admission discharge/death outcome hidden, prior outcome kept;
    fails_when=dischtime/deathtime/discharge_location/hospital_expire_flag of current admission visible;
    no_claim=deathtime in other tables (transfers)."""
    visible = _timeline().visible(Table.ADMISSIONS).set_index("hadm_id")

    current, prior = visible.loc[CURRENT_HADM], visible.loc[PRIOR_HADM]
    for column in ("dischtime", "deathtime", "discharge_location", "hospital_expire_flag"):
        assert pd.isna(current[column]), column
    assert current[IS_TRUNCATED] and current["edouttime"] == ADMITTIME + HOUR
    assert prior["discharge_location"] == "HOME" and not prior[IS_TRUNCATED]


def test_admission_future_edouttime_withheld():
    """protects=ED departure hidden while patient still in ED;
    fails_when=edouttime > at visible;
    no_claim=edregtime (always before admittime)."""
    admissions = [_admission(CURRENT_HADM, AT - HOUR, AT + DAY, edregtime=AT - 2 * HOUR, edouttime=AT + HOUR)]
    icustays = [dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, stay_id=STAY, first_careunit="MICU",
                     intime=AT - MINUTE, outtime=AT + DAY)]
    row = _timeline(ADMISSIONS=admissions, ICUSTAYS=icustays).visible(Table.ADMISSIONS).iloc[0]
    assert pd.isna(row["edouttime"])


def test_icustay_end_withheld_while_in_unit():
    """protects=current stay outtime/los/last_careunit hidden;
    fails_when=length of stay or discharge unit leaks;
    no_claim=icustays of other admissions."""
    row = _timeline().visible(Table.ICUSTAYS).iloc[0]
    for column in ("outtime", "los", "last_careunit"):
        assert pd.isna(row[column]), column
    assert row[IS_TRUNCATED]


def test_transfers_and_services_from_start_time():
    """protects=transfers/services visible from intime/transfertime inclusive, future outtime withheld;
    fails_when=next transfer or service visible early;
    no_claim=careunit naming."""
    transfers = [dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, transfer_id=1, careunit="MICU", intime=AT,
                      outtime=AT + HOUR),
                 dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, transfer_id=2, careunit="SICU", intime=AT + MINUTE)]
    services = [dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, transfertime=AT, curr_service="MED"),
                dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, transfertime=AT + MINUTE, curr_service="SURG")]
    timeline = _timeline(TRANSFERS=transfers, SERVICES=services)

    visible_transfers = timeline.visible(Table.TRANSFERS)
    assert visible_transfers["transfer_id"].tolist() == [1]
    assert pd.isna(visible_transfers["outtime"].iloc[0])
    assert _rows(timeline, Table.SERVICES) == [0]


def test_patient_drops_dod_and_computes_age():
    """protects=date of death never exposed; age shifted to `at` year;
    fails_when=dod/anchor_year_group present or age uses anchor_age unshifted;
    no_claim=age accuracy beyond MIMIC's anchor scheme."""
    timeline = _timeline()
    patient = timeline.patient()
    assert "dod" not in patient and "anchor_year_group" not in patient
    assert timeline.age_at(patient) == 62  # 60 + (2180 - 2178)


def test_split_by_current_admittime():
    """protects=current/past partition at current admittime;
    fails_when=row at exactly admittime counted as past;
    no_claim=rows with null hadm_id (split ignores hadm_id by design)."""
    frame = pd.DataFrame({EVENT_TIME: [ADMITTIME - MINUTE, ADMITTIME, ADMITTIME + MINUTE]})
    current, past = _timeline().split(frame, ADMITTIME)
    assert current[EVENT_TIME].tolist() == [ADMITTIME, ADMITTIME + MINUTE]
    assert past[EVENT_TIME].tolist() == [ADMITTIME - MINUTE]


# ── current encounter selection ─────────────────────────────────────────────


def test_encounter_requires_icu_stay_at_t():
    """protects=D7 `at` must lie in [intime, outtime);
    fails_when=snapshot built outside ICU or at outtime;
    no_claim=error message wording."""
    icustays = [dict(subject_id=SUBJECT, hadm_id=CURRENT_HADM, stay_id=STAY, first_careunit="MICU",
                     intime=AT - DAY, outtime=AT)]
    with pytest.raises(NotInIcuStayError):
        _timeline(ICUSTAYS=icustays)


def test_encounter_rejects_overlapping_admissions():
    """protects=D2 single current admission;
    fails_when=one of two containing admissions silently chosen;
    no_claim=frequency of overlaps in MIMIC."""
    admissions = [_admission(CURRENT_HADM, AT - DAY, AT + DAY), _admission(201, AT - HOUR, AT + HOUR)]
    with pytest.raises(AmbiguousEncounterError):
        _timeline(ADMISSIONS=admissions)


def test_encounter_icu_without_admission_rejected():
    """protects=no snapshot without a current admission;
    fails_when=ICU stay used with undefined current admission;
    no_claim=MIMIC rows where ICU intime precedes admittime."""
    admissions = [_admission(CURRENT_HADM, AT + MINUTE, AT + DAY)]
    with pytest.raises(NotInIcuStayError):
        _timeline(ADMISSIONS=admissions)
