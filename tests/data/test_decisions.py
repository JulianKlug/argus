"""Unit tests: decision rules on visible frames produced by a stub Timeline (no Parquet)."""

import csv

import pandas as pd
import pytest

from argus.data.catalog import CATALOG_COLUMNS, CATALOG_PATH, NOT_DECISION, DecisionCatalog, Source, Status
from argus.data.decisions import SOURCE_TABLES, decisions
from argus.data.domains import (
    ACTION, COMMON_COLUMNS, DECISION_COLUMNS, EVENT_TIME, LABEL, ORDER_ROW, SOURCE_ROW, SOURCE_TABLE, SUBCATEGORY,
    DecisionAction, DecisionCategory, Table,
)
from test_availability import AT, CURRENT_HADM, HOUR, MINUTE, STAY, STAY_IDS, _timeline

WINDOW = 6 * HOUR
START = AT - WINDOW
PEEP, FIO2, TIDAL_SET, HEART_RATE = 220339, 223835, 224684, 220045
NOREPI, ORAL_WATER, INSULIN, DEXTROSE_5 = 221906, 220454, 223258, 220950
INVASIVE_VENT, INTUBATION, XRAY, ARTERIAL_LINE = 225792, 224385, 221216, 225752
PCA_RATE, RESTRAINTS = 224331, 224856
ARTERIAL_LINE_REMOVED, FOLEY, CHEST_TUBE_REMOVED = 225737, 229351, 227712
MAIN = "Main order parameter"
DRIP, BOLUS, ORAL = "01-Drips", "05-Med Bolus", "14-Oral/Gastric Intake"
FLUID, SC_INSULIN = "02-Fluids (Crystalloids)", "06-Insulin (Non IV)"

# Catalog rows: (source, key, category, subcategory, drug).
CATALOG_ROWS = [
    (Source.D_ITEMS, PEEP, "DEVICE_SETTING", "VENTILATOR", None),
    (Source.D_ITEMS, FIO2, "DEVICE_SETTING", "VENTILATOR", None),
    (Source.D_ITEMS, TIDAL_SET, "DEVICE_SETTING", "VENTILATOR", None),
    (Source.D_ITEMS, HEART_RATE, NOT_DECISION, None, None),
    (Source.D_ITEMS, NOREPI, "MEDICATION", None, "norepinephrine"),
    (Source.D_ITEMS, ORAL_WATER, "MEDICATION", None, "water"),
    (Source.D_ITEMS, INSULIN, "MEDICATION", None, "insulin"),
    (Source.D_ITEMS, DEXTROSE_5, "MEDICATION", None, "dextrose 5%"),
    (Source.D_ITEMS, INVASIVE_VENT, "DEVICE_SETTING", "VENTILATOR", None),
    (Source.D_ITEMS, INTUBATION, "PROCEDURE", "AIRWAY", None),
    (Source.D_ITEMS, ARTERIAL_LINE, "PROCEDURE", "LINE", None),
    (Source.D_ITEMS, PCA_RATE, "MEDICATION", "PCA", None),
    (Source.D_ITEMS, RESTRAINTS, "PROCEDURE", "RESTRAINT", None),
    (Source.D_ITEMS, ARTERIAL_LINE_REMOVED, "EQUIPMENT_REMOVAL", "LINE", None),
    (Source.D_ITEMS, FOLEY, "PROCEDURE", "CATHETER", None),
    (Source.D_ITEMS, CHEST_TUBE_REMOVED, "EQUIPMENT_REMOVAL", None, None),
    (Source.D_ITEMS, XRAY, NOT_DECISION, None, None),
    (Source.INPUT_ORDER_CATEGORY, DRIP, "MEDICATION", "DRIP", None),
    (Source.INPUT_ORDER_CATEGORY, BOLUS, "MEDICATION", "BOLUS", None),
    (Source.INPUT_ORDER_CATEGORY, ORAL, NOT_DECISION, None, None),
    (Source.INPUT_ORDER_CATEGORY, FLUID, "MEDICATION", "FLUID", None),
    (Source.INPUT_ORDER_CATEGORY, SC_INSULIN, "MEDICATION", "SUBCUTANEOUS", None),
    (Source.POE, "Radiology|CT Scan", "TEST_ORDER", "IMAGING", None),
    (Source.POE, "Respiratory|Mechanical Ventilation", "DEVICE_SETTING", "VENTILATOR", None),
    (Source.POE, "Medications|", "MEDICATION", None, None),
    (Source.POE, "General Care|Activity", NOT_DECISION, None, None),
    (Source.POE, "Lab|", "TEST_ORDER", "LAB", None),
    (Source.POE, "Consults|Nephrology", "CONSULT", None, None),
    (Source.POE, "ADT orders|Transfer", "PATIENT_MOVEMENT", None, None),
]


@pytest.fixture(scope="module")
def catalog(tmp_path_factory) -> DecisionCatalog:
    path = tmp_path_factory.mktemp("catalog") / "decision_catalog.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        for source, key, category, subcategory, drug in CATALOG_ROWS:
            writer.writerow(dict(source=source.value, key=key, label="", category=category,
                                 subcategory=subcategory or "", drug=drug or "", status=Status.DRAFT.name))
    return DecisionCatalog(path)


def test_shipped_catalog_has_valid_categories_and_statuses():
    """protects=decision_catalog.csv loads: known sources, categories, statuses, unique keys;
    fails_when=a hand edit introduces a typo (e.g. 'REVIEWD', 'DEVICE_SETING') or a duplicate key;
    no_claim=clinical correctness of the categories."""
    frame = pd.read_csv(CATALOG_PATH, dtype=str, keep_default_na=False)

    assert tuple(frame.columns) == CATALOG_COLUMNS
    assert set(frame["source"]) <= {s.value for s in Source}
    assert set(frame["category"]) <= {c.name for c in DecisionCategory} | {NOT_DECISION}
    assert set(frame["status"]) <= {s.name for s in Status}
    assert not frame.duplicated(["source", "key"]).any()
    DecisionCatalog()


def test_prescription_drugs_normalize_to_icu_drug_names(catalog):
    """protects=prescription drug strings map to the ICU items' drug names used for linking;
    fails_when=case, brand suffix or combination names break the match;
    no_claim=correct ingredient for every MIMIC drug string (real_data link rate)."""
    drugs = pd.Series([
        "NORepinephrine", "Sodium Chloride 0.9% / Norepinephrine", "Vancomycin", "0.9%", None,
        "Dextrose 5%", "Dextrose 50%",  # strength is part of the drug: D5 is not D50
    ])

    normalized = catalog.prescription_drugs(drugs)

    assert normalized.tolist() == [
        "norepinephrine", "norepinephrine", "vancomycin", None, None, "dextrose 5%", "dextrose",
    ]


def _decisions(catalog: DecisionCatalog, **tables: list[dict]) -> dict[DecisionCategory, pd.DataFrame]:
    timeline = _timeline(**tables)
    return decisions({t: timeline.visible(t) for t in SOURCE_TABLES}, catalog, START)


def _actions(frame: pd.DataFrame, table: Table) -> list[tuple[int, DecisionAction, pd.Timestamp]]:
    """(source_row, action, decision time) of `table`'s decisions, ordered by row then time."""
    rows = frame[frame[SOURCE_TABLE] == table.source_name]
    return sorted(zip(rows[SOURCE_ROW], rows[ACTION], rows[EVENT_TIME]), key=lambda a: (a[0], a[2]))


def _chart(itemid: int, value: str, time: pd.Timestamp, stay_id: int = STAY) -> dict:
    return dict(STAY_IDS | dict(stay_id=stay_id), itemid=itemid, value=value, charttime=time, storetime=time)


def _segment(start, end, rate, status="FinishedRunning", category=DRIP, component=MAIN, itemid=NOREPI) -> dict:
    return dict(STAY_IDS, itemid=itemid, starttime=start, endtime=end, storetime=start, rate=rate,
                amount=1.0, ordercategoryname=category, ordercomponenttypedescription=component,
                statusdescription=status)


def _rx(drug: str, start, route="IV", drug_type="MAIN", poe_id=None, stop=None) -> dict:
    return dict(subject_id=1, hadm_id=CURRENT_HADM, drug=drug, drug_type=drug_type, route=route,
                starttime=start, stoptime=stop, poe_id=poe_id)


def _poe(poe_id: str, order_type: str, subtype, transaction: str, time, discontinue_of=None) -> dict:
    return dict(poe_id=poe_id, poe_seq=1, subject_id=1, hadm_id=CURRENT_HADM, ordertime=time, order_type=order_type,
                order_subtype=subtype, transaction_type=transaction, discontinue_of_poe_id=discontinue_of)


# ── device settings ──────────────────────────────────────────────────────────


def test_settings_first_value_in_window_and_changes(catalog):
    """protects=spec criterion 2: kept setting = CONTINUE, changed = CHANGE, new = START, re-charting silent;
    fails_when=re-documented unchanged value emitted, prior value ignored, prior from another stay used,
    or measured items (heart rate) treated as settings;
    no_claim=which items are settings (catalog review)."""
    chart = [
        _chart(PEEP, "5", START - HOUR),  # 0 before window
        _chart(PEEP, "5", START + HOUR),  # 1 CONTINUE
        _chart(PEEP, "5", START + 2 * HOUR),  # 2 unchanged re-charting
        _chart(PEEP, "8", START + 3 * HOUR),  # 3 CHANGE
        _chart(FIO2, "40", START + HOUR),  # 4 START: no prior
        _chart(FIO2, "60", START - HOUR, stay_id=900),  # 5 earlier stay: not a prior for stay 1000
        _chart(PEEP, "10", START + HOUR, stay_id=900),  # 6 other stay, own START
        _chart(HEART_RATE, "90", START + HOUR),  # 7 measurement
        _chart(PEEP, "5", START),  # 8 exactly at start: excluded, still the prior value
        _chart(TIDAL_SET, "400", AT),  # 9 exactly at `at`: included
    ]

    result = _decisions(catalog, CHARTEVENTS=chart)[DecisionCategory.DEVICE_SETTING]

    assert _actions(result, Table.CHARTEVENTS) == [
        (1, DecisionAction.CONTINUE, START + HOUR),
        (3, DecisionAction.CHANGE, START + 3 * HOUR),
        (4, DecisionAction.START, START + HOUR),
        (6, DecisionAction.START, START + HOUR),
        (9, DecisionAction.START, AT),
    ]
    assert set(result[SUBCATEGORY]) == {"VENTILATOR"}


# ── infusions ────────────────────────────────────────────────────────────────


def test_infusion_start_titration_bag_change_stop_restart(catalog):
    """protects=spec criterion 3 infusion actions;
    fails_when=bag change (same rate, contiguous) emitted, diluent emitted, rate change missed,
    stop not at endtime, restart after a gap not a START, or drip/bolus of one drug merged;
    no_claim=MIMIC rate units or titration inside one segment."""
    inputs = [
        _segment(START + HOUR, START + 2 * HOUR, 0.05),  # 0 START
        _segment(START + 2 * HOUR, START + 3 * HOUR, 0.05),  # 1 bag change: none
        _segment(START + 3 * HOUR, START + 4 * HOUR, 0.10, status="Stopped"),  # 2 CHANGE + STOP at end
        _segment(START + 4 * HOUR + 30 * MINUTE, AT + HOUR, 0.10),  # 3 restart after 30 min: START
        _segment(START + HOUR, START + 2 * HOUR, None, component="Mixed solution"),  # 4 diluent
        _segment(START + HOUR + MINUTE, START + HOUR + 2 * MINUTE, None, category=BOLUS),  # 5 bolus: own START
        _segment(START + HOUR, START + HOUR + MINUTE, None, category=ORAL, itemid=ORAL_WATER),  # 6 intake
        _segment(START - 2 * HOUR, START - HOUR, 0.05, status="Stopped"),  # 7 before window
    ]

    result = _decisions(catalog, INPUTEVENTS=inputs)[DecisionCategory.MEDICATION]

    assert _actions(result, Table.INPUTEVENTS) == [
        (0, DecisionAction.START, START + HOUR),
        (2, DecisionAction.CHANGE, START + 3 * HOUR),
        (2, DecisionAction.STOP, START + 4 * HOUR),
        (3, DecisionAction.START, START + 4 * HOUR + 30 * MINUTE),
        (5, DecisionAction.START, START + HOUR + MINUTE),
    ]


def test_infusion_links_active_prescription_by_route(catalog):
    """protects=spec criterion 4 prescription link;
    fails_when=link ignores drug, active interval or route preference, or links a diluent (BASE) row;
    no_claim=real-data link rate (reported by real_data test)."""
    prescriptions = [
        _rx("NORepinephrine", START - HOUR, route="IV"),  # 0 PRN-style IV order
        _rx("NORepinephrine", START, route="IV DRIP"),  # 1 drip order
        _rx("NORepinephrine", START, drug_type="BASE"),  # 2 diluent: never linked
        _rx("Vancomycin", START, route="IV"),  # 3 other drug
        _rx("NORepinephrine", START - 3 * HOUR, stop=START - 2 * HOUR),  # 4 stopped before segments
    ]
    inputs = [
        _segment(START + HOUR, START + 2 * HOUR, 0.05),  # 0 drip -> rx 1
        _segment(START + HOUR, START + HOUR + MINUTE, None, category=BOLUS),  # 1 bolus -> rx 0
    ]

    result = _decisions(catalog, PRESCRIPTIONS=prescriptions, INPUTEVENTS=inputs)[DecisionCategory.MEDICATION]

    segments = result[result[SOURCE_TABLE] == Table.INPUTEVENTS.source_name].set_index(SOURCE_ROW)
    assert segments.loc[0, ORDER_ROW] == 1
    assert segments.loc[1, ORDER_ROW] == 0
    assert (segments["order_table"] == Table.PRESCRIPTIONS.source_name).all()


def test_infusion_without_active_prescription_stays_unlinked(catalog):
    """protects=no forced link when no prescription of the drug is active;
    fails_when=segment linked to a prescription starting > 1 h later or of another drug;
    no_claim=link precision on real data."""
    prescriptions = [_rx("NORepinephrine", START + 3 * HOUR), _rx("Vancomycin", START)]
    inputs = [_segment(START + HOUR, START + 2 * HOUR, 0.05)]

    result = _decisions(catalog, PRESCRIPTIONS=prescriptions, INPUTEVENTS=inputs)[DecisionCategory.MEDICATION]

    segment = result[result[SOURCE_TABLE] == Table.INPUTEVENTS.source_name].iloc[0]
    assert pd.isna(segment[ORDER_ROW]) and pd.isna(segment["order_table"])


def test_infusion_link_requires_compatible_route(catalog):
    """protects=link never crosses SC and IV: subcutaneous insulin is not the insulin drip's order;
    fails_when=SC insulin linked to an IV DRIP prescription, or an IV drip linked to an SC one
    (5/50 wrong links in the real-data sample, 2026-10-25);
    no_claim=insulin type (regular vs glargine) when the prescription says only 'Insulin'."""
    prescriptions = [
        _rx("Insulin Human Regular", START, route="IV DRIP"),  # 0 drip order
        _rx("Insulin", START, route="SC"),  # 1 sliding scale
    ]
    inputs = [
        _segment(START + HOUR, START + HOUR + MINUTE, None, category=SC_INSULIN, itemid=INSULIN),  # 0 -> rx 1
        _segment(START + HOUR, START + 2 * HOUR, 2.0, itemid=INSULIN),  # 1 drip -> rx 0
    ]
    only_drip_order = prescriptions[:1]
    only_sc_order = prescriptions[1:]

    def links(rx: list[dict]) -> dict:
        result = _decisions(catalog, PRESCRIPTIONS=rx, INPUTEVENTS=inputs)[DecisionCategory.MEDICATION]
        segments = result[result[SOURCE_TABLE] == Table.INPUTEVENTS.source_name]
        return dict(zip(segments[SOURCE_ROW], segments[ORDER_ROW]))

    assert links(prescriptions) == {0: 1, 1: 0}
    assert pd.isna(links(only_drip_order)[0])  # SC dose, only an IV order active
    assert pd.isna(links(only_sc_order)[1])  # IV drip, only an SC order active


def test_infusion_link_distinguishes_strength(catalog):
    """protects=dextrose 5% maintenance fluid is not linked to a dextrose 50% bolus order;
    fails_when=drug key drops the strength (real-data sample: D5W linked to D50);
    no_claim=strengths written differently on both sides (e.g. 'D5W')."""
    inputs = [_segment(START + HOUR, START + 2 * HOUR, 100.0, category=FLUID, itemid=DEXTROSE_5)]

    def link(drug: str):
        result = _decisions(catalog, PRESCRIPTIONS=[_rx(drug, START)], INPUTEVENTS=inputs)[DecisionCategory.MEDICATION]
        return result[result[SOURCE_TABLE] == Table.INPUTEVENTS.source_name].iloc[0][ORDER_ROW]

    assert pd.isna(link("Dextrose 50%"))
    assert link("Dextrose 5%") == 0


# ── prescriptions and poe ────────────────────────────────────────────────────


def test_prescription_decided_at_order_time(catalog):
    """protects=spec criterion 5 decision time = poe ordertime, else starttime; diluents skipped;
    fails_when=scheduled start inside window counted although ordered before it, BASE row emitted,
    or window bounds not (start, at];
    no_claim=verbal-order timing."""
    poe = [
        _poe("p-1", "Medications", None, "New", START - HOUR),
        _poe("p-2", "Medications", None, "New", START + HOUR),
    ]
    prescriptions = [
        _rx("Vancomycin", START + 2 * HOUR, poe_id="p-1"),  # 0 ordered before window
        _rx("Vancomycin", START + 2 * HOUR, poe_id="p-2"),  # 1 ORDER at START + 1h
        _rx("Vancomycin", START + 3 * HOUR),  # 2 no poe: ORDER at starttime
        _rx("NORepinephrine", START + 3 * HOUR, drug_type="BASE"),  # 3 diluent
        _rx("Vancomycin", START),  # 4 exactly at window start: excluded (spec criterion 8)
        _rx("Vancomycin", AT),  # 5 exactly at `at`: included
    ]

    result = _decisions(catalog, POE=poe, PRESCRIPTIONS=prescriptions)[DecisionCategory.MEDICATION]

    assert _actions(result, Table.PRESCRIPTIONS) == [
        (1, DecisionAction.ORDER, START + HOUR),
        (2, DecisionAction.ORDER, START + 3 * HOUR),
        (5, DecisionAction.ORDER, AT),
    ]


def test_charted_medications_and_procedures_follow_catalog(catalog):
    """protects=charted non-setting decisions (PCA rate, restraints) use the first-value + change rule
    under their own category;
    fails_when=charted medication ignored, filed as DEVICE_SETTING, or re-charting emitted;
    no_claim=which charted items are decisions (catalog review)."""
    chart = [
        _chart(PCA_RATE, "5", START - HOUR),  # 0 before window
        _chart(PCA_RATE, "5", START + HOUR),  # 1 CONTINUE
        _chart(PCA_RATE, "8", START + 2 * HOUR),  # 2 CHANGE
        _chart(RESTRAINTS, "Yes", START + HOUR),  # 3 START
        _chart(RESTRAINTS, "Yes", START + 2 * HOUR),  # 4 unchanged re-charting
    ]

    result = _decisions(catalog, CHARTEVENTS=chart)

    assert _actions(result[DecisionCategory.MEDICATION], Table.CHARTEVENTS) == [
        (1, DecisionAction.CONTINUE, START + HOUR),
        (2, DecisionAction.CHANGE, START + 2 * HOUR),
    ]
    assert result[DecisionCategory.MEDICATION][SUBCATEGORY].tolist() == ["PCA", "PCA"]
    assert _actions(result[DecisionCategory.PROCEDURE], Table.CHARTEVENTS) == [(3, DecisionAction.START, START + HOUR)]
    assert result[DecisionCategory.DEVICE_SETTING].empty


def test_charted_removals_stop_unless_unplanned(catalog):
    """protects=charted line discontinuation is an EQUIPMENT_REMOVAL STOP each time it is charted;
    accidental (unplanned) removals are not decisions;
    fails_when=removal filed as START/CONTINUE, a second removal of the same item type silenced
    by the first-value rule, or a patient-pulled line counted;
    no_claim=which line was removed when several of a type are in place."""
    chart = [
        _chart(ARTERIAL_LINE_REMOVED, "Planned", START - HOUR),  # 0 before window
        _chart(ARTERIAL_LINE_REMOVED, "Planned", START + HOUR),  # 1 STOP (same value as before)
        _chart(ARTERIAL_LINE_REMOVED, "Unplanned - Patient-initiated", START + 2 * HOUR),  # 2 accident
        _chart(ARTERIAL_LINE_REMOVED, "Infiltrated", START + 3 * HOUR),  # 3 STOP: removed for a complication
    ]

    result = _decisions(catalog, CHARTEVENTS=chart)[DecisionCategory.EQUIPMENT_REMOVAL]

    assert _actions(result, Table.CHARTEVENTS) == [
        (1, DecisionAction.STOP, START + HOUR),
        (3, DecisionAction.STOP, START + 3 * HOUR),
    ]


def test_poe_orders_follow_catalog_and_transaction(catalog):
    """protects=poe orders: catalog category, New/Change/D-C actions, medication orders stop-only;
    fails_when=medication New duplicated from poe, non-decision types emitted, D/C label not the drug;
    no_claim=completeness of the poe catalog (real_data coverage)."""
    poe = [
        _poe("p-1", "Medications", None, "New", START - HOUR),  # 0 duplicate of prescription
        _poe("p-2", "Radiology", "CT Scan", "New", START + HOUR),  # 1 TEST_ORDER ORDER
        _poe("p-3", "Respiratory", "Mechanical Ventilation", "Change", START + HOUR),  # 2 DEVICE CHANGE
        _poe("p-4", "Medications", None, "D/C", START + 2 * HOUR, discontinue_of="p-1"),  # 3 MEDICATION STOP
        _poe("p-5", "General Care", "Activity", "New", START + HOUR),  # 4 not a decision
        _poe("p-6", "Lab", None, "New", START + HOUR),  # 5 lab order: TEST_ORDER at order time
        _poe("p-7", "Medications", None, "New", START + HOUR),  # 6 in window, still a prescription duplicate
        _poe("p-8", "Consults", "Nephrology", "New", START + HOUR),  # 7 CONSULT
        _poe("p-9", "ADT orders", "Transfer", "New", START + 2 * HOUR),  # 8 PATIENT_MOVEMENT
    ]
    prescriptions = [_rx("Vancomycin", START - HOUR, poe_id="p-1")]

    result = _decisions(catalog, POE=poe, PRESCRIPTIONS=prescriptions)

    tests = result[DecisionCategory.TEST_ORDER]
    assert _actions(tests, Table.POE) == [(1, DecisionAction.ORDER, START + HOUR), (5, DecisionAction.ORDER, START + HOUR)]
    assert tests[SUBCATEGORY].tolist() == ["IMAGING", "LAB"]
    assert _actions(result[DecisionCategory.CONSULT], Table.POE) == [(7, DecisionAction.ORDER, START + HOUR)]
    assert result[DecisionCategory.CONSULT][LABEL].tolist() == ["Nephrology"]
    assert _actions(result[DecisionCategory.PATIENT_MOVEMENT], Table.POE) == [(8, DecisionAction.ORDER, START + 2 * HOUR)]
    assert _actions(result[DecisionCategory.DEVICE_SETTING], Table.POE) == [(2, DecisionAction.CHANGE, START + HOUR)]

    medication = result[DecisionCategory.MEDICATION]
    stops = medication[medication[SOURCE_TABLE] == Table.POE.source_name]
    assert _actions(stops, Table.POE) == [(3, DecisionAction.STOP, START + 2 * HOUR)]
    assert stops[LABEL].tolist() == ["Vancomycin"]


# ── procedures ───────────────────────────────────────────────────────────────


def test_procedure_intervals_start_and_stop(catalog):
    """protects=device intervals: START at starttime, STOP at visible endtime; procedures START only;
    catheter interval end and removal point events are EQUIPMENT_REMOVAL STOP;
    fails_when=running interval gets a STOP (endtime is unknown at `at`), imaging emitted, or intubation STOP;
    no_claim=clinical meaning of each procedure item (catalog review)."""
    procedures = [
        dict(STAY_IDS, itemid=INVASIVE_VENT, starttime=START + HOUR, endtime=START + 3 * HOUR, value=120.0),  # 0
        dict(STAY_IDS, itemid=INTUBATION, starttime=START + HOUR, endtime=START + HOUR + MINUTE, value=1.0),  # 1
        dict(STAY_IDS, itemid=INVASIVE_VENT, starttime=START + 4 * HOUR, endtime=AT + 24 * HOUR, value=1.0),  # 2
        dict(STAY_IDS, itemid=XRAY, starttime=START + HOUR, endtime=START + HOUR + MINUTE, value=1.0),  # 3
        dict(STAY_IDS, itemid=ARTERIAL_LINE, starttime=START - HOUR, endtime=START + 2 * HOUR, value=1.0),  # 4
        dict(STAY_IDS, itemid=FOLEY, starttime=START - HOUR, endtime=START + 3 * HOUR, value=1.0),  # 5
        dict(STAY_IDS, itemid=CHEST_TUBE_REMOVED, starttime=START + 2 * HOUR, endtime=START + 2 * HOUR + MINUTE,
             value=1.0),  # 6
    ]

    result = _decisions(catalog, PROCEDUREEVENTS=procedures)

    assert _actions(result[DecisionCategory.DEVICE_SETTING], Table.PROCEDUREEVENTS) == [
        (0, DecisionAction.START, START + HOUR),
        (0, DecisionAction.STOP, START + 3 * HOUR),
        (2, DecisionAction.START, START + 4 * HOUR),
    ]
    assert _actions(result[DecisionCategory.PROCEDURE], Table.PROCEDUREEVENTS) == [
        (1, DecisionAction.START, START + HOUR),  # line 4: inserted before the window; its removal is charted
    ]
    assert _actions(result[DecisionCategory.EQUIPMENT_REMOVAL], Table.PROCEDUREEVENTS) == [
        (5, DecisionAction.STOP, START + 3 * HOUR),  # Foley removed: no charted removal item exists
        (6, DecisionAction.STOP, START + 2 * HOUR),
    ]



# ── tests ordered ────────────────────────────────────────────────────────────


def test_specimens_one_order_without_results(catalog):
    """protects=spec criterion 6: one TEST_ORDER per specimen at draw time, tests listed, results absent;
    fails_when=one row per analyte, any result value (4.0, 'abnormal', organism) present,
    or pending specimen dropped;
    no_claim=order time (draw time is used; MIMIC has no lab order time)."""
    lab = dict(subject_id=1, hadm_id=CURRENT_HADM, value="4.0", valuenum=4.0, flag="abnormal")
    labs = [
        dict(lab, labevent_id=1, specimen_id=7, itemid=50971, charttime=START + HOUR, storetime=START + 2 * HOUR),
        dict(lab, labevent_id=2, specimen_id=7, itemid=50983, charttime=START + HOUR, storetime=START + 2 * HOUR),
        dict(lab, labevent_id=3, specimen_id=8, itemid=50813, charttime=AT - MINUTE, storetime=AT + HOUR),  # pending
    ]
    micro = [
        dict(subject_id=1, hadm_id=CURRENT_HADM, microevent_id=i, micro_specimen_id=9, spec_type_desc="BLOOD",
             test_name="Blood Culture", org_name="E. COLI", charttime=START + HOUR, chartdate=START.normalize(),
             storetime=START + 2 * HOUR, storedate=START.normalize())
        for i in (1, 2)
    ]
    timeline = _timeline(LABEVENTS=labs, MICROBIOLOGYEVENTS=micro)
    frames = {t: timeline.visible(t) for t in SOURCE_TABLES}
    frames[Table.LABEVENTS][LABEL] = ["Potassium", "Sodium", "Lactate"]

    result = decisions(frames, catalog, START)[DecisionCategory.TEST_ORDER]

    assert _actions(result, Table.LABEVENTS) == [
        (0, DecisionAction.ORDER, START + HOUR),
        (2, DecisionAction.ORDER, AT - MINUTE),
    ]
    assert _actions(result, Table.MICROBIOLOGYEVENTS) == [(0, DecisionAction.ORDER, START + HOUR)]
    labels = result.set_index([SOURCE_TABLE, SOURCE_ROW])[LABEL]
    assert labels[(Table.LABEVENTS.source_name, 0)] == "Potassium, Sodium"
    cells = set(result.astype(object).to_numpy().ravel().tolist())
    for result_value in ("4.0", 4.0, "abnormal", "E. COLI"):
        assert result_value not in cells, result_value


# ── shape ────────────────────────────────────────────────────────────────────


def test_frames_have_decision_schema_even_when_empty(catalog):
    """protects=every category returned, common + decision columns first, helper columns hidden;
    fails_when=missing category raises, column order differs, or '_'-prefixed columns leak;
    no_claim=dtypes of source columns."""
    result = _decisions(catalog)

    assert set(result) == set(DecisionCategory)
    for frame in result.values():
        assert frame.empty
        leading = list(COMMON_COLUMNS) + list(DECISION_COLUMNS)
        assert list(frame.columns[: len(leading)]) == leading
        assert not [c for c in frame.columns if c.startswith("_")]
