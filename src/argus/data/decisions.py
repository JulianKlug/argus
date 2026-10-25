"""Clinician decisions in a window (start, at], derived from rows visible at `at`.

Window logic only: rows arrive already filtered by availability.py, so nothing here
can leak. Each rule turns one source table into decision rows (action + decision time):

    visible rows ──► rule (category, action, event_time = decision time) ──► keep start < event_time

Internal helper columns start with "_" and never reach callers.
"""

from collections.abc import Mapping

import pandas as pd

from argus.data.catalog import CATEGORY, DRUG, POE_KEY_SEPARATOR, DecisionCatalog, Source
from argus.data.domains import (
    ACTION, COMMON_COLUMNS, DECISION_COLUMNS, EVENT_TIME, HADM_ID, LABEL, ORDER_ROW, ORDER_TABLE,
    SOURCE_ROW, SOURCE_TABLE, STAY_ID, SUBCATEGORY, DecisionAction, DecisionCategory, Table,
)

# Tables whose visible rows the rules read.
SOURCE_TABLES = (
    Table.CHARTEVENTS, Table.INPUTEVENTS, Table.PROCEDUREEVENTS, Table.PRESCRIPTIONS, Table.POE,
    Table.LABEVENTS, Table.MICROBIOLOGYEVENTS,
)

# Tables whose category comes from the catalog: they can yield any category.
_CATALOG_TABLES = (Table.CHARTEVENTS, Table.PROCEDUREEVENTS, Table.POE)

# Tables that can yield each category.
_CATEGORY_TABLES = {
    category: _CATALOG_TABLES for category in DecisionCategory
} | {
    DecisionCategory.MEDICATION: (Table.PRESCRIPTIONS, Table.INPUTEVENTS) + _CATALOG_TABLES,
    DecisionCategory.TEST_ORDER: (Table.LABEVENTS, Table.MICROBIOLOGYEVENTS) + _CATALOG_TABLES,
}

_CATEGORY = "_category"
_DRUG = "_drug"
_IN_WINDOW = "_in_window"

_STOP_STATUSES = frozenset({"Stopped", "Paused"})  # inputevents: clinician stopped or held
_MAIN_COMPONENT = "Main order parameter"  # inputevents drug row; diluents are "Mixed solution"
_BAG_CHANGE_GAP = pd.Timedelta(minutes=10)  # 90% of drip segments are contiguous, p90 gap 3 min
_POINT_EVENT = pd.Timedelta(minutes=1)  # procedureevents point events (intubation) last 1 min
_UNPLANNED = "Unplanned"  # charted removal reason prefix: accident, not a decision
_REMOVED_AT_END = frozenset({"CATHETER"})  # procedure subcategories whose interval end is the removal
_STOPPED_AT_END = frozenset({DecisionCategory.DEVICE_SETTING, DecisionCategory.MEDICATION})
_LINK_SLACK = pd.Timedelta(hours=1)  # infusion may start shortly before its prescription
_BASE_DRUG_TYPE = "BASE"  # diluent of a prescription, not a decision on its own
_PRESCRIBED_ORDER_TYPE = "Medications"  # poe New/Change duplicate prescriptions
_DRIP_SUBCATEGORY = "DRIP"
_DRIP_ROUTE = "IV DRIP"
_IV_ROUTE = "IV"
_IV_ROUTES = frozenset({_IV_ROUTE, _DRIP_ROUTE, "IV BOLUS"})
_SC_ROUTE_PREFIX = "SC"  # SC, SCPUMP
_NON_IV_ORDER_CATEGORY = "(Non IV)"  # e.g. '06-Insulin (Non IV)', '11-Prophylaxis (Non IV)'
_POE_ACTIONS = {
    "New": DecisionAction.ORDER,
    "Change": DecisionAction.CHANGE,
    "D/C": DecisionAction.STOP,
    "H": DecisionAction.STOP,  # hold
}
_LAB_COLUMNS = ("specimen_id", "priority")
_MICRO_COLUMNS = ("micro_specimen_id", "spec_itemid", "spec_type_desc")
_LABEL_SEPARATOR = ", "


def decisions(
    frames: Mapping[Table, pd.DataFrame], catalog: DecisionCatalog, start: pd.Timestamp
) -> dict[DecisionCategory, pd.DataFrame]:
    """Decision frames per category for rows with start < event_time."""
    prescriptions, poe = frames[Table.PRESCRIPTIONS], frames[Table.POE]
    candidates = {
        Table.CHARTEVENTS: _charted(frames[Table.CHARTEVENTS], catalog, start),
        Table.INPUTEVENTS: _infusions(frames[Table.INPUTEVENTS], prescriptions, catalog),
        Table.PROCEDUREEVENTS: _procedures(frames[Table.PROCEDUREEVENTS], catalog),
        Table.PRESCRIPTIONS: _prescriptions(prescriptions, poe),
        Table.POE: _orders(poe, prescriptions, catalog),
        Table.LABEVENTS: _specimens(frames[Table.LABEVENTS], "specimen_id", _LAB_COLUMNS),
        Table.MICROBIOLOGYEVENTS: _specimens(frames[Table.MICROBIOLOGYEVENTS], "micro_specimen_id", _MICRO_COLUMNS),
    }

    # Concatenate only the tables a category can come from: its schema stays fixed, other tables' columns out.
    result = {}
    for category, tables in _CATEGORY_TABLES.items():
        rows = pd.concat([candidates[t][candidates[t][_CATEGORY] == category] for t in tables], ignore_index=True)
        result[category] = _finish(rows[rows[EVENT_TIME] > start])
    return result


# ── rules ──────────────────────────────────────────────────────────────────


def _charted(chart: pd.DataFrame, catalog: DecisionCatalog, start: pd.Timestamp) -> pd.DataFrame:
    """Charted decisions (device settings, PCA/epidural rates...): first value in window
    (START / CONTINUE / CHANGE), then every change.

    e.g. PEEP 5 before window, 5 at 01:00, 5 at 02:00, 8 at 03:00 -> CONTINUE 01:00, CHANGE 03:00.
    Previous value = last one charted in the same ICU stay.
    Equipment removals: every in-window charting is a STOP, except unplanned (accidental) ones.
    """
    classes = catalog.classify(Source.D_ITEMS, chart["itemid"])
    rows = chart.assign(**{SUBCATEGORY: classes[SUBCATEGORY], _CATEGORY: classes[CATEGORY]})
    rows = rows[rows[_CATEGORY].notna()]
    rows = rows.sort_values([STAY_ID, "itemid", EVENT_TIME, SOURCE_ROW])
    rows[_IN_WINDOW] = rows[EVENT_TIME] > start

    group = rows.groupby([STAY_ID, "itemid"], dropna=False, sort=False)
    previous = group["value"].shift()
    first = rows[_IN_WINDOW] & ~group[_IN_WINDOW].shift(fill_value=False).astype(bool)
    changed = _differs(rows["value"], previous)
    no_prior = previous.isna()

    action = pd.Series(None, index=rows.index, dtype=object)
    action[rows[_IN_WINDOW] & ~first & changed] = DecisionAction.CHANGE
    action[first & ~no_prior & changed] = DecisionAction.CHANGE
    action[first & ~no_prior & ~changed] = DecisionAction.CONTINUE
    action[first & no_prior] = DecisionAction.START

    removal = rows[_CATEGORY] == DecisionCategory.EQUIPMENT_REMOVAL
    planned = ~rows["value"].astype("string").str.startswith(_UNPLANNED).fillna(False).astype(bool)
    action[removal] = None
    action[removal & rows[_IN_WINDOW] & planned] = DecisionAction.STOP

    rows[ACTION] = action
    return rows[action.notna()]


def _infusions(inputs: pd.DataFrame, prescriptions: pd.DataFrame, catalog: DecisionCatalog) -> pd.DataFrame:
    """ICU medications/fluids chained per (stay, item, order category).

    START after a gap, CHANGE on a new rate, STOP on stop/hold.
    e.g. norepinephrine 0.05 (10:00-12:00), 0.05 new bag (12:00-13:00), 0.10 (13:00-15:00, Stopped)
         -> START 10:00, CHANGE 13:00, STOP 15:00.
    Diluents ('Mixed solution' components) are part of their drip, not decisions.
    """
    item = catalog.classify(Source.D_ITEMS, inputs["itemid"])
    order = catalog.classify(Source.INPUT_ORDER_CATEGORY, inputs["ordercategoryname"])
    rows = inputs.assign(**{_CATEGORY: item[CATEGORY], SUBCATEGORY: order[SUBCATEGORY], _DRUG: item[DRUG]})
    is_medication = (item[CATEGORY] == DecisionCategory.MEDICATION) & (order[CATEGORY] == DecisionCategory.MEDICATION)
    rows = rows[is_medication & (rows["ordercomponenttypedescription"] == _MAIN_COMPONENT)]

    # Segment continues the previous one of the same item when it starts within the bag-change gap.
    # Previous endtime null = still running at `at` -> continuous.
    chain = [STAY_ID, "itemid", "ordercategoryname"]  # same drug as drip vs bolus = two chains
    rows = rows.sort_values(chain + ["starttime", SOURCE_ROW])
    group = rows.groupby(chain, dropna=False, sort=False)
    previous_end = group["endtime"].shift()
    gap = rows["starttime"] - previous_end
    continues = (group.cumcount() > 0) & (previous_end.isna() | (gap <= _BAG_CHANGE_GAP).fillna(False))
    rate_changed = continues & _differs(rows["rate"], group["rate"].shift())
    rows = _link_prescriptions(rows, prescriptions, catalog)

    starts = rows[~continues | rate_changed].copy()
    starts[ACTION] = continues[starts.index].map({False: DecisionAction.START, True: DecisionAction.CHANGE})

    # Stop time is known only once endtime <= at (availability nulls it otherwise).
    stops = rows[rows["statusdescription"].isin(_STOP_STATUSES) & rows["endtime"].notna()].copy()
    stops[ACTION] = DecisionAction.STOP
    stops[EVENT_TIME] = stops["endtime"]
    return pd.concat([starts, stops])


def _link_prescriptions(rows: pd.DataFrame, prescriptions: pd.DataFrame, catalog: DecisionCatalog) -> pd.DataFrame:
    """Attach the active same-drug prescription (no shared key in MIMIC).

    Candidates: same hadm_id and drug, rx.starttime - 1 h <= segment start <= rx.stoptime (null = open),
    compatible route ('(Non IV)' order category -> not IV, else -> not SC; e.g. SC insulin never links
    to the insulin drip order).
    Tie: route matching the segment (drip -> 'IV DRIP', else 'IV'), then latest rx.starttime.
    """
    rows = rows.copy()
    rows[ORDER_TABLE] = None
    rows[ORDER_ROW] = pd.Series(pd.NA, index=rows.index, dtype="Int64")

    rx = prescriptions[prescriptions["drug_type"] != _BASE_DRUG_TYPE]
    rx = pd.DataFrame({
        HADM_ID: rx[HADM_ID],
        _DRUG: catalog.prescription_drugs(rx["drug"]),
        "_rx_start": rx["starttime"],
        "_rx_stop": rx["stoptime"],
        "_rx_route": rx["route"],
        "_rx_row": rx[SOURCE_ROW],
    }).dropna(subset=[_DRUG])

    columns = [HADM_ID, _DRUG, "starttime", "ordercategoryname", SUBCATEGORY]
    segments = rows[columns].dropna(subset=[_DRUG]).reset_index(names="_segment")
    candidates = segments.merge(rx, on=[HADM_ID, _DRUG])
    active = (candidates["_rx_start"] - _LINK_SLACK <= candidates["starttime"]) & (
        candidates["_rx_stop"].isna() | (candidates["starttime"] <= candidates["_rx_stop"])
    )

    non_iv = candidates["ordercategoryname"].str.contains(_NON_IV_ORDER_CATEGORY, regex=False).fillna(False)
    route = candidates["_rx_route"].fillna("")
    compatible = (non_iv & ~route.isin(_IV_ROUTES)) | (~non_iv & ~route.str.startswith(_SC_ROUTE_PREFIX))
    candidates = candidates[(active & compatible).fillna(False).astype(bool)]
    if candidates.empty:
        return rows

    preferred = candidates[SUBCATEGORY].map(lambda s: _DRIP_ROUTE if s == _DRIP_SUBCATEGORY else _IV_ROUTE)
    candidates = candidates.assign(_route_match=candidates["_rx_route"] == preferred)
    best = candidates.sort_values(["_segment", "_route_match", "_rx_start"], ascending=[True, False, False])
    best = best.drop_duplicates("_segment").set_index("_segment")

    rows.loc[best.index, ORDER_TABLE] = Table.PRESCRIPTIONS.source_name
    rows.loc[best.index, ORDER_ROW] = best["_rx_row"].astype("Int64")
    return rows


def _procedures(procedures: pd.DataFrame, catalog: DecisionCatalog) -> pd.DataFrame:
    """procedureevents: START at starttime; STOP at visible endtime for device / inhaled-drug intervals.

    Point events (intubation: 1 min) only START. Removal items (chest tube removed) STOP at starttime.
    Catheter interval end (Foley) -> EQUIPMENT_REMOVAL STOP. Line interval ends are not emitted:
    the charted line discontinuation records the removal (with its reason).
    """
    classes = catalog.classify(Source.D_ITEMS, procedures["itemid"])
    rows = procedures.assign(**{_CATEGORY: classes[CATEGORY], SUBCATEGORY: classes[SUBCATEGORY]})
    rows = rows[rows[_CATEGORY].notna()]

    removal = rows[_CATEGORY] == DecisionCategory.EQUIPMENT_REMOVAL
    starts = rows.assign(**{ACTION: removal.map({True: DecisionAction.STOP, False: DecisionAction.START})})

    is_interval = ((rows["endtime"] - rows["starttime"]) > _POINT_EVENT).fillna(False).astype(bool)
    removed_at_end = rows[SUBCATEGORY].isin(_REMOVED_AT_END)
    ends = rows[is_interval & (rows[_CATEGORY].isin(_STOPPED_AT_END) | removed_at_end)].copy()
    ends[ACTION] = DecisionAction.STOP
    ends[EVENT_TIME] = ends["endtime"]
    ends.loc[removed_at_end[ends.index], _CATEGORY] = DecisionCategory.EQUIPMENT_REMOVAL
    return pd.concat([starts, ends])


def _prescriptions(prescriptions: pd.DataFrame, poe: pd.DataFrame) -> pd.DataFrame:
    """Ward/ICU prescriptions: ORDER at poe ordertime (decision), else starttime. Diluents skipped."""
    rows = prescriptions[prescriptions["drug_type"] != _BASE_DRUG_TYPE].copy()
    ordertime = _lookup(rows["poe_id"], poe.drop_duplicates("poe_id").set_index("poe_id")["ordertime"])
    rows[EVENT_TIME] = ordertime.astype(rows[EVENT_TIME].dtype).fillna(rows[EVENT_TIME])
    rows[_CATEGORY] = DecisionCategory.MEDICATION
    rows[ACTION] = DecisionAction.ORDER
    return rows


def _orders(poe: pd.DataFrame, prescriptions: pd.DataFrame, catalog: DecisionCatalog) -> pd.DataFrame:
    """poe orders per catalog (imaging, ECG, ventilation, blood products, ...).

    Medication orders: only stops/holds (prescriptions carry the orders); label = stopped drug.
    """
    keys = poe["order_type"] + POE_KEY_SEPARATOR + poe["order_subtype"].fillna("")
    classes = catalog.classify(Source.POE, keys)
    rows = poe.assign(**{
        _CATEGORY: classes[CATEGORY],
        SUBCATEGORY: classes[SUBCATEGORY],
        ACTION: poe["transaction_type"].map(_POE_ACTIONS).astype(object),
        LABEL: poe["order_subtype"].fillna(poe[LABEL]),
    })
    rows = rows[rows[_CATEGORY].notna() & rows[ACTION].notna()]

    prescribed = rows["order_type"] == _PRESCRIBED_ORDER_TYPE
    rows = rows[~prescribed | (rows[ACTION] == DecisionAction.STOP)].copy()
    prescribed = rows["order_type"] == _PRESCRIBED_ORDER_TYPE

    drugs = prescriptions.dropna(subset=["poe_id"]).drop_duplicates("poe_id").set_index("poe_id")["drug"]
    stopped_drug = _lookup(rows["discontinue_of_poe_id"], drugs)
    rows.loc[prescribed, LABEL] = stopped_drug[prescribed].fillna(rows.loc[prescribed, LABEL])
    return rows


def _specimens(tests: pd.DataFrame, specimen: str, kept: tuple[str, ...]) -> pd.DataFrame:
    """One TEST_ORDER per specimen at draw time; label = tests; result columns dropped.

    e.g. 3 labevents rows of specimen 7 (Na, K, Cl) -> one row labelled 'Chloride, Potassium, Sodium'.
    """
    rows = tests.sort_values([EVENT_TIME, SOURCE_ROW])
    labels = rows.groupby(specimen)[LABEL].agg(lambda s: _LABEL_SEPARATOR.join(sorted(set(s.dropna()))))

    first = rows.drop_duplicates(specimen)[list(COMMON_COLUMNS) + list(kept)].copy()
    first[LABEL] = first[specimen].map(labels)
    first[_CATEGORY] = DecisionCategory.TEST_ORDER
    first[ACTION] = DecisionAction.ORDER
    return first


# ── mechanics ──────────────────────────────────────────────────────────────


def _lookup(keys: pd.Series, table: pd.Series) -> pd.Series:
    """`keys.map(table)` that keeps table's dtype; pandas 3 map fails on an empty datetime table."""
    return pd.Series(table.reindex(keys.to_numpy()).to_numpy(), index=keys.index, dtype=table.dtype)


def _differs(values: pd.Series, previous: pd.Series) -> pd.Series:
    """Null-safe `!=`: missing vs value differs, missing vs missing does not."""
    both_missing = values.isna() & previous.isna()
    return (values != previous).fillna(True).astype(bool) & ~both_missing


def _finish(rows: pd.DataFrame) -> pd.DataFrame:
    """Common + decision columns first, helper columns dropped, ordered by decision time."""
    rows = rows.copy()
    for column in (SUBCATEGORY, ORDER_TABLE):
        if column not in rows:
            rows[column] = None
    if ORDER_ROW not in rows:
        rows[ORDER_ROW] = pd.NA
    rows[ORDER_ROW] = rows[ORDER_ROW].astype("Int64")

    leading = list(COMMON_COLUMNS) + list(DECISION_COLUMNS)
    rest = [c for c in rows.columns if c not in leading and not c.startswith("_")]
    rows = rows.sort_values([EVENT_TIME, SOURCE_TABLE, SOURCE_ROW])
    return rows[leading + rest].reset_index(drop=True)
