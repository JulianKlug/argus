"""Leak policy: the single module that compares record times with `at` (spec rule 6).

Per snapshot, a `Timeline` fetches one subject's raw rows from the store and,
per table, decides which rows clinicians could see at `at` and which fields
must be withheld:

    raw rows ──► rule (event_time, available_time, redactions) ──► keep available_time <= at

Times: `ct` clinical time, `st` storetime, `EOD(d)` = d + 24h for date-only columns.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from argus.data.domains import (
    AVAILABLE_TIME, COMMON_COLUMNS, EVENT_TIME, HADM_ID, IS_PENDING, IS_TRUNCATED,
    SOURCE_ROW, SOURCE_TABLE, STAY_ID, Table,
)
from argus.data.errors import AmbiguousEncounterError, NotInIcuStayError, SubjectNotFoundError
from argus.data.store import ROW_COLUMN, MimicStore

_DAY = pd.Timedelta(hours=24)

_LAB_RESULT_COLUMNS = ("value", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "comments")
_MICRO_RESULT_COLUMNS = (
    "org_itemid", "org_name", "isolate_num", "quantity", "ab_itemid", "ab_name",
    "dilution_text", "dilution_comparison", "dilution_value", "interpretation", "comments",
)
_MICRO_PENDING_KEY = ["micro_specimen_id", "test_itemid"]
_ADMISSION_OUTCOME_COLUMNS = ("dischtime", "deathtime", "discharge_location", "hospital_expire_flag")
_ICUSTAY_END_COLUMNS = ("outtime", "los", "last_careunit")
_INTERVAL_END_COLUMNS = ("endtime", "statusdescription")
_POE_HIDDEN_COLUMNS = ["order_status"]

# Detail table -> (parent table, join key). Detail rows inherit the parent's visibility.
_DETAIL_PARENTS = {
    Table.EMAR_DETAIL: (Table.EMAR, "emar_id"),
    Table.POE_DETAIL: (Table.POE, "poe_id"),
    Table.RADIOLOGY_DETAIL: (Table.RADIOLOGY, "note_id"),
    Table.DISCHARGE_DETAIL: (Table.DISCHARGE, "note_id"),
}


@dataclass(frozen=True)
class Encounter:
    """Current admission and ICU stay rows, already redacted at `at`."""

    admission: pd.Series
    icustay: pd.Series


def _latest(clinical: pd.Series, stored: pd.Series) -> pd.Series:
    """D1: max(ct, st); st null -> ct."""
    stored = stored.fillna(clinical)
    return clinical.where(clinical >= stored, stored)


def _eod(date: pd.Series) -> pd.Series:
    """Date-only column becomes available at the end of that day."""
    return date + _DAY


def _prorate(quantity: pd.Series, start: pd.Series, end: pd.Series, at: pd.Timestamp) -> pd.Series:
    """Linear share of `quantity` delivered by `at`, e.g. 600 over 10:00-16:00 at 12:00 -> 200."""
    fraction = (at - start) / (end - start)
    return quantity * fraction.astype("Float64")


def _withhold(rows: pd.DataFrame, mask: pd.Series, columns) -> pd.Series:
    """Null `columns` where `mask`; return rows where a known value was actually hidden."""
    mask = mask.fillna(False).astype(bool)
    columns = list(columns)
    hidden = mask & rows[columns].notna().any(axis=1)
    rows.loc[mask, columns] = None
    return hidden


def _lookup(keys: pd.Series, table: pd.Series) -> pd.Series:
    """`keys.map(table)` that keeps table's dtype; pandas 3 map fails on an empty datetime table."""
    return pd.Series(table.reindex(keys.to_numpy()).to_numpy(), index=keys.index, dtype=table.dtype)


def _is_after(times: pd.Series, at: pd.Timestamp) -> pd.Series:
    """True where time > at; null -> False."""
    return (times > at).fillna(False).astype(bool)


class Timeline:
    """One subject's records as visible at one instant `at`."""

    def __init__(self, store: MimicStore, subject_id: int, at: pd.Timestamp) -> None:
        self._store = store
        self._subject_id = subject_id
        self._at = at
        self._raw_cache: dict[Table, pd.DataFrame] = {}
        self._visible_cache: dict[Table, pd.DataFrame] = {}
        self._current_hadm_id: int | None = None

        self._rules: dict[Table, Callable[[Table, pd.DataFrame], pd.DataFrame]] = {
            Table.CHARTEVENTS: self._charted,
            Table.DATETIMEEVENTS: self._charted,
            Table.OUTPUTEVENTS: self._charted,
            Table.EMAR: self._charted,
            Table.RADIOLOGY: self._charted,
            Table.DISCHARGE: self._discharge_note,
            Table.LABEVENTS: self._lab,
            Table.MICROBIOLOGYEVENTS: self._micro,
            Table.PRESCRIPTIONS: self._prescription,
            Table.PHARMACY: self._pharmacy,
            Table.POE: self._order,
            Table.INPUTEVENTS: self._input,
            Table.INGREDIENTEVENTS: self._input,
            Table.PROCEDUREEVENTS: self._procedure_event,
            Table.OMR: self._outpatient,
            Table.DIAGNOSES_ICD: self._coded_at_discharge,
            Table.DRGCODES: self._coded_at_discharge,
            Table.PROCEDURES_ICD: self._coded_procedure,
            Table.HCPCSEVENTS: self._coded_procedure,
            Table.ADMISSIONS: self._admission,
            Table.ICUSTAYS: self._icustay,
            Table.TRANSFERS: self._transfer,
            Table.SERVICES: self._service,
        }

    # ── public API ────────────────────────────────────────────────────────

    def patient(self) -> pd.Series:
        """Patient row without `dod` / `anchor_year_group`; raises if subject unknown."""
        patients = self._raw(Table.PATIENTS)
        if patients.empty:
            raise SubjectNotFoundError(f"subject_id {self._subject_id} not in store")

        return patients.iloc[0].drop(["dod", "anchor_year_group"])

    def age_at(self, patient: pd.Series) -> int:
        """anchor_age + (at.year - anchor_year)."""
        return int(patient["anchor_age"] + (self._at.year - patient["anchor_year"]))

    def current_encounter(self) -> Encounter:
        """D7 ICU check, then D2 admission containing `at`."""
        icustays = self._raw(Table.ICUSTAYS)
        in_icu = icustays[(icustays["intime"] <= self._at) & (self._at < icustays["outtime"])]
        if in_icu.empty:
            raise NotInIcuStayError(f"{self._at} outside every ICU stay of subject {self._subject_id}")

        admissions = self._raw(Table.ADMISSIONS)
        containing = admissions[(admissions["admittime"] <= self._at) & (self._at < admissions["dischtime"])]
        if len(containing) > 1:
            raise AmbiguousEncounterError(f"{len(containing)} admissions contain {self._at}")
        if containing.empty:
            raise NotInIcuStayError(f"ICU stay contains {self._at} but no admission does")

        self._current_hadm_id = int(containing.iloc[0][HADM_ID])

        # Prefer the stay belonging to the current admission.
        same_admission = in_icu[in_icu[HADM_ID] == self._current_hadm_id]
        stay_id = (same_admission if not same_admission.empty else in_icu).iloc[0][STAY_ID]

        admission = self.visible(Table.ADMISSIONS)
        icustay = self.visible(Table.ICUSTAYS)
        return Encounter(
            admission=admission[admission[HADM_ID] == self._current_hadm_id].iloc[0],
            icustay=icustay[icustay[STAY_ID] == stay_id].iloc[0],
        )

    def visible(self, table: Table) -> pd.DataFrame:
        """Rows of `table` available at `at`, redacted, with common columns first.

        Requires current_encounter() first: some rules depend on the current admission.
        """
        if table in self._visible_cache:
            return self._visible_cache[table]

        rows = self._raw(table).copy()
        result = self._inherit(table, rows) if table in _DETAIL_PARENTS else self._rules[table](table, rows)
        self._visible_cache[table] = result
        return result

    def split(self, frame: pd.DataFrame, admittime: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(current, past): event_time >= admittime vs < admittime."""
        is_current = frame[EVENT_TIME] >= admittime
        return frame[is_current].reset_index(drop=True), frame[~is_current].reset_index(drop=True)

    # ── shared mechanics ──────────────────────────────────────────────────

    def _raw(self, table: Table) -> pd.DataFrame:
        if table not in self._raw_cache:
            self._raw_cache[table] = self._store.rows(table, self._subject_id)
        return self._raw_cache[table]

    def _finish(
        self,
        table: Table,
        rows: pd.DataFrame,
        event_time: pd.Series,
        available_time: pd.Series,
        is_pending: pd.Series | None = None,
        is_truncated: pd.Series | None = None,
        extra_keep: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Attach common columns, keep rows with event_time and available_time <= at."""
        false = pd.Series(False, index=rows.index)
        rows[SOURCE_TABLE] = table.source_name
        rows[SOURCE_ROW] = rows[ROW_COLUMN].astype("int64")
        rows[EVENT_TIME] = event_time
        rows[AVAILABLE_TIME] = available_time
        rows[IS_PENDING] = false if is_pending is None else is_pending.astype(bool)
        rows[IS_TRUNCATED] = false if is_truncated is None else is_truncated.astype(bool)
        for column in (HADM_ID, STAY_ID):
            if column not in rows:
                rows[column] = pd.Series(pd.NA, index=rows.index, dtype="Int64")

        keep = (rows[AVAILABLE_TIME] <= self._at) & (rows[EVENT_TIME] <= self._at)
        if extra_keep is not None:
            keep &= extra_keep
        keep = keep.fillna(False).astype(bool)

        ordered = list(COMMON_COLUMNS) + [c for c in rows.columns if c not in COMMON_COLUMNS and c != ROW_COLUMN]
        return rows.loc[keep, ordered].reset_index(drop=True)

    def _inherit(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """Detail rows visible iff their parent row is; times and ids from the parent."""
        parent_table, key = _DETAIL_PARENTS[table]
        parent = self.visible(parent_table)
        parent = parent[[key, HADM_ID, STAY_ID, EVENT_TIME, AVAILABLE_TIME]].drop_duplicates(key)

        rows = rows.drop(columns=[c for c in (HADM_ID, STAY_ID) if c in rows]).merge(parent, on=key, how="inner")
        return self._finish(table, rows, rows.pop(EVENT_TIME), rows.pop(AVAILABLE_TIME))

    # ── rules, one per row of the spec availability table ────────────────

    def _charted(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """chartevents, datetimeevents, outputevents, emar, radiology: max(charttime, st) <= at."""
        available = _latest(rows["charttime"], rows["storetime"])
        return self._finish(table, rows, rows["charttime"], available)

    def _discharge_note(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """Prior admissions only (D3); null hadm_id excluded (cannot prove not current)."""
        available = _latest(rows["charttime"], rows["storetime"])
        not_current = (rows[HADM_ID] != self._current_hadm_id).fillna(False)
        return self._finish(table, rows, rows["charttime"], available, extra_keep=not_current)

    def _lab(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """D12: drawn -> visible; result fields withheld until stored."""
        drawn = rows["charttime"]
        resulted = _latest(drawn, rows["storetime"])
        pending = _is_after(resulted, self._at)
        _withhold(rows, pending, _LAB_RESULT_COLUMNS)
        available = resulted.where(~pending, drawn)
        return self._finish(table, rows, drawn, available, is_pending=pending)

    def _micro(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """Collected -> visible; results withheld until stored; one pending row per specimen test."""
        collected = rows["charttime"].fillna(_eod(rows["chartdate"]))
        stored = rows["storetime"].fillna(_eod(rows["storedate"]))
        resulted = _latest(collected, stored)
        pending = stored.isna() | _is_after(resulted, self._at)
        _withhold(rows, pending, _MICRO_RESULT_COLUMNS)
        available = resulted.where(~pending, collected)

        visible = self._finish(table, rows, collected, available, is_pending=pending)
        duplicate = visible[IS_PENDING] & visible.duplicated(_MICRO_PENDING_KEY + [IS_PENDING])
        return visible[~duplicate].reset_index(drop=True)

    def _prescription(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """max(starttime, poe.ordertime) <= at; future stoptime withheld."""
        ordertime = _lookup(rows["poe_id"], self._poe_ordertimes())
        available = _latest(rows["starttime"], ordertime.astype(rows["starttime"].dtype))
        truncated = _withhold(rows, _is_after(rows["stoptime"], self._at), ["stoptime"])
        return self._finish(table, rows, rows["starttime"], available, is_truncated=truncated)

    def _pharmacy(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """coalesce(entertime, starttime) <= at; future times and unsettled status withheld."""
        entered = rows["entertime"].fillna(rows["starttime"])
        stopped = (rows["stoptime"] <= self._at).fillna(False).astype(bool)

        truncated = _withhold(rows, ~stopped, ["status"])
        for column in ("stoptime", "expirationdate", "verifiedtime"):
            truncated |= _withhold(rows, _is_after(rows[column], self._at), [column])
        return self._finish(table, rows, entered, entered, is_truncated=truncated)

    def _order(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """ordertime <= at; status dropped; future discontinuation withheld."""
        rows = rows.drop(columns=_POE_HIDDEN_COLUMNS)
        discontinued_at = _lookup(rows["discontinued_by_poe_id"], self._poe_ordertimes())
        not_yet = rows["discontinued_by_poe_id"].notna() & ~(discontinued_at <= self._at).fillna(False).astype(bool)
        truncated = _withhold(rows, not_yet, ["discontinued_by_poe_id"])
        return self._finish(table, rows, rows["ordertime"], rows["ordertime"], is_truncated=truncated)

    def _input(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """inputevents, ingredientevents: D5 interval clipped at `at`, amount prorated."""
        return self._interval(table, rows, "amount")

    def _procedure_event(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """procedureevents: D5 interval clipped at `at`, value prorated."""
        return self._interval(table, rows, "value")

    def _interval(self, table: Table, rows: pd.DataFrame, quantity: str) -> pd.DataFrame:
        running = _is_after(rows["endtime"], self._at)
        rows.loc[running, quantity] = _prorate(
            rows.loc[running, quantity], rows.loc[running, "starttime"], rows.loc[running, "endtime"], self._at
        )
        _withhold(rows, running, _INTERVAL_END_COLUMNS)
        return self._finish(table, rows, rows["starttime"], rows["starttime"], is_truncated=running)

    def _outpatient(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """omr: EOD(chartdate) <= at."""
        return self._finish(table, rows, rows["chartdate"], _eod(rows["chartdate"]))

    def _coded_at_discharge(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """diagnoses_icd, drgcodes: coded at owning admission's dischtime."""
        dischtime = self._dischtime_of(rows)
        return self._finish(table, rows, dischtime, dischtime, extra_keep=self._not_current(rows))

    def _coded_procedure(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """procedures_icd, hcpcsevents: event at chartdate, coded at dischtime."""
        dischtime = self._dischtime_of(rows)
        event_time = rows["chartdate"].fillna(dischtime)
        return self._finish(table, rows, event_time, dischtime, extra_keep=self._not_current(rows))

    def _admission(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        """admittime <= at; outcome of the current (or any unfinished) admission withheld."""
        unfinished = _is_after(rows["dischtime"], self._at) | (rows[HADM_ID] == self._current_hadm_id).fillna(False)
        truncated = _withhold(rows, unfinished, _ADMISSION_OUTCOME_COLUMNS)
        truncated |= _withhold(rows, _is_after(rows["edouttime"], self._at), ["edouttime"])
        return self._finish(table, rows, rows["admittime"], rows["admittime"], is_truncated=truncated)

    def _icustay(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        truncated = _withhold(rows, _is_after(rows["outtime"], self._at), _ICUSTAY_END_COLUMNS)
        return self._finish(table, rows, rows["intime"], rows["intime"], is_truncated=truncated)

    def _transfer(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        truncated = _withhold(rows, _is_after(rows["outtime"], self._at), ["outtime"])
        return self._finish(table, rows, rows["intime"], rows["intime"], is_truncated=truncated)

    def _service(self, table: Table, rows: pd.DataFrame) -> pd.DataFrame:
        return self._finish(table, rows, rows["transfertime"], rows["transfertime"])

    # ── lookups ───────────────────────────────────────────────────────────

    def _poe_ordertimes(self) -> pd.Series:
        """poe_id -> ordertime over all the subject's orders."""
        poe = self._raw(Table.POE)
        return poe.drop_duplicates("poe_id").set_index("poe_id")["ordertime"]

    def _dischtime_of(self, rows: pd.DataFrame) -> pd.Series:
        """Owning admission's dischtime per row; unknown admission -> NaT (never visible)."""
        admissions = self._raw(Table.ADMISSIONS).drop_duplicates(HADM_ID).set_index(HADM_ID)["dischtime"]
        return rows[HADM_ID].map(admissions).astype(admissions.dtype)

    def _not_current(self, rows: pd.DataFrame) -> pd.Series:
        return (rows[HADM_ID] != self._current_hadm_id).fillna(False).astype(bool)
