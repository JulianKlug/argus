"""Record, decision and note lines. Fixed per-table column whitelists; item identifiers never rendered.

    record    [E3HP0RT2C] T-02:15 Potassium: 5.9 mEq/L, abnormal, ref 3.3-5.1
    decision  [D7Q2KX1MA] T-04:50 MEDICATION START Norepinephrine (DRIP): 0.08 mcg/kg/min; order [E3HP0RT2C]
    pending   [E0JX8W3NR] T-01:05 Lactate: PENDING
    ongoing   [E2CC7YH4D] since T-04:50 Norepinephrine: 0.08 mcg/kg/min, ONGOING AT WINDOW END

Every string goes through escape.py; every time through timefmt.py.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from argus.data.domains import ACTION, AVAILABLE_TIME, EVENT_TIME, LABEL, SUBCATEGORY
from argus.rendering import escape, timefmt

PENDING = "PENDING"
ONGOING = "ONGOING AT WINDOW END"
NOT_RECORDED = "NOT RECORDED"
NO_RECORDS = "NO RECORDS AVAILABLE"
SHORTENED = "SHORTENED"
LATE = "(late)"
DOCUMENTED = "documented"
FIELD_SEPARATOR = ", "
SIGNIFICANT_DIGITS = 6

# A string that is only a timestamp (e.g. datetimeevents.value) is a shifted date: render it relative.
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?")

# Note section heading at line start, e.g. 'IMPRESSION:' or 'Brief Hospital Course:'.
_NOTE_HEADING = re.compile(r"^\s*([A-Za-z][A-Za-z /&()'-]{1,60}?)\s*:")
_MIN_CAPS_HEADING = 3  # 'IMPRESSION:' counts, 'PA:' does not


@dataclass(frozen=True)
class Field:
    """Columns rendered together, e.g. ('value', 'valueuom') -> '118 bpm'. Skipped when the first is missing."""

    columns: tuple[str, ...]
    separator: str = " "
    prefix: str = ""


def _f(*columns: str, separator: str = " ", prefix: str = "") -> Field:
    return Field(columns, separator, prefix)


_VALUE = _f("value", "valueuom")
_AMOUNT = _f("amount", "amountuom")
_RATE = _f("rate", "rateuom")
_UNTIL = _f("endtime", prefix="until ")

# Source table -> rendered columns after the label (record lines).
RECORD_FIELDS: dict[str, tuple[Field, ...]] = {
    "chartevents": (_VALUE,),
    "datetimeevents": (_f("value"),),
    "labevents": (_VALUE, _f("flag"), _f("ref_range_lower", "ref_range_upper", separator="-", prefix="ref "), _f("comments")),
    "microbiologyevents": (
        _f("spec_type_desc"), _f("org_name"), _f("quantity"), _f("ab_name", "interpretation"),
        _f("dilution_text"), _f("comments"),
    ),
    "poe": (_f("order_subtype"), _f("transaction_type")),
    "poe_detail": (_f("field_value"),),
    "prescriptions": (
        _f("dose_val_rx", "dose_unit_rx"), _f("route"), _f("doses_per_24_hrs", prefix="doses/24h "),
        _f("stoptime", prefix="until "),
    ),
    "pharmacy": (_f("route"), _f("frequency"), _f("infusion_type"), _f("status"), _f("stoptime", prefix="until ")),
    "emar": (_f("event_txt"),),
    "emar_detail": (_f("dose_given", "dose_given_unit"), _f("infusion_rate", "infusion_rate_unit"), _f("route")),
    "inputevents": (_AMOUNT, _RATE, _f("statusdescription"), _UNTIL),
    "ingredientevents": (_AMOUNT, _RATE),
    "outputevents": (_VALUE,),
    "procedureevents": (_VALUE, _f("location"), _f("statusdescription"), _UNTIL),
    "procedures_icd": (),
    "hcpcsevents": (),
    "diagnoses_icd": (),
    "drgcodes": (_f("drg_type"),),
    "omr": (_f("result_value"),),
    "admissions": (_f("admission_location"), _f("discharge_location", prefix="discharged to ")),
    "icustays": (),
    "transfers": (_f("eventtype"), _f("outtime", prefix="until ")),
    "services": (_f("prev_service", prefix="from "),),
    "radiology_detail": (_f("field_value"),),
    "discharge_detail": (_f("field_value"),),
}

# Source table -> dose, rate, amount, route or setting on decision lines.
DECISION_FIELDS: dict[str, tuple[Field, ...]] = {
    "chartevents": (_VALUE,),
    "inputevents": (_RATE, _AMOUNT),
    "procedureevents": (_f("location"),),
    "prescriptions": (_f("dose_val_rx", "dose_unit_rx"), _f("route")),
    "poe": (_f("order_type"),),
    "labevents": (_f("priority"),),
    "microbiologyevents": (_f("spec_type_desc"),),
}

# Pending tests: identity only, never a value, organism or susceptibility.
PENDING_FIELDS: dict[str, tuple[Field, ...]] = {
    "labevents": (),
    "microbiologyevents": (_f("spec_type_desc"),),
}

# Ongoing intervals: D1's clipped state; end time and status are already withheld.
ONGOING_FIELDS: dict[str, tuple[Field, ...]] = {
    "inputevents": (_AMOUNT, _RATE),
    "emar": (_f("event_txt"),),
    "emar_detail": (_f("dose_given", "dose_given_unit"), _f("infusion_rate", "infusion_rate_unit"), _f("route")),
    "procedureevents": (_VALUE, _f("location")),
}

NOTE_TITLES = {"radiology": "Radiology report", "discharge": "Discharge summary"}


@dataclass(frozen=True)
class Lines:
    """One rendered unit: a head line plus escaped block lines (multi-line free text)."""

    head: str
    block: tuple[str, ...] = ()

    def all(self) -> list[str]:
        return [self.head, *self.block]


# ── scalar values ─────────────────────────────────────────────────────────


def present(value) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def number(value: float) -> str:
    """Six significant digits, no exponent, e.g. 0.08 -> '0.08', 199.99999 -> '200', 7.0 -> '7'."""
    return np.format_float_positional(float(value), precision=SIGNIFICANT_DIGITS, unique=True, fractional=False, trim="-")


def scalar(value, at: datetime) -> tuple[str, list[str]]:
    """(inline, block) text of one cell; times relative to `at`."""
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return timefmt.relative(pd.Timestamp(value), at), []
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower(), []
    if isinstance(value, (int, float, np.integer, np.floating)):
        return number(value), []

    text = str(value)
    if _TIMESTAMP.fullmatch(text.strip()):
        return timefmt.relative(pd.Timestamp(text.strip()), at), []
    return escape.split(text)


def fields(row: Mapping, specs: tuple[Field, ...], at: datetime) -> tuple[list[str], list[str]]:
    """Rendered fields and block lines of `row` per `specs`."""
    rendered, blocks = [], []
    for spec in specs:
        if not present(row.get(spec.columns[0])):
            continue
        parts = []
        for column in spec.columns:
            if not present(row.get(column)):
                continue
            text, block = scalar(row[column], at)
            blocks += block
            if text:
                parts.append(text)
        if parts:
            rendered.append(spec.prefix + spec.separator.join(parts))
    return rendered, blocks


def label(row: Mapping) -> str:
    """Escaped label; source table name when MIMIC has none."""
    value = row.get(LABEL)
    text = escape.inline(escape.clean(str(value))).replace("\n", " ") if present(value) else ""
    return text or row["source_table"]


# ── lines ─────────────────────────────────────────────────────────────────


def record(row: Mapping, ref: str, at: datetime, min_delay: timedelta) -> Lines:
    """'[E..] T-05:00 Label: fields; documented T-02:00' (documented only when known >= `min_delay` later)."""
    rendered, block = fields(row, RECORD_FIELDS.get(row["source_table"], ()), at)
    head = _line(f"[{ref}] {timefmt.relative(row[EVENT_TIME], at)} {label(row)}", rendered, block)
    return _documented(head, row, at, min_delay)


def pending(row: Mapping, ref: str, at: datetime) -> Lines:
    rendered, block = fields(row, PENDING_FIELDS.get(row["source_table"], ()), at)
    return _line(f"[{ref}] {timefmt.relative(row[EVENT_TIME], at)} {label(row)}", rendered + [PENDING], block)


def ongoing(row: Mapping, ref: str, at: datetime) -> Lines:
    rendered, block = fields(row, ONGOING_FIELDS.get(row["source_table"], ()), at)
    return _line(f"[{ref}] since {timefmt.relative(row[EVENT_TIME], at)} {label(row)}", rendered + [ONGOING], block)


def core_value(name: str, row: Mapping, ref: str, at: datetime, min_delay: timedelta) -> Lines:
    """'[E..] Potassium: 5.9 mEq/L, abnormal, ref 3.3-5.1, T-05:00; documented T-02:00': full fields, configured name."""
    rendered, block = fields(row, RECORD_FIELDS[row["source_table"]], at)
    head = _line(f"[{ref}] {name}", rendered + [timefmt.relative(row[EVENT_TIME], at)], block)
    return _documented(head, row, at, min_delay)


def documented(event: datetime, known, at: datetime, min_delay: timedelta) -> str | None:
    """'documented T-02:00' when the fact became known at least `min_delay` after its event time, else None.

    Within-window hindsight: a lab drawn T-05:00 and resulted T-02:00 was unknown to a T-04:00 decision.
    Short charting lags stay silent to save tokens.
    """
    if not present(known):
        return None
    event, known = pd.Timestamp(event), pd.Timestamp(known)
    if known <= event or known - event < min_delay:
        return None
    return f"{DOCUMENTED} {timefmt.relative(known, at)}"


def _documented(lines: Lines, row: Mapping, at: datetime, min_delay: timedelta) -> Lines:
    note = documented(row[EVENT_TIME], row.get(AVAILABLE_TIME), at, min_delay)
    return Lines(f"{lines.head}; {note}", lines.block) if note else lines


def citation(name: str, ref_ids: list[str], status: str | None = None) -> Lines:
    """Record owned by another section: status and refs only, e.g. 'PEEP set: see [D..]'."""
    refs = " ".join(f"[{r}]" for r in ref_ids)
    parts = ([status] if status else []) + [f"see {refs}"]
    return Lines(f"{name}: {FIELD_SEPARATOR.join(parts)}")


def decision(row: Mapping, category_name: str, ref: str, order_ref: str | None, start: datetime, at: datetime) -> Lines:
    """'[D..] T-04:50 MEDICATION START Label (SUBCATEGORY): fields; order [E..]; documented T-01:10 (late)'."""
    event = pd.Timestamp(row[EVENT_TIME])
    known = max(event, pd.Timestamp(row["available_time"])) if present(row.get("available_time")) else event

    head = f"[{ref}] {timefmt.relative(event, at)} {category_name} {row[ACTION].name} {label(row)}"
    if present(row.get(SUBCATEGORY)):
        head += f" ({escape.inline(escape.clean(str(row[SUBCATEGORY])))})"

    rendered, block = fields(row, DECISION_FIELDS.get(row["source_table"], ()), at)
    if rendered:
        head += ": " + FIELD_SEPARATOR.join(rendered)
    if order_ref is not None:
        head += f"; order [{order_ref}]"
    if known != event:
        head += f"; {DOCUMENTED} {timefmt.relative(known, at)}"
        if event <= pd.Timestamp(start):
            head += f" {LATE}"
    return Lines(head, tuple(block))


def _line(head: str, rendered: list[str], block: list[str]) -> Lines:
    if rendered:
        head += ": " + FIELD_SEPARATOR.join(rendered)
    return Lines(head, tuple(block))


# ── notes ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NoteSection:
    name: str  # upper case; "" = text before the first heading
    text: str


def note_sections(text: str, recognized: tuple[str, ...]) -> list[NoteSection]:
    """Split at heading lines: configured names (any case) or ALL-CAPS names, e.g. 'FINDINGS:'."""
    known = set(recognized)
    sections, name, lines = [], "", []
    for line in escape.clean(text).split("\n"):
        match = _NOTE_HEADING.match(line)
        heading = match.group(1).strip() if match else ""
        is_heading = heading.upper() in known or (len(heading) >= _MIN_CAPS_HEADING and heading.isupper())
        if not is_heading:
            lines.append(line)
            continue
        if lines:
            sections.append(NoteSection(name, "\n".join(lines)))
        name, lines = heading.upper(), [line]
    if lines:
        sections.append(NoteSection(name, "\n".join(lines)))
    return sections


def shortened_sections(sections: list[NoteSection], priority: tuple[str, ...]) -> list[list[NoteSection]]:
    """Growing candidate subsets, in text order: first present priority section, then each next present one.

    e.g. sections [INDICATION, FINDINGS, IMPRESSION], priority (IMPRESSION, FINDINGS)
         -> [[IMPRESSION], [FINDINGS, IMPRESSION]]
    """
    present_names = [name for name in priority if any(s.name == name for s in sections)]
    candidates = []
    for count in range(1, len(present_names) + 1):
        kept = set(present_names[:count])
        candidates.append([s for s in sections if s.name in kept])
    return candidates


def note(row: Mapping, ref: str, at: datetime, body: str, kept: list[str] | None) -> Lines:
    """'[E..] T-1d 02:00 Radiology report[, SHORTENED (kept: IMPRESSION)]' + '| ' block."""
    head = f"[{ref}] {timefmt.relative(row[EVENT_TIME], at)} {NOTE_TITLES.get(row['source_table'], label(row))}"
    if kept is not None:
        head += f", {SHORTENED} (kept: {FIELD_SEPARATOR.join(kept)})"
    return Lines(head, tuple(escape.block(body)) if body.strip() else ())
