"""Renderer: DecisionWindow -> deterministic clinical text within the token budget.

    window ─► sorted frames ─► ownership + items per section ─► budget ─► RenderedWindow

    1. header, decisions, coverage (mandatory): over total under any counter ─► RenderingBudgetError
    2. current state, trends, notes, past admissions: fill own quota
    3. recent evidence: fill the remainder
    4. coverage with final counts; count the whole text
    5. whole text over total ─► drop recent evidence from the end (then earlier optional sections)

Clinical data only: no instructions to the model (A1 frames the prompt).
"""

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from argus.data import DecisionCategory, DecisionWindow, Domain
from argus.data.catalog import POE_KEY_SEPARATOR
from argus.data.domains import (
    ACTION, EVENT_TIME, HADM_ID, IS_PENDING, IS_TRUNCATED, LABEL, ORDER_ROW, ORDER_TABLE, SOURCE_ROW, SOURCE_TABLE,
    STAY_ID, DecisionAction,
)
from argus.rendering import escape, format, timefmt
from argus.rendering.config import CoreVariable, Demographics, RenderingConfig
from argus.rendering.errors import RenderingBudgetError, RenderingConfigError, RenderingError
from argus.rendering.references import References
from argus.rendering.select import Counters, Filled, Item, Ownership, count, drop, fill, fits
from argus.rendering.trends import Trend, trends
from argus.rendering.types import OPTIONAL_SECTIONS, RenderedWindow, Section, SourceRow

SECTION_SEPARATOR = "\n\n"
HEADINGS = {
    Section.HEADER: "WINDOW",
    Section.DECISIONS: "DECISIONS",
    Section.CURRENT_STATE: "CURRENT STATE",
    Section.TRENDS: "TRENDS",
    Section.RECENT_EVIDENCE: "RECENT EVIDENCE",
    Section.NOTES: "NOTES",
    Section.PAST_ADMISSIONS: "PAST ADMISSIONS",
    Section.COVERAGE: "COVERAGE",
}
COVERAGE_NAMES = {
    Section.CURRENT_STATE: "Current state",
    Section.TRENDS: "Trends",
    Section.RECENT_EVIDENCE: "Recent evidence",
    Section.NOTES: "Notes",
    Section.PAST_ADMISSIONS: "Past admissions",
}
DOMAIN_NAMES = {domain: domain.value.replace("_", " ").capitalize() for domain in Domain}
INDENT = "  "

# Step 5 order: recent evidence first, then quota sections from the last one back.
_OVERFLOW_ORDER = (
    Section.RECENT_EVIDENCE, Section.PAST_ADMISSIONS, Section.NOTES, Section.TRENDS, Section.CURRENT_STATE,
)
_QUOTA_SECTIONS = (Section.CURRENT_STATE, Section.TRENDS, Section.NOTES, Section.PAST_ADMISSIONS)

# Current-state domain orders (spec): ongoing intervals, then pending tests.
_ONGOING_DOMAINS = (Domain.INPUTS, Domain.MEDICATION_ADMINISTRATIONS, Domain.PROCEDURES)
_PENDING_DOMAINS = (Domain.LABS, Domain.MICROBIOLOGY)
_CORE_DOMAINS = {"chartevents": Domain.CHARTED_OBSERVATIONS, "labevents": Domain.LABS}

# Header: admission and ICU rows cited inline; services and care units listed in time order.
_ICUSTAYS = "icustays"
_HEADER_PATHS = (("services", "Services"), ("transfers", "Care units"))

# Notes: past discharge summaries (only clinician-written history at T), current radiology, past radiology;
# newest first in each.
_RADIOLOGY = "radiology"
_DISCHARGE = "discharge"
_NOTE_TABLES = (_RADIOLOGY, _DISCHARGE)

# Recent evidence, poe orders. MIMIC-IV poe names no drug (prescriptions carry it, linked by poe_id) and
# no lab test (no link to labevents). New/Change medication orders duplicate their prescription; stops are
# labelled with the stopped drug; orders left without subtype, drug or detail say nothing and are skipped,
# as are configured routine subtypes without detail (e.g. 'General Care|Activity': level not recorded).
_POE = "poe"
_POE_DRUG_TYPES = frozenset({"Medications", "IV therapy", "TPN"})
_POE_DUPLICATED = frozenset({"New", "Change"})
_ORDER_SUBTYPE = "order_subtype"

# Recent evidence: detail rows shown under their parent record.
_DETAIL_PARENTS = {"emar_detail": ("emar", "emar_id"), "poe_detail": ("poe", "poe_id")}

# Past admissions.
_ADMISSIONS = "admissions"
_PAST_CODES = (("diagnoses_icd", Domain.DIAGNOSES, "Diagnosis"), ("procedures_icd", Domain.PROCEDURES, "Procedure"))
_SEQ_NUM = "seq_num"
_HEADER_RANK, _CODE_RANK = 0, 1

_ACTION_ORDER = {action: rank for rank, action in enumerate(DecisionAction)}
_SORT_KEYS = [EVENT_TIME, SOURCE_TABLE, SOURCE_ROW]


@dataclass(frozen=True)
class _Frames:
    """Snapshot frames sorted by (event_time, source_table, source_row): input row order never matters."""

    current: Mapping[Domain, pd.DataFrame]
    past: Mapping[Domain, pd.DataFrame]


class Renderer:
    def __init__(self, config: RenderingConfig, counters: Mapping[str, Callable[[str], int]]) -> None:
        """`counters` must be exactly the configured tokenizer set: one canonical text per window."""
        if set(counters) != set(config.tokenizers):
            raise RenderingConfigError("counters differ from the configured tokenizer set")

        self._config = config
        self._counters: Counters = {name: counters[name] for name in config.tokenizers}
        self._digest = config.digest()

    def render(self, window: DecisionWindow) -> RenderedWindow:
        return _Render(self._config, self._counters, window).run(self._digest)


class _Render:
    """State of one render call."""

    def __init__(self, config: RenderingConfig, counters: Counters, window: DecisionWindow) -> None:
        self._config = config
        self._counters = counters
        self._window = window
        self._snapshot = window.snapshot
        self._at = pd.Timestamp(window.snapshot.at)
        self._start = pd.Timestamp(window.start)
        self._refs = References()
        self._owners = Ownership()
        self._decision_refs: dict[SourceRow, list[str]] = defaultdict(list)
        self._frames = _Frames(
            current={d: _sorted(window.snapshot.current.get(d)) for d in Domain},
            past={d: _sorted(window.snapshot.past.get(d)) for d in Domain},
        )

    def run(self, digest: str) -> RenderedWindow:
        self._check_notes()
        budget = self._config.budget
        total = {name: budget.total for name in self._counters}

        # Mandatory sections; items of every optional section built in ownership order.
        decisions = self._section(Section.DECISIONS, self._decisions())
        header = self._section(Section.HEADER, self._header())
        items = {
            Section.CURRENT_STATE: self._current_state(),
            Section.TRENDS: self._trends(),
            Section.RECENT_EVIDENCE: self._recent_evidence(),
            Section.NOTES: self._notes(),
            Section.PAST_ADMISSIONS: self._past_admissions(),
        }
        composers = {section: self._composer(section) for section in OPTIONAL_SECTIONS}
        filled = {section: Filled(items[section]) for section in OPTIONAL_SECTIONS}

        # 1. Mandatory content alone, coverage with zero omissions.
        mandatory = SECTION_SEPARATOR.join([header, decisions, self._coverage({s: Filled([]) for s in OPTIONAL_SECTIONS})])
        if not fits(count(mandatory, self._counters), total):
            raise RenderingBudgetError("mandatory sections exceed the total budget")

        # 2. Quota sections.
        quotas = {
            Section.CURRENT_STATE: budget.current_state,
            Section.TRENDS: budget.trends,
            Section.NOTES: budget.notes,
            Section.PAST_ADMISSIONS: budget.past_admissions,
        }
        for section in _QUOTA_SECTIONS:
            quota = {name: quotas[section] for name in self._counters}
            filled[section] = fill(items[section], quota, self._counters, composers[section])

        # 3. Recent evidence fills what the rest of the text leaves.
        filled[Section.RECENT_EVIDENCE].text = composers[Section.RECENT_EVIDENCE](filled[Section.RECENT_EVIDENCE])
        rest = count(self._assemble(header, decisions, filled, without=Section.RECENT_EVIDENCE), self._counters)
        remainder = {name: max(0, total[name] - rest[name]) for name in self._counters}
        filled[Section.RECENT_EVIDENCE] = fill(
            items[Section.RECENT_EVIDENCE], remainder, self._counters, composers[Section.RECENT_EVIDENCE],
        )

        # 4-5. Whole text with final coverage; trim optional content until it fits.
        text = self._assemble(header, decisions, filled)
        tokens = count(text, self._counters)
        while not fits(tokens, total):
            section = next((s for s in _OVERFLOW_ORDER if filled[s].chosen), None)
            if section is None:
                raise RenderingBudgetError("text exceeds the total budget after removing all optional content")
            worst = max(total, key=lambda n: tokens[n] - total[n])
            drop(filled[section], tokens[worst] - total[worst], worst, self._counters, composers[section])
            text = self._assemble(header, decisions, filled)
            tokens = count(text, self._counters)

        sections = {
            Section.HEADER: header,
            Section.DECISIONS: decisions,
            **{s: filled[s].text for s in OPTIONAL_SECTIONS},
            Section.COVERAGE: self._coverage(filled),
        }
        return RenderedWindow(
            text=text,
            refs=self._refs.used_in(text),
            section_tokens={s: count(sections[s], self._counters) for s in Section},
            total_tokens=tokens,
            omissions={s: filled[s].omitted for s in OPTIONAL_SECTIONS},
            rendered_items={s: len(filled[s].chosen) for s in OPTIONAL_SECTIONS},
            config_digest=digest,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )

    # ── assembly ──────────────────────────────────────────────────────────

    def _assemble(self, header: str, decisions: str, filled: dict[Section, Filled], without: Section | None = None) -> str:
        """Sections in fixed order, separated by a blank line, text ending with a newline."""
        parts = [header, decisions]
        parts += [filled[s].text for s in OPTIONAL_SECTIONS if s is not without]
        parts.append(self._coverage(filled))
        return SECTION_SEPARATOR.join(parts) + "\n"

    def _section(self, section: Section, body: list[str]) -> str:
        return "\n".join([f"=== {HEADINGS[section]} ===", *(body or [format.NO_RECORDS])])

    def _composer(self, section: Section) -> Callable[[Filled], str]:
        """Section text of a selection: items in display order, omission line last."""

        def compose(filled: Filled) -> str:
            if not filled.items:
                return self._section(section, [])

            body = self._grouped(filled) if section is Section.RECENT_EVIDENCE else [t for _, t in filled.selected()]
            if filled.omitted:
                body.append(f"{filled.omitted} RECORDS OMITTED BY RENDERING BUDGET")
            return "\n".join([f"=== {HEADINGS[section]} ===", *body])

        return compose

    def _grouped(self, filled: Filled) -> list[str]:
        """Recent evidence: one subheading per domain, domains in priority order (display key starts with rank)."""
        lines, current = [], None
        for item, text in filled.selected():
            domain = self._config.recent_domain_priority[item.display[0]]
            if domain is not current:
                lines.append(f"--- {DOMAIN_NAMES[domain]} ---")
                current = domain
            lines.append(text)
        return lines

    def _coverage(self, filled: Mapping[Section, Filled]) -> str:
        """Counts only, e.g. 'Recent evidence: 120 rendered, 41 omitted by rendering budget.'"""
        lines = []
        for section in OPTIONAL_SECTIONS:
            name, state = COVERAGE_NAMES[section], filled[section]
            if not state.items:
                lines.append(f"{name}: {format.NO_RECORDS}.")
                continue
            line = f"{name}: {len(state.chosen)} rendered, {state.omitted} omitted by rendering budget"
            if section is Section.NOTES:
                line += f", {state.shortened} shortened"
            lines.append(line + ".")
        return self._section(Section.COVERAGE, lines)

    # ── 1. header ─────────────────────────────────────────────────────────

    def _header(self) -> list[str]:
        demographics, encounter, at = self._snapshot.demographics, self._snapshot.current_encounter, self._at
        first, last = demographics.care_years
        lines = [
            f"Decisions visible in ({timefmt.relative(self._start, at)}, {timefmt.relative(at, at)}]. "
            "Times relative to window end T.",
            f"Patient: {demographics.age_at_t} y, {_text(demographics.sex)}. Care years {first}-{last}.",
        ]
        if self._config.demographics is Demographics.FULL:
            extra = (
                ("Race", demographics.race), ("Language", demographics.language),
                ("Insurance", demographics.insurance), ("Marital status", demographics.marital_status),
            )
            lines.append(" ".join(f"{name}: {_text(value)}." for name, value in extra if format.present(value)))

        # Admission context known at T, e.g. "Admission [E..]: EW EMER., from EMERGENCY ROOM, hospital day 3 (...)".
        arrival = f" ED arrival {timefmt.relative(encounter.edregtime, at)}." if encounter.edregtime else ""
        lines.append(
            f"Admission{self._encounter_ref(_ADMISSIONS, HADM_ID, encounter.hadm_id)}: "
            f"{_text(encounter.admission_type)}, from {_text(encounter.admission_location)}, "
            f"hospital day {timefmt.day_number(encounter.admittime, at)} ({timefmt.relative(encounter.admittime, at)})."
            f"{arrival}"
        )
        lines.append(
            f"ICU{self._encounter_ref(_ICUSTAYS, STAY_ID, encounter.stay_id)}: {_text(encounter.first_careunit)}, "
            f"ICU day {timefmt.day_number(encounter.icu_intime, at)} ({timefmt.relative(encounter.icu_intime, at)})."
        )
        lines += [self._encounter_path(table, name) for table, name in _HEADER_PATHS]
        return lines

    def _encounter_ref(self, table: str, column: str, value: int) -> str:
        """' [E..]' for the current admission or ICU stay row, owned by the header; '' when absent."""
        frame = self._frames.current[Domain.ENCOUNTERS]
        rows = _records(frame[(frame[SOURCE_TABLE] == table) & (frame[column] == value)])
        if not rows:
            return ""

        source = _source(rows[0])
        self._owners.claim(source, Section.HEADER)
        return f" [{self._refs.record(source)}]"

    def _encounter_path(self, table: str, name: str) -> str:
        """Every visible service or care-unit row in time order, e.g. "Services: MED [E..] T-2d 03:00; CSURG [E..] T-05:10."
        Bounded by admission (max 34 transfers in MIMIC-IV), so mandatory."""
        frame = self._frames.current[Domain.ENCOUNTERS]
        entries = []
        for row in _records(frame[frame[SOURCE_TABLE] == table]):
            source = _source(row)
            self._owners.claim(source, Section.HEADER)
            time = timefmt.relative(pd.Timestamp(row[EVENT_TIME]), self._at)
            entries.append(f"{_text(row.get(LABEL))} [{self._refs.record(source)}] {time}")

        return f"{name}: {'; '.join(entries) if entries else format.NOT_RECORDED}."

    # ── 2. decisions ──────────────────────────────────────────────────────

    def _decisions(self) -> list[str]:
        """Every decision, one subheading per category in enum order, empty ones included."""
        lines = []
        for category in DecisionCategory:
            lines.append(f"--- {category.name} ---")
            rows = _records(self._window.get(category))
            rows.sort(key=lambda r: (r[EVENT_TIME], _ACTION_ORDER[r[ACTION]], r[SOURCE_TABLE], r[SOURCE_ROW]))
            if not rows:
                lines.append(format.NO_RECORDS)

            for row in rows:
                source = _source(row)
                self._owners.claim(source, Section.DECISIONS)
                ref = self._refs.decision(source, category, row[ACTION], row[EVENT_TIME])
                self._decision_refs[source].append(ref)

                order_ref = None
                if format.present(row.get(ORDER_TABLE)) and format.present(row.get(ORDER_ROW)):
                    order_ref = self._refs.record(SourceRow(str(row[ORDER_TABLE]), int(row[ORDER_ROW])))
                lines += format.decision(row, category.name, ref, order_ref, self._start, self._at).all()
        return lines

    # ── 3. current state ──────────────────────────────────────────────────

    def _current_state(self) -> list[Item]:
        """Core variables (latest value or NOT RECORDED), ongoing intervals, pending tests."""
        items = [self._core_item(variable) for variable in self._config.core_variables]

        for domains, flag, render, status in (
            (_ONGOING_DOMAINS, IS_TRUNCATED, format.ongoing, format.ONGOING),
            (_PENDING_DOMAINS, IS_PENDING, format.pending, format.PENDING),
        ):
            for domain in domains:
                frame = self._frames.current[domain]
                for row in _records(frame[frame[flag].astype(bool)]):
                    items.append(self._owned_item(row, Section.CURRENT_STATE, render, status))
        return items

    def _core_item(self, variable: CoreVariable) -> Item:
        observed = self._core_rows(variable)
        observed = observed[~observed[IS_PENDING].astype(bool)]
        observed = observed[observed["value"].notna() | observed["valuenum"].notna()]
        if observed.empty:
            return Item((f"{variable.label}: {format.NOT_RECORDED}",))

        row = _records(observed.tail(1))[0]
        source = _source(row)
        if not self._owners.claim(source, Section.CURRENT_STATE):
            return Item((format.citation(variable.label, self._citations(source)).head,))
        return Item(_joined(format.core_value(variable.label, row, self._refs.record(source), self._at)))

    def _owned_item(self, row: dict, section: Section, render, status: str) -> Item:
        """Full line if `section` owns the row, else a citation with its status."""
        source = _source(row)
        if not self._owners.claim(source, section):
            return Item((format.citation(format.label(row), self._citations(source), status).head,))
        return Item(_joined(render(row, self._refs.record(source), self._at)))

    def _core_rows(self, variable: CoreVariable) -> pd.DataFrame:
        frame = self._frames.current[_CORE_DOMAINS[variable.table]]
        return frame[(frame[SOURCE_TABLE] == variable.table) & (frame["itemid"] == variable.itemid).fillna(False)]

    def _citations(self, source: SourceRow) -> list[str]:
        """Decision refs of a decision-owned row, else its record ref."""
        return self._decision_refs.get(source) or [self._refs.record(source)]

    # ── 4. trends ─────────────────────────────────────────────────────────

    def _trends(self) -> list[Item]:
        items = []
        for variable in self._config.core_variables:
            series = trends(
                self._core_rows(variable), self._at, self._config.trend_lookback, self._config.trend_min_points,
            )
            for trend in series:
                for source in trend.sources:
                    self._owners.claim(source, Section.TRENDS)
                items.append(Item((self._trend_line(variable, trend),)))
        return items

    def _trend_line(self, variable: CoreVariable, trend: Trend) -> str:
        """'[T1] Heart rate (bpm), last 24h: n 22; first 98 T-23:40 [E..]; min ..; max ..; change +20 to latest [E..]'."""
        def point(name, p):
            return f"{name} {format.number(p.value)} {timefmt.relative(p.time, self._at)} [{self._refs.record(p.source)}]"

        unit = f" ({_text(trend.unit)})" if trend.unit else ""
        hours = int(self._config.trend_lookback.total_seconds() // 3600)
        change = format.number(trend.change)
        change = change if change.startswith("-") else f"+{change}"
        parts = [
            f"n {len(trend.points)}",
            point("first", trend.first),
            point("min", trend.minimum),
            point("max", trend.maximum),
            f"change {change} to latest [{self._refs.record(trend.latest.source)}]",
        ]
        ref = self._refs.trend(trend.sources)
        return f"[{ref}] {variable.label}{unit}, last {hours}h: " + "; ".join(parts)

    # ── 5. recent evidence ────────────────────────────────────────────────

    def _recent_evidence(self) -> list[Item]:
        """Unowned current records in the lookback; in-window first, domain priority, newest first."""
        lower = self._at - self._config.recent_lookback
        candidates = []
        for rank, domain in enumerate(self._config.recent_domain_priority):
            frame = self._frames.current[domain]
            frame = frame[(frame[EVENT_TIME] > lower) & (frame[EVENT_TIME] <= self._at)]
            rows = _records(frame)
            if domain is Domain.ORDERS:
                rows = self._orders(rows)
            rows = [r for r in rows if self._owners.claim(_source(r), Section.RECENT_EVIDENCE)]
            candidates += self._with_details(rank, rows)

        candidates.sort(key=lambda c: c[0])
        return [item for _, item in candidates]

    def _orders(self, rows: list[dict]) -> list[dict]:
        """poe rows that say something, e.g. 'Medications: D/C' -> 'Medications: Furosemide, D/C'."""
        drugs = self._drugs()
        detailed = {r.get("poe_id") for r in rows if r[SOURCE_TABLE] != _POE}
        duplicates = set()
        result = []
        for row in rows:
            if row[SOURCE_TABLE] != _POE or row.get("order_type") not in _POE_DRUG_TYPES:
                result.append(row)
                continue

            # Duplicate of its prescription; its detail rows go with it.
            if row.get("transaction_type") in _POE_DUPLICATED and row.get("poe_id") in drugs:
                duplicates.add(row["poe_id"])
                continue

            drug = drugs.get(row.get("discontinue_of_poe_id"))
            if drug is not None and not format.present(row.get(_ORDER_SUBTYPE)):
                row = {**row, _ORDER_SUBTYPE: drug}
            result.append(row)

        result = [r for r in result if r[SOURCE_TABLE] == _POE or r.get("poe_id") not in duplicates]
        return [r for r in result if r[SOURCE_TABLE] != _POE or r.get("poe_id") in detailed or self._has_content(r)]

    def _has_content(self, row: Mapping) -> bool:
        """poe row without detail: a subtype (or resolved drug) not listed as contentless."""
        subtype = row.get(_ORDER_SUBTYPE)
        if not format.present(subtype):
            return False
        return f"{row.get('order_type')}{POE_KEY_SEPARATOR}{subtype}" not in self._config.contentless_orders

    def _drugs(self) -> dict[str, str]:
        """poe_id -> drug of visible prescriptions (current and past), first in sorted order."""
        drugs: dict[str, str] = {}
        for frames in (self._frames.current, self._frames.past):
            for row in _records(frames[Domain.MEDICATION_ORDERS]):
                if format.present(row.get("poe_id")) and format.present(row.get("drug")):
                    drugs.setdefault(row["poe_id"], str(row["drug"]))
        return drugs

    def _with_details(self, rank: int, rows: list[dict]) -> list[tuple[tuple, Item]]:
        """(selection key, item); detail rows (emar_detail, poe_detail) indented under their parent."""
        children: dict[tuple[str, str], list[dict]] = defaultdict(list)
        parents = {(r[SOURCE_TABLE], r.get(key)) for r in rows for key in ("emar_id", "poe_id") if format.present(r.get(key))}
        units = []
        for row in rows:
            parent = _DETAIL_PARENTS.get(row[SOURCE_TABLE])
            if parent is not None and (parent[0], row.get(parent[1])) in parents:
                children[(parent[0], row[parent[1]])].append(row)
            else:
                units.append(row)

        result = []
        for row in units:
            lines = format.record(row, self._refs.record(_source(row)), self._at).all()
            key_column = next((p[1] for p in _DETAIL_PARENTS.values() if p[0] == row[SOURCE_TABLE]), None)
            for child in children.get((row[SOURCE_TABLE], row.get(key_column)), []) if key_column else []:
                child_lines = format.record(child, self._refs.record(_source(child)), self._at)
                if _informative(child):
                    lines += [INDENT + line for line in child_lines.all()]

            time = pd.Timestamp(row[EVENT_TIME])
            in_window = time > self._start
            selection = (not in_window, rank, -time.value, row[SOURCE_TABLE], row[SOURCE_ROW])
            display = (rank, time, row[SOURCE_TABLE], row[SOURCE_ROW])
            result.append((selection, Item(("\n".join(lines),), display)))
        return result

    # ── 6. notes ──────────────────────────────────────────────────────────

    def _check_notes(self) -> None:
        """D1 must never release a current-admission discharge note; fail closed if one appears."""
        current_hadm = self._snapshot.current_encounter.hadm_id
        for frames in (self._frames.current, self._frames.past):
            notes = frames[Domain.NOTES]
            discharge = notes[notes[SOURCE_TABLE] == _DISCHARGE]
            if frames is self._frames.current and not discharge.empty:
                raise RenderingError("current-admission discharge note in snapshot")
            if (discharge[HADM_ID] == current_hadm).fillna(False).any():
                raise RenderingError("current-admission discharge note in snapshot")

    def _notes(self) -> list[Item]:
        """Whole note first; shortened variants (most sections first) when the whole does not fit."""
        current, past = self._frames.current[Domain.NOTES], self._frames.past[Domain.NOTES]
        groups = (
            past[past[SOURCE_TABLE] == _DISCHARGE],
            current[current[SOURCE_TABLE] == _RADIOLOGY],
            past[past[SOURCE_TABLE] == _RADIOLOGY],
        )

        items = []
        for frame in groups:
            for row in reversed(_records(frame)):
                source = _source(row)
                self._owners.claim(source, Section.NOTES)
                ref = self._refs.record(source)
                text = str(row["text"]) if format.present(row.get("text")) else ""

                variants = ["\n".join(format.note(row, ref, self._at, text, None).all())]
                sections = format.note_sections(text, self._config.note_section_priority)
                for kept in reversed(format.shortened_sections(sections, self._config.note_section_priority)):
                    body = "\n".join(s.text for s in kept)
                    names = [s.name for s in kept]
                    variants.append("\n".join(format.note(row, ref, self._at, body, names).all()))
                items.append(Item(tuple(variants)))
        return items

    # ── 7. past admissions ────────────────────────────────────────────────

    def _past_admissions(self) -> list[Item]:
        """Pass 1: every admission with its first N diagnoses and procedures; pass 2: remaining codes.
        Newest admission first in both passes."""
        for domain, frame in self._frames.past.items():
            if domain is Domain.NOTES:
                continue
            for source in map(_source, _records(frame[[SOURCE_TABLE, SOURCE_ROW]])):
                self._owners.claim(source, Section.PAST_ADMISSIONS)

        encounters = self._frames.past[Domain.ENCOUNTERS]
        admissions = _records(encounters[encounters[SOURCE_TABLE] == _ADMISSIONS])
        admissions.sort(key=lambda r: (r["admittime"], r[SOURCE_ROW]), reverse=True)

        first_pass, second_pass = [], []
        limit = self._config.past_dx_per_admission
        for rank, admission in enumerate(admissions):
            first_pass.append(Item((self._admission_line(admission),), (rank, _HEADER_RANK, 0, 0, 0)))
            for kind, (table, domain, name) in enumerate(_PAST_CODES):
                codes = self._codes(admission[HADM_ID], table, domain)
                for position, row in enumerate(codes):
                    line = f"{INDENT}[{self._refs.record(_source(row))}] {name} {_seq(row)}: {format.label(row)}"
                    item = Item((line,), (rank, _CODE_RANK, kind, _seq(row), row[SOURCE_ROW]))
                    (first_pass if position < limit else second_pass).append(item)
        return first_pass + second_pass

    def _admission_line(self, row: dict) -> str:
        """'[E..] Admission 412d 03:00 before current admission, length of stay 6d 02:00: EW EMER., discharged to HOME'."""
        current = pd.Timestamp(self._snapshot.current_encounter.admittime)
        before = timefmt.duration(max(current - pd.Timestamp(row["admittime"]), pd.Timedelta(0)))
        stay = "unknown"
        if format.present(row.get("dischtime")):
            stay = timefmt.duration(max(pd.Timestamp(row["dischtime"]) - pd.Timestamp(row["admittime"]), pd.Timedelta(0)))

        rendered, _ = format.fields(row, format.RECORD_FIELDS[_ADMISSIONS], self._at)
        fields = format.FIELD_SEPARATOR.join([format.label(row), *rendered])
        ref = self._refs.record(_source(row))
        return f"[{ref}] Admission {before} before current admission, length of stay {stay}: {fields}"

    def _codes(self, hadm_id, table: str, domain: Domain) -> list[dict]:
        frame = self._frames.past[domain]
        frame = frame[(frame[SOURCE_TABLE] == table) & (frame[HADM_ID] == hadm_id).fillna(False)]
        rows = _records(frame)
        rows.sort(key=lambda r: (_seq(r), r[SOURCE_ROW]))
        return rows


# ── helpers ───────────────────────────────────────────────────────────────


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(_SORT_KEYS, kind="mergesort").reset_index(drop=True)


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.to_dict("records")


def _source(row: Mapping) -> SourceRow:
    return SourceRow(str(row[SOURCE_TABLE]), int(row[SOURCE_ROW]))


def _seq(row: Mapping) -> int:
    value = row.get(_SEQ_NUM)
    return int(value) if format.present(value) else 0


def _text(value) -> str:
    return escape.inline(escape.clean(str(value))).replace("\n", " ") if format.present(value) else "unknown"


def _joined(lines: format.Lines) -> tuple[str]:
    return ("\n".join(lines.all()),)


def _informative(row: Mapping) -> bool:
    """Detail row with a label or any whitelisted value; empty eMAR detail rows are noise."""
    specs = format.RECORD_FIELDS.get(row[SOURCE_TABLE], ())
    has_field = any(format.present(row.get(spec.columns[0])) for spec in specs)
    return has_field or format.present(row.get("label"))
