# Spec: Window rendering to prompt text (D5)

## Context

ARGUS agents evaluate clinical decisions within sequential 6 h ICU windows. D1 provides a leak-free patient snapshot, D2 separates decisions from the available clinical evidence, and D4 provides provider-independent token counting and LLM inference.

D5 converts a `DecisionWindow` into deterministic, clinically readable text that fits the configured models' context budgets.

The renderer must preserve the distinction between **decisions**, **observed evidence**, and **missing or unavailable information**. It must not introduce hindsight information, silently discard decisions, or make clinical interpretations.

Information order and quantity are experimental variables: Hager et al. (2024) report model sensitivity to both. D5 therefore fixes rendering order, selection rules and budgets before agent development.

Roadmap exit criterion:

> Token count per domain measured on real windows per configured tokenizer; budget fixed for trends, notes and past admissions.

## Design

```
SnapshotEngine.get_window(...)            config/rendering.toml
         |                                        |
         v                                        v
    DecisionWindow ──────────────────►  Renderer(config, counters)
                                                  |
                     rendering/                   |
                       timefmt.py     Relative clinical times
                       escape.py      Untrusted free text
                       references.py  Evidence reference IDs
                       format.py      Record and decision lines
                       trends.py      Descriptive statistics
                       select.py      Ownership, priority, quotas
                       renderer.py    Assembly, token counting
                                                  |
                                                  v
                                           RenderedWindow
                                     text | refs | tokens | omissions
                                                  |
                                                  v
                                   A1 (wraps text in agent prompt)
                                                  |
                                                  v
                                            D4 LlmClient
```

D5 consumes only the D1 and D2 public interfaces. It does not query the store and does not implement time visibility rules.

D5 does not call an LLM. Rendering, selection and trend extraction are deterministic.

### Responsibilities

| Component | Responsibility |
|---|---|
| D1 | Clinical data visibility, redaction and snapshot assembly |
| D2 | Identification and classification of decisions |
| D4 | Token counting, model capabilities and request context enforcement |
| D5 | Clinical text, record selection, evidence budgets, evidence references |
| D8 | Agent communication, role access and evidence exchange |
| A1 | Agent prompt framing, request preflight, run logging and replay |
| A2–A5 | Clinical interpretation, reasoning and output |

D5 produces **clinical data only**. It contains no instructions to the model. Framing such as "the record is retrospective and deidentified", "consider only information available at window end" or "a missing record is not evidence of an action not performed" belongs to A1's prompt templates.

## Public API

New package: `src/argus/rendering/`.

```python
class Section(Enum):
    HEADER = "header"
    DECISIONS = "decisions"
    CURRENT_STATE = "current_state"
    TRENDS = "trends"
    RECENT_EVIDENCE = "recent_evidence"
    NOTES = "notes"
    PAST_ADMISSIONS = "past_admissions"
    COVERAGE = "coverage"

class RefKind(Enum):
    RECORD = "record"        # one source row
    DECISION = "decision"    # one D2 decision occurrence
    TREND = "trend"          # statistics derived from several rows

@dataclass(frozen=True)
class SourceRow:
    table: str               # Table.source_name
    row: int                 # source_row

@dataclass(frozen=True)
class EvidenceRef:
    kind: RefKind
    sources: tuple[SourceRow, ...]          # exactly 1 for RECORD and DECISION; >= 1 for TREND
    category: DecisionCategory | None       # DECISION only
    action: DecisionAction | None           # DECISION only

@dataclass(frozen=True)
class RenderedWindow:
    text: str
    refs: Mapping[str, EvidenceRef]                     # prompt ID -> reference
    section_tokens: Mapping[Section, Mapping[str, int]] # section -> backend id -> tokens
    total_tokens: Mapping[str, int]                     # backend id -> tokens of `text`
    omissions: Mapping[Section, int]                    # records not rendered for budget
    config_digest: str                                  # SHA256 of canonical config
    content_hash: str                                   # SHA256 of UTF-8 `text`

class Renderer:
    def __init__(self, config: RenderingConfig, counters: Mapping[str, Callable[[str], int]]) -> None: ...
    def render(self, window: DecisionWindow) -> RenderedWindow: ...
```

`counters` maps backend IDs to `LlmClient.count_text_tokens`.

**Fixed tokenizer set.** `config/rendering.toml` lists the backend IDs whose tokenizers constrain rendering. The constructor raises `RenderingConfigError` unless `counters` has exactly those keys. The caller cannot change the set, so the same window always yields the same text, whichever backend later consumes it. This is what makes A7 arms comparable and A1 replay exact.

Selection fits the strictest counter. `section_tokens` reports every counter; `total_tokens` counts the complete text and is not assumed to be the sum of sections.

`RenderingBudgetError` is raised when mandatory content cannot fit. No partial result is returned.

## Time representation

MIMIC dates are shifted per patient; calendar dates such as 2180-03-14 are meaningless to the model and may mislead it. Intervals within a patient are preserved.

All times are rendered **relative to the window end `at`**:

| Offset | Rendering |
|---|---|
| < 24 h before `at` | `T-02:15` |
| >= 24 h before `at` | `T-3d 04:15` |
| At `at` | `T-00:00` |

Durations use the same unit format, e.g. `duration 3d 02:00`.

The header gives anchors: hours since hospital admission, hours since ICU admission and ICU day (day 1 = first 24 h). Past admissions are placed relative to the current admission, e.g. `admitted 412d before current admission, length of stay 6d`.

`care_years` is the only calendar information rendered, for guideline-era context.

## Rendering structure

Fixed section order:

| Order | Section | Content | Tier |
|---|---|---|---|
| 1 | Header | Window bounds, demographics, encounter anchors | Mandatory |
| 2 | Decisions | All D2 decisions in the window | Mandatory |
| 3 | Current state | Latest value of each core variable, ongoing intervals, pending tests | Optional, quota |
| 4 | Trends | Statistics of core variable series | Optional, quota |
| 5 | Recent evidence | Remaining current-admission records in the lookback | Optional, remainder |
| 6 | Notes | Radiology reports, past discharge summaries | Optional, quota |
| 7 | Past admissions | Prior encounters, diagnoses, procedures | Optional, quota |
| 8 | Coverage | Omission counts, shortened notes, empty sections | Mandatory |

Mandatory content is exactly sections 1, 2 and 8. Every section heading is always rendered; an empty section says `NO RECORDS AVAILABLE`.

Example (illustrative values only):

```
=== WINDOW ===
Decisions visible in (T-06:00, T-00:00]. Times relative to window end T.
Patient: 67 y, F. Care years 2014-2016.
Admission: EW EMER., hospital day 3 (T-2d 05:10). ICU: MICU, ICU day 2 (T-1d 08:40).

=== DECISIONS ===
[D7Q2KX1MA] T-04:50 MEDICATION START Norepinephrine (DRIP) 0.08 mcg/kg/min IV, order [E3HP0RT2C]
[D9LAA02WF] T-01:10 TEST_ORDER ORDER Lactate; documented T-01:10, event T-07:30 (late)

=== CURRENT STATE ===
[E5KM1ZQ0B] Heart Rate 118 bpm T-00:20
[E0JX8W3NR] Lactate PENDING, drawn T-01:05
[E2CC7YH4D] Norepinephrine 0.08 mcg/kg/min, ONGOING AT WINDOW END since T-04:50
...

=== COVERAGE ===
Recent evidence: 41 records omitted by rendering budget.
Notes: 1 report shortened (IMPRESSION kept).
```

Section markers (`=== NAME ===`) are reserved; see Escaping.

### 1. Header

- Window bounds as relative times: start exclusive, end inclusive.
- Age at `at` and sex.
- `care_years`.
- Admission type, hospital admission anchor, ICU care unit, ICU admission anchor.

**Excluded demographics:** `race`, `language`, `insurance` and `marital_status` are not rendered. They are not needed to judge ICU decisions and may introduce demographic bias into flags. Including them is a separate experimental condition (see Configuration).

Encounter identifiers are not rendered.

### 2. Decisions

All seven `DecisionCategory` values are rendered as subheadings, in enum order, including empty ones.

Each decision line contains:

- Decision ref ID.
- Clinical decision time (`event_time`).
- Category and action, taken unchanged from D2.
- Label and subcategory.
- Dose, rate, amount, route, unit or setting, using a fixed per-source-table column whitelist in `format.py`.
- Linked order ref ID when D2 supplies `order_table`/`order_row`.
- `documented <time>` when `max(event_time, available_time)` differs from `event_time`, marked `(late)` when `event_time` precedes the window start.

Order: `event_time`, then category enum order, then action enum order, then `source_table`, `source_row`.

One source row yielding several decisions (e.g. procedure start and stop) gives several decision lines, each with its own decision ref, all pointing to the same `SourceRow`.

No decision is ever dropped for budget. If sections 1, 2 and 8 alone exceed the total budget, rendering fails.

### Record ownership

A source record's full rendering appears **at most once** in the text. Ownership is assigned in this order; the first match wins:

1. **Decisions:** rows that are a decision's source row.
2. **Current state:** latest visible record of each core variable; records with `is_truncated` (ongoing intervals); records with `is_pending`.
3. **Trends:** remaining observations of core variables in the trend lookback. They are summarised, never listed individually.
4. **Recent evidence:** remaining current-admission records with `event_time` in the recent lookback.
5. **Notes:** `NOTES` domain records.
6. **Past admissions:** `past` record frames, except notes.

Other sections may cite an owned record by its ref ID but never repeat its content. Trend statistics cite the records they highlight.

### Core variables

A fixed list in `config/rendering.toml`, keyed by `(source table, itemid)` with a display label. The initial list:

| Group | Variables |
|---|---|
| Vital signs | Heart rate, arterial and non-invasive mean BP, systolic BP, respiratory rate, SpO2, temperature |
| Neurology | GCS eye, verbal, motor |
| Respiratory support | FiO2, PEEP set, ventilator mode |
| Labs | Lactate, creatinine, urea nitrogen, potassium, sodium, glucose, hemoglobin, platelets, WBC, bilirubin total, INR, arterial pH, pO2, pCO2 |

`scripts/measure_rendering.py` reports each core variable's presence rate. Changing the list changes `config_digest`.

### 3. Current state

For each core variable, in list order: latest visible value, unit and time, or `NOT RECORDED` when no value exists in the current admission.

Then, in fixed domain order (inputs, medication administrations, procedures):

- Records with `is_truncated`: visible state with `ONGOING AT WINDOW END`. D1 has already clipped the interval; D5 never reconstructs end times, status or totals.

Then, in fixed domain order (labs, microbiology):

- Records with `is_pending`: test identity, draw or collection time and `PENDING`. No value, organism, interpretation or susceptibility appears.

### 4. Trends

For each core variable with at least `trend_min_points` numeric observations sharing one unit within `trend_lookback` (default 24 h, ending at `at`):

- Count.
- First, minimum, maximum with times and ref IDs.
- Absolute change, latest minus first. Latest is cited by its ref ID in current state.

Observations with differing units form separate series; nothing is converted. Non-numeric, null or pending values are excluded from the statistics. No interpolation, no assumption of regular sampling.

The trend's `TREND` ref lists every contributing `SourceRow`. Highlighted points get `RECORD` refs.

D5 computes descriptive statistics only: no severity scores, deterioration judgements or treatment-effect claims.

### 5. Recent evidence

Records not owned earlier, with `event_time` in `recent_lookback` (default 24 h, ending at `at`), selected by:

1. Records with `event_time` in the window before pre-window records.
2. Domain priority, fixed in config:
   1. Medication administrations
   2. Inputs
   3. Labs
   4. Microbiology
   5. Outputs
   6. Procedures
   7. Medication orders
   8. Orders
   9. Charted observations
   10. Encounters
   11. Outpatient measurements
3. Newest `event_time` first.
4. `source_table`, `source_row`.

Records are rendered grouped by domain, chronologically within a domain. Selection order and display order are distinct: selection decides *what* fits, display decides *how* it reads.

Each record shows label, value, unit, qualifiers (flag, comments) and time. Units are not converted. Item identifiers are never rendered.

### 6. Notes

D1 exposes current-admission radiology reports and past-admission discharge notes. Current-admission discharge notes are prohibited by D1 and must never be rendered; the renderer asserts this and raises if one appears.

Selection order:

1. Current-admission radiology, newest first.
2. Past discharge summaries, newest first.
3. Past radiology, newest first.

A note that fits is rendered whole. A note that does not fit the remaining notes quota is shortened at explicit section boundaries: the `IMPRESSION` (or `FINDINGS` when absent) section first, then following sections whole while they fit. A note whose first kept section does not fit is omitted. Every shortened note is labelled `SHORTENED` with the kept section names.

No LLM summarisation.

### 7. Past admissions

Newest admission first. Per admission:

- Relative admission time and length of stay.
- Admission type and discharge location.
- Diagnoses in coded sequence order.
- ICD procedures in coded sequence order.

Fill policy: first pass renders the first `past_dx_per_admission` diagnoses and procedures of every admission; second pass adds remaining codes, newest admission first. This favours breadth across admissions over depth in one.

Past discharge summaries belong to Notes.

Current-admission diagnoses are not visible at `at`: D1 releases coded diagnoses only after discharge. D5 does not work around this.

### 8. Coverage

Per optional section: records rendered, records omitted by budget, notes shortened. Empty sections are listed. Counts only; no instruction text.

### Missingness labels

| State | Rendering |
|---|---|
| Observed value | Value, unit, time, ref ID |
| Pending result | `PENDING` |
| Core variable never recorded in current admission | `NOT RECORDED` |
| Ongoing interval | `ONGOING AT WINDOW END` |
| Section without records | `NO RECORDS AVAILABLE` |
| Records omitted by budget | `N RECORDS OMITTED BY RENDERING BUDGET` in the section and in Coverage |

## Escaping

All free text is untrusted: note bodies, lab and microbiology comments, order text, and any other string column not taken from a dictionary label.

Rules, applied in `escape.py`:

1. Normalise line endings to `\n`; remove control characters except `\n` and `\t`.
2. Render multi-line text as a block with every line prefixed by `| `.
3. Single-line values containing `===` are rendered with the sequence replaced by `= = =`.

Consequence: no untrusted text can produce a line starting with `===`, so it cannot open a section or impersonate structure. Escaping is structural only; it does not guarantee the model ignores embedded instructions (see Residual Risks).

## Evidence references

Prompt IDs are opaque, short and stable:

| Kind | ID | Derivation |
|---|---|---|
| Record | `E` + 8 chars | Base32 of SHA256(`"{table}:{row}"`), truncated |
| Decision | `D` + 8 chars | Base32 of SHA256(`"{table}:{row}:{category}:{action}:{event_time}"`), truncated |
| Trend | `T` + index | Per-window sequence in render order |

Record and decision IDs are stable across windows for a fixed store version. Trend IDs are window-local.

Collisions: the renderer checks that every ID in a window maps to one key. A collision raises `RenderingError`; it is never resolved silently. The measurement script also reports any collision across all sampled windows. 40 bits gives a negligible collision probability at per-patient scale; a reported collision triggers a longer ID and a format version bump.

The ref map is returned separately from the text. A1 persists it; D8 defines how agent claims cite it. A reference proves provenance, not clinical correctness.

## Token budget

### Initial budgets

| Component | Initial tokens | Notes |
|---|---:|---|
| Total clinical text | 16,000 | Leaves room within the smallest configured context, 32,768 (mistral24) |
| Current state | 3,000 | Quota |
| Trends | 2,500 | Quota |
| Notes | 3,000 | Quota |
| Past admissions | 1,500 | Quota |
| Recent evidence | Remainder | Total minus mandatory minus used quotas |

These are **starting values, not results**. Measurement (below) revises or confirms them; final values are fixed in `config/rendering.toml`.

### Enforcement

1. Render mandatory sections (1, 2 and 8 with zero omissions). If over total with any counter, raise `RenderingBudgetError`.
2. For each quota section in order (3, 4, 6, 7): add records in selection order while the section's rendered text fits its quota under every counter. Stop at the first record that does not fit.
3. Recent evidence fills the remaining total in the same way.
4. Re-render Coverage with final counts. Count the complete text with every counter.
5. If the complete text exceeds the total (section boundaries or the coverage update can add tokens), remove recent-evidence records from the end of the selection order, then re-count. Repeat until it fits.

Section fits are checked incrementally: each candidate record's line is counted once and running sums are used, with the exact section text re-counted only at the boundary. Step 5 runs only on overflow, so typical cost is one full-text count per counter.

No mandatory content is removed. Text is never cut at an arbitrary character or token position.

Before inference, A1 counts the complete request with D4's `count_request_tokens()` and reserves output tokens. Any overflow there is reported, never fixed by altering clinical content.

### Backend comparability

The fixed tokenizer set means one canonical text per window. All backends in a matched A7 arm receive identical clinical content. Experiments that vary budgets, order, core variables or demographic inclusion use a different config and are recorded as separate conditions via `config_digest`.

## Configuration

`config/rendering.toml`:

- `format_version`.
- `tokenizers`: backend IDs from `config/llm.toml`.
- Budgets: total and per-section quotas.
- `recent_lookback`, `trend_lookback`, `trend_min_points`, `past_dx_per_admission`.
- Core variable list.
- Recent evidence domain priority.
- Note section priority (`IMPRESSION`, `FINDINGS`, ...).
- `demographics`: `minimal` (age, sex; default) or `full`.

Strict validation: unknown keys, negative values, quotas summing above the total, unknown backend IDs or duplicate core variables raise `RenderingConfigError`.

No clinical data, credentials or prompts in this file. Any change alters `config_digest`; A1 records it and V1 freezes it.

## Real-window measurement

`scripts/measure_rendering.py`, using `SnapshotEngine.get_window()` and D4 counters only. No inference.

**Sample:**

- 200 seeded random valid ICU windows.
- 50 high-volume windows: top 50 by visible current-admission record count among 2,000 seeded candidates.

**Validation-cohort hygiene:** D6 does not exist yet. The script writes the sampled `subject_id` list, with seed and digest, to a local file outside Git. D6 must exclude these patients from the validation draw, so budgets are never tuned on validation patients.

**Measured per window, per counter:**

- Unbudgeted tokens per section and per `Domain` (all 14) and per `DecisionCategory` (all 7).
- Budgeted tokens per section and total.
- Records selected and omitted per section.
- Mandatory overflow failures.
- Core variable presence.
- Reference ID collisions.
- Render time, separately from `get_window()`.

**Reported per section and counter:** n, median, p95, p99, max tokens; selected and omitted records; windows hitting quota; mandatory failures; render time p50 and p95.

Committed reports (`docs/rendering_budget.md`) contain aggregates only: no patient text, no identifiers. They record command, seed, config digest, model digests and store version.

Budgets are accepted when measurement shows the policy is feasible on the sample. Residual mandatory overflows are documented; the runtime still fails closed on unseen windows.

## Acceptance Criteria

1. **Determinism.** Same window and config give byte-identical text, refs, omissions and token counts, independent of DataFrame row order.
2. **Fixed tokenizer set.** Constructing a `Renderer` with counters different from the configured set fails.
3. **Visibility.** Every rendered fact derives from the supplied snapshot or decision frames. Fixtures with planted future rows show none in the output.
4. **Decision completeness.** Every D2 decision appears exactly once with correct category, action, times and refs. All seven category subheadings render, empty ones included.
5. **Late documentation.** Late decisions show both times and the `(late)` mark; ordering follows the stated keys.
6. **Ownership.** No source record's full rendering appears twice.
7. **Pending and ongoing.** Pending tests show identity and `PENDING` only. Ongoing intervals show only D1's clipped state.
8. **Missingness.** The six states in the table render distinctly.
9. **Trends.** Synthetic series give the expected count, first, min, max and change. Mixed units split series. Null, non-numeric and pending values are excluded.
10. **References.** Every ID maps to the correct `SourceRow`(s); decision occurrences from one row get distinct IDs; trend contributors are complete; a forced collision raises.
11. **Relative time.** No shifted calendar date appears in the text.
12. **Escaping.** Adversarial free text in notes, comments and order text cannot produce a `===` line or change section structure.
13. **Notes.** A current-admission discharge note in the input raises. Shortened notes are labelled with kept sections.
14. **Selection under pressure.** Repeated runs select and omit the same records; omissions are reported per section.
15. **Budget.** The complete text fits the total under every counter; quota sections fit their quotas.
16. **Mandatory overflow.** Oversized mandatory content raises `RenderingBudgetError`; no partial output.
17. **Measurement.** 250 seeded real windows measured as specified; aggregates documented per counter, section, domain and category.
18. **Fixed budgets.** Final values in `config/rendering.toml`, justified in `docs/rendering_budget.md`, used by default.
19. **No leakage into logs.** Prompt text, note content and ref maps never appear in logs, exception messages or committed outputs.

## Testing Plan

| Layer | Tests |
|---|---|
| Unit | Relative time formatting, boundaries, multi-day offsets |
| Unit | Escaping: control characters, `===` injection, multi-line blocks |
| Unit | Ref ID derivation, stability, forced collision |
| Unit | Decision lines, ordering, late documentation, multi-decision rows |
| Unit | Ownership assignment, missingness labels |
| Unit | Trends: units, nulls, pending, minimum points |
| Unit | Quota filling, overflow trimming, mandatory refusal |
| Unit | Note shortening at section boundaries |
| Fixture integration | Synthetic D1/D2 windows with planted future, pending and truncated rows |
| Fixture integration | Fake counters with different token densities; identical text |
| Real data | Measurement script, `real_data` marker, `ARGUS_STORE_DIR` |

Each test states what it protects, what wrong implementation fails it, and what passing does not establish, following the ARGUS test convention. No real patient text in fixtures or reports.

## Implementation order

1. Config schema and validation; `Section`, ref types; `timefmt.py`, `escape.py`, `references.py`.
2. **Unbudgeted measurement:** render every record of every domain with minimal formatting and run the measurement on the real sample. This shows actual domain sizes before the selection policy is tuned.
3. Header and decision formatting.
4. Ownership, current state, trends.
5. Recent evidence, notes, past admissions.
6. Budget enforcement and coverage.
7. Unit and fixture tests throughout steps 1–6.
8. Budgeted measurement; finalise budgets and core variables; write `docs/rendering_budget.md`.

## Rules for the implementing agent

Follow the rules in `docs/spec_snapshot_engine.md` and `docs/spec_llm_backbone.md`, including test integrity, honest reporting of verification results and sensitive-data restrictions.

Additionally:

1. Do not modify D1 availability logic, D2 decision classification or D4 to simplify rendering.
2. No LLM calls, no clinical inference.
3. No tokenizer implementations outside D4.
4. No silent decision drops, suppressed overflow errors or ad hoc priority changes.
5. No token or latency figures that were not measured.
6. No instruction text in rendered output.

## Files Reference

| File | Change |
|---|---|
| `docs/spec_rendering.md` | This specification |
| `config/rendering.toml` | New: format, tokenizers, budgets, core variables, priorities |
| `src/argus/rendering/__init__.py` | New: public exports |
| `src/argus/rendering/config.py` | New: config loading and validation |
| `src/argus/rendering/renderer.py` | New: `Renderer`, assembly, token counting |
| `src/argus/rendering/timefmt.py` | New: relative time formatting |
| `src/argus/rendering/escape.py` | New: untrusted text escaping |
| `src/argus/rendering/references.py` | New: ref IDs and map |
| `src/argus/rendering/format.py` | New: record and decision lines |
| `src/argus/rendering/trends.py` | New: trend statistics |
| `src/argus/rendering/select.py` | New: ownership, priority, quotas |
| `scripts/measure_rendering.py` | New: real-window measurement |
| `tests/rendering/` | New: unit and fixture tests |
| `docs/rendering_budget.md` | New: measured aggregates, final budget rationale |
| `docs/spec_cohort.md` (D6, future) | Must exclude D5 measurement patients from validation draw |

`argus.data` and `argus.llm` public interfaces remain unchanged.

## Out of Scope

- Agent prompt framing and instructions (A1).
- Request-level preflight and output-token reserve (A1).
- Diagnosis, guideline interpretation, error detection.
- LLM summarisation or compression.
- Severity scores or derived risk scores.
- Memory across windows (A2).
- Collaboration messages and citation validation (D8).
- Cohort selection (D6).
- Reviewer interface.
- Embedding-based retrieval.

## Known Residual Risks

- **Incomplete documentation.** Missing records may be read as missing interventions. Labels mitigate; A1 framing and D7's "cannot determine" label address the rest.
- **Late charting.** Both times are shown; documentation absent at `at` cannot be recovered.
- **Budget information loss.** Omitted optional evidence may matter. Coverage reporting and A7 sensitivity arms address this.
- **Core variable choice.** A fixed list may miss variables relevant to some decisions; they still appear in recent evidence when budget allows.
- **Trend simplification.** First, last and extrema hide intermediate variation.
- **Note coverage.** MIMIC-IV-Note lacks progress and nursing notes.
- **Prompt injection.** Escaping prevents structural injection, not model compliance with embedded text.
- **Backend sensitivity.** Identical content does not guarantee identical interpretation.
- **Oversized windows.** Windows whose decisions exceed the total fail explicitly.

## Related

- `docs/roadmap.md`: D1, D2, D4, D5, D6, D8, A1, A2, A3, A7.
- `docs/spec_snapshot_engine.md`: visibility, redaction, `is_pending`, `is_truncated`.
- `docs/spec_decision_window.md`: categories, actions, temporal rules.
- `docs/spec_llm_backbone.md`: token counting, context enforcement.
- `docs/backends/gpu_budget.md`: measured context limits.
- `docs/literature_review.md`: order and quantity sensitivity.
