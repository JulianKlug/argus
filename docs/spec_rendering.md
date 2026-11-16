# Spec: Window rendering to prompt text (D5)

## Context

ARGUS agents evaluate clinical decisions within sequential 6 h ICU windows. D1 provides a leak-free patient snapshot, D2 separates decisions from the available clinical evidence, and D4 provides provider-independent token counting and LLM inference.

D5 converts a `DecisionWindow` into deterministic, clinically readable prompt text that fits the configured models' context budgets.

The renderer must preserve the distinction between **decisions**, **observed evidence**, and **missing or unavailable information**. It must not introduce hindsight information, silently discard decisions, or make clinical interpretations.

The ordering and quantity of information are important experimental variables. Hager et al. (2024), cited in the literature review, report model sensitivity to information order and quantity. D5 therefore fixes rendering order, selection rules and budgets before agent development.

The roadmap exit criterion is:

> Token count per domain measured on real windows per configured tokenizer; budget fixed for trends, notes and past admissions.

D5 must satisfy this criterion while providing an auditable input representation for A1 through A7 and the retrospective validation study.

## Design

```
SnapshotEngine.get_window(...)
         |
         v
    DecisionWindow
         |
         v
  rendering/
    format.py       Clinical formatting
    trends.py       Deterministic trends
    select.py       Record selection and budgets
    references.py   Evidence references
    renderer.py     Assembly and token counting
         |
         v
    RenderedWindow
      |       |
      |       +-- Evidence reference map
      |       +-- Token usage by section
      |       +-- Selection and omission report
      |
      v
  Prompt text
      |
      v
  A1 / A2 / A3 / A5
      |
      v
  D4 LlmClient
```

D5 consumes only data exposed through the D1 and D2 public interfaces. It must not query the MIMIC store directly or implement its own time visibility rules.

D5 does not call an LLM. All rendering, selection and trend extraction are deterministic.

### Responsibilities

| Component | Responsibility |
|---|---|
| D1 | Clinical data visibility, redaction and snapshot assembly |
| D2 | Identification and classification of decisions |
| D4 | Token counting, model capabilities and request context enforcement |
| D5 | Clinical text representation, record selection and evidence budgets |
| D8 | Agent communication, role access and evidence exchange |
| A1 | Prompt assembly, run logging, execution and replay |
| A2–A5 | Clinical interpretation, reasoning and output |

D5 creates a clinical context block, not a complete agent prompt. Agent instructions, output schemas and collaboration messages are provided by later components.

## Public API

New package: `src/argus/rendering/`.

```python
@dataclass(frozen=True)
class RenderBudget:
    max_tokens: int
    trend_tokens: int
    notes_tokens: int
    past_tokens: int

@dataclass(frozen=True)
class EvidenceRef:
    source_table: str
    source_row: int

@dataclass(frozen=True)
class RenderedWindow:
    text: str
    evidence_refs: Mapping[str, EvidenceRef]
    section_tokens: Mapping[str, Mapping[str, int]]
    omissions: Mapping[str, int]
    total_tokens: Mapping[str, int]
    content_hash: str

def render_window(
    window: DecisionWindow,
    *,
    counters: Mapping[str, Callable[[str], int]],
    budget: RenderBudget,
) -> RenderedWindow:
    ...
```

`counters` maps backend identifiers to the existing `LlmClient.count_text_tokens` method.

The same rendered text must be usable across backends. When multiple counters are supplied, selection must fit the strictest token count.

`section_tokens` reports counts for every configured tokenizer. `total_tokens` counts the complete rendered text independently rather than assuming that section token counts are additive.

`content_hash` is a SHA256 digest of the final UTF-8 text, allowing A1 to verify the exact clinical input used during a run.

A dedicated `RenderingBudgetError` is raised when mandatory content cannot fit. No partial result is returned as though it were complete.

## Rendering structure

The renderer always uses the following section order:

| Order | Section | Purpose |
|---|---|---|
| 1 | Window header | Decision time, observation boundary, demographics, current encounter |
| 2 | Clinical decisions | All D2 decisions within the window |
| 3 | Current clinical state | Latest available observations, medications, devices and relevant investigations |
| 4 | Recent clinical evidence | Relevant visible records from the current admission |
| 5 | Trends | Deterministic longitudinal summaries |
| 6 | Notes | Available radiology reports and previous discharge notes |
| 7 | Past admissions | Prior encounter and diagnostic history |
| 8 | Coverage statement | Omitted counts, truncated content and missing information |

The structure and section order are fixed by the rendering configuration. Empty sections are identified explicitly.

### 1. Window header

The header contains:

- Decision window start, exclusive, and end, inclusive.
- Patient age at the decision time and sex.
- Current admission type and ICU admission time.
- Current care unit information available through D1.
- `care_years` range for historical guideline matching.
- A statement that the clinical record is retrospective and deidentified.
- A statement that only information available at the window end may be considered.

MIMIC shifted calendar dates must not be presented as actual historical treatment dates. The `care_years` range is the appropriate metadata for guideline era selection.

Current encounter identifiers may be retained in the audit metadata but are not required in the prompt text.

### 2. Clinical decisions

All seven `DecisionCategory` values are rendered, regardless of the future evaluation configuration:

1. `MEDICATION`
2. `DEVICE_SETTING`
3. `TEST_ORDER`
4. `PROCEDURE`
5. `CONSULT`
6. `PATIENT_MOVEMENT`
7. `EQUIPMENT_REMOVAL`

Each decision must retain:

- A unique decision reference.
- Decision category and action.
- Human-readable clinical label and subcategory.
- Clinical decision time (`event_time`).
- Time first available (`available_time`), where different.
- Relevant dose, rate, quantity, route, unit or device setting when present.
- The linked order reference where D2 supplies one.
- The underlying source evidence reference.

The valid actions are `ORDER`, `START`, `CHANGE`, `CONTINUE` and `STOP`.

Actions are taken directly from D2 and must not be inferred again in D5.

Decisions are sorted chronologically by decision time. Ties use the fixed category order, action and source identity. The tie-breaking rule must be deterministic.

**Late documentation:** A decision with `event_time` preceding the current window may still belong to that window if it first became visible during the window. Preserve both timestamps and identify the difference without moving the decision into an earlier window.

**Duplicates:** A single source row may produce multiple decisions, such as a procedure start and stop. Each decision occurrence receives a distinct decision reference while retaining the same underlying evidence reference.

**Completeness:** All decisions are mandatory content. The renderer may use compact formatting but must not silently discard a decision to satisfy the token budget.

### 3. Current clinical state

Represent the latest available clinical information relevant to interpreting the decisions, including:

- Vital signs and charted observations.
- Available laboratory and microbiology results.
- Medication orders and administrations.
- Infusions, fluids, inputs and outputs.
- Device settings and procedures.
- Relevant current encounter and diagnostic information.

Use descriptive labels rather than MIMIC item identifiers.

Preserve units, qualifiers and time information. Numeric values must not be converted unless an explicitly validated conversion rule exists.

Missing values, pending tests and unfinished intervals require distinct representations:

| State | Rendering |
|---|---|
| Observed value | Value, unit, clinical time and evidence reference |
| Pending result | `PENDING`; no result value |
| Missing value | `NOT RECORDED` |
| Ongoing interval | `ONGOING AT WINDOW END` |
| No visible records | `NO RECORDS AVAILABLE` |
| Content omitted by budget | `RECORDS OMITTED BY RENDERING BUDGET` |

A missing record must never be rendered as evidence that the clinical action was not performed.

For ongoing infusions and procedures, D1 has already redacted future end times and adjusted interval values. D5 must present the visible state without reconstructing the original interval or future status.

Pending laboratory and microbiology records may identify the ordered or collected test, but must never expose withheld values, organisms, interpretations or susceptibility results.

### 4. Recent clinical evidence

Current admission data may cover substantially more than one 6 h window. Rendering the complete admission for every window would produce large, repetitive prompts.

D5 therefore applies a fixed selection policy:

- Include visible records belonging to the decision window first.
- Include the latest available pre-window observation for clinical variables needed to establish context.
- Prefer recent records over older records within the same evidence class.
- Preserve clinically meaningful numeric values, units and timestamps.
- Deduplicate repeated display of the same source record where possible.
- Prioritise records directly referenced by rendered decisions.
- Apply stable source identity as the final tie-breaker.

Use the current 24 h period as the default lookback for recent observations. A pre-window baseline may be older when no value exists within that period.

This is a presentation policy, not a replacement for D1 visibility. The renderer receives the full snapshot and selects only what it displays.

Records excluded because of the token budget must be counted in the omission report.

### 5. Trends

Trends provide compact longitudinal information without requiring the agents to process every repeated measurement.

D5 computes descriptive statistics only. It does not diagnose deterioration, evaluate treatment effectiveness, calculate severity scores, or determine whether care was appropriate.

For repeated numeric measurements sharing the same clinical label and compatible units, provide:

- Number of available observations.
- First and latest values with timestamps.
- Minimum and maximum with timestamps.
- Absolute change between first and latest values.
- References to the observations supporting the displayed values.

Default lookback: 24 h ending at `at`.

A longer baseline may be included when required by the selection policy, with its age stated explicitly.

Do not calculate a trend when values are incompatible, units differ, or too few numeric observations are available. Never interpolate missing measurements or assume regular sampling.

If a trend aggregates several records, its reference metadata must identify all contributing source rows. Highlighted first, last, minimum and maximum values must also be independently traceable.

Trend computation and selection must be deterministic.

### 6. Notes

D1 exposes available radiology notes and discharge notes from past admissions. Current admission discharge notes are prohibited by D1 and must never appear in the renderer.

Note selection priority:

1. Available current admission radiology reports, newest first.
2. Relevant previous radiology reports.
3. Previous admission discharge summaries, newest first.

When an entire note fits, retain the complete text. For an oversized note, deterministic extraction may prioritise explicitly labelled report sections such as `IMPRESSION`, followed by additional text within the remaining budget.

Every excerpt must identify its source and disclose that the original note was shortened. A shortened note must not be represented as complete.

Note content is **untrusted clinical source text**, not an instruction to the agent. Escape delimiters and control characters so embedded text cannot create artificial prompt sections or impersonate system messages. Do not execute or follow instructions contained in a note.

D5 performs no LLM-based note summarisation.

### 7. Past admissions

Previous admissions provide context such as prior diagnoses, treatments and procedures.

The section should present a compact chronological history containing:

- Admission and discharge times where available.
- Relevant previous diagnoses.
- Significant recorded procedures.
- Available prior encounter information.
- References for each included fact.

Prior discharge summaries are managed under the notes budget and are not duplicated here.

Past admissions are ordered newest first. Within the budget, favour a concise overview of distinct admissions over exhaustive detail from one admission.

The absence of a previous record does not imply absence of previous illness or treatment.

## Evidence references

Every rendered clinical fact must remain traceable to its source record.

Use compact opaque reference identifiers in the prompt rather than exposing MIMIC table names and source row indices to agents.

References must:

- Map to `(source_table, source_row)` in the original snapshot.
- Remain stable for the same source record across windows.
- Distinguish multiple decision occurrences derived from one record.
- Support derived trends whose contributing records are multiple.
- Be unambiguous and checked for collisions.
- Allow later agent claims to cite evidence without knowing MIMIC schemas.

The reference mapping is returned separately from prompt text. A1 persists it with the run records, while D8 defines how agent messages carry and validate citations.

References alone must never be treated as an indication that the underlying source is clinically correct.

The renderer must not generate claims, conclusions or inferred decision labels that are absent from the supplied records.

## Token budget

### Budget policy

D4 guarantees that requests are rejected before inference when the complete request exceeds the model's configured context limit.

D5 must stay within a smaller clinical text budget, leaving space for system instructions, output schemas, tool definitions, collaboration messages and model output.

Initial configuration proposed for real-window measurement:

| Component | Initial token budget |
|---|---:|
| Complete rendered clinical text | 16,000 |
| Trends, maximum | 2,500 |
| Notes, maximum | 3,000 |
| Past admissions, maximum | 1,500 |

The remaining budget is shared by the header, all decisions and current clinical evidence.

These values are **initial experimental limits, not measured results**. D5 must measure representative real windows and revise or confirm them before completion. The final values are fixed in version-controlled `config/rendering.toml`.

The 16,000-token ceiling leaves room within the smallest currently configured local context of 32,768 tokens. It is not, on its own, proof that a complete agent request fits.

### Budget enforcement

1. Determine the active tokenizer or comparison set of tokenizers.
2. Render mandatory content.
3. If mandatory content exceeds the total limit, raise `RenderingBudgetError`.
4. Select optional evidence using fixed priority and section quotas.
5. Assemble all sections and the coverage statement.
6. Measure the final complete text with each supplied tokenizer.
7. If over budget, remove the lowest-priority optional record and repeat.
8. Return text, token counts, references and the complete omission report.

No mandatory decision may be removed.

The renderer must not truncate text at an arbitrary character or token position. Record selection and any permitted note excerpting must respect explicit boundaries.

`count_text_tokens()` measures the rendered clinical text. Before actual inference, A1 must additionally call D4's `count_request_tokens()` on the complete request, including schema, tools and conversation history, and reserve the configured output tokens.

Any resulting context overflow must be reported, never handled by silently changing the content.

### Comparable backend inputs

A7 compares different LLM backends and collaboration strategies. Tokenization differences must not automatically create different patient evidence for otherwise matched experimental arms.

When matched inputs are required, D5 renders one canonical text using the strictest supplied token counter. The exact same text is then used by each backend.

Any experiment intentionally comparing different content budgets or ordering must record the rendering configuration and be identified as a separate experimental condition.

## Configuration

Add `config/rendering.toml`, with:

- Rendering format version.
- Total clinical text budget.
- Trends, notes and past admission budgets.
- Recent evidence lookback.
- Trend lookback.
- Selection priorities and tie-breaking policy.
- Note excerpt policy.
- Section inclusion and ordering.

Configuration must be validated strictly. Unknown keys, negative budgets and inconsistent limits cause an explicit error.

No clinical data, credentials or model prompts are committed to this file.

Changing the rendering configuration changes the experimental input and must be recorded in A1 and frozen in V1.

## Real-window measurement

Add `scripts/measure_rendering.py`.

It uses `SnapshotEngine.get_window()` and the existing D4 token-counting interface. No model inference is required.

Measurement must include:

- At least 200 seeded random valid ICU windows from the real store.
- An additional 50 high-volume windows selected by a documented record-volume criterion.
- Every configured local backend tokenizer.
- Unbudgeted section sizes, where feasible, and final budgeted sizes.
- Counts and fractions of selected and omitted records.
- Mandatory-content overflow and total budget failures.
- The final full-text token count.
- Render time.

Report per section and backend:

| Measure | Required |
|---|---|
| Number of windows | Yes |
| Median tokens | Yes |
| p95 tokens | Yes |
| p99 tokens | Yes |
| Maximum tokens | Yes |
| Number of records selected | Yes |
| Number of records omitted | Yes |
| Windows exceeding section quota | Yes |
| Windows failing mandatory budget | Yes |

Measurements must include distributions for the seven decision categories and the thirteen snapshot domains, even when some domains contribute no text in the final format.

Use a fixed seed and publish the command, configuration digest, model digests, store version and aggregate results.

Real patient text and record identifiers must not appear in measurement reports committed to Git.

The final budgets are accepted only after measurement demonstrates that the selected policy is feasible on the sampled windows. All remaining overflows must be documented, and the runtime must still fail closed on an unseen oversized window.

## Acceptance Criteria

1. **Determinism.** The same `DecisionWindow`, tokenizer set and rendering configuration produce byte-identical text, reference maps, omissions and token counts, independent of DataFrame input row order.

2. **Visibility.** Every rendered clinical fact derives exclusively from the supplied D1 snapshot or D2 decision frames. No future evidence or independently queried store records enter the output.

3. **Decision completeness.** Every D2 decision appears exactly once as a decision occurrence, with its correct category, action, time and underlying reference. All seven categories are represented, including empty categories.

4. **Temporal correctness.** Late-recorded decisions preserve their clinical and availability times. Decision-time ordering is stable. Window boundaries remain those defined by D2.

5. **Pending results.** Pending labs and microbiology display test identity and pending status without any withheld result value or interpretation.

6. **Ongoing intervals.** Running infusions and procedures display only the clipped information available at `at`; future end times, status and total amounts are never reconstructed.

7. **Missingness.** Pending, absent, omitted and ongoing records have distinct representations. An absent record is never described as an action not performed.

8. **Reference integrity.** Every cited record maps back to the correct source row; repeated source rows and multi-action decisions are handled; trend contributor references can be reconstructed.

9. **Trend correctness.** Synthetic measurement series produce the expected count, first, last, minimum, maximum and change. Incompatible units, missing values and pending results never enter numeric calculations.

10. **Notes.** Current admission discharge notes never appear. Shortened notes are marked, text is escaped, and adversarial note content cannot modify the renderer's section structure.

11. **Selection reproducibility.** Budget pressure leads to the same selected records and omissions on repeated runs. Any excluded content is disclosed by section.

12. **Token accounting.** Section and complete-text counts are reported for every configured tokenizer. The complete rendered text respects all supplied tokenizer limits.

13. **Mandatory overflow.** If mandatory content exceeds the budget, rendering fails explicitly. No decision is silently dropped or replaced with an incomplete list.

14. **D4 compatibility.** Rendered output can be inserted into a D4 `Request`. Full request pre-counting enforces the configured context and output-token reserve without changing D4.

15. **Backend comparability.** Given multiple tokenizer counters, the renderer produces identical clinical text for every backend in a matched comparison, fitting the strictest token budget.

16. **Real-data measurement.** At least 250 seeded real windows are measured, including 50 high-volume windows. Token distributions and omission statistics are documented for each configured tokenizer and domain.

17. **Fixed budgets.** The final trends, notes and past admission limits are written to `config/rendering.toml`, justified by measured data, and used unchanged by default.

18. **Security.** No sensitive prompt text, note excerpts or evidence maps enter ordinary logs, exception messages, committed test outputs or public reports.

19. **Regression.** D1, D2 and D4 existing tests remain green. No snapshot visibility, decision classification or LLM backend behaviour changes are required.

20. **Performance.** Rendering latency is measured separately from `get_window()` and LLM inference. p50 and p95 results are reported for real windows; substantial overhead must be investigated before D5 is accepted.

## Testing Plan

| Layer | Tests |
|---|---|
| Unit | Formatting, timestamps, missing values, units, stable ordering |
| Unit | Decision completeness, duplicate source rows, start/stop actions |
| Unit | Trends, null values, incompatible units, boundary measurements |
| Unit | Evidence reference generation and reconstruction |
| Unit | Token quotas, selection priorities, overflow refusal |
| Unit | Notes, escaping, excerpt boundaries, malicious instructions |
| Fixture integration | D1/D2 synthetic windows with planted future, pending and truncated data |
| D4 integration | Multiple fake tokenizer counters and full request preflight |
| Real data | Seeded window measurement and latency, `real_data` marker |

Every test states what behaviour it protects, what incorrect implementation would cause it to fail, and what a passing result does not establish, following the existing ARGUS test convention.

Tests using the real MIMIC store require `ARGUS_STORE_DIR`. No real patient text is stored in repository fixtures or test reports.

## Implementation order

1. Define public rendering types, strict configuration and evidence reference mapping.
2. Implement deterministic formatting for headers and all decision categories.
3. Add current clinical evidence formatting and explicit missingness handling.
4. Implement trend extraction, notes and past admission selection.
5. Implement budgets, omission reporting and multi-tokenizer measurement.
6. Add unit tests and synthetic integration tests.
7. Measure real windows, finalise budgets and document observed results.

## Rules for the implementing agent

Follow the existing rules in `docs/spec_snapshot_engine.md` and `docs/spec_llm_backbone.md`, including test integrity, reporting of verification results and restrictions on sensitive clinical data.

Additionally:

1. Do not modify D1 availability logic or D2 decision classification to simplify rendering.
2. Do not introduce an LLM call or a clinical inference step into D5.
3. Do not bypass D4 token counting or duplicate tokenizer implementations.
4. Do not silently drop decisions, suppress overflow errors or change fixed selection priorities.
5. Do not report token measurements or runtime performance that have not actually been measured.
6. Preserve clear separation between clinical data, rendering metadata and later agent instructions.

## Files Reference

| File | Change |
|---|---|
| `docs/spec_rendering.md` | New: this specification |
| `config/rendering.toml` | New: rendering format and fixed budgets |
| `src/argus/rendering/__init__.py` | New: public exports |
| `src/argus/rendering/renderer.py` | New: main rendering interface |
| `src/argus/rendering/format.py` | New: clinical record formatting |
| `src/argus/rendering/trends.py` | New: deterministic trend extraction |
| `src/argus/rendering/select.py` | New: selection and budget enforcement |
| `src/argus/rendering/references.py` | New: evidence reference mapping |
| `scripts/measure_rendering.py` | New: real-window token measurement |
| `tests/rendering/` | New: unit and integration tests |
| `docs/rendering_budget.md` | New: measured counts and final budget rationale |

Existing `argus.data` and `argus.llm` public interfaces should remain unchanged.

## Out of Scope

- Agent-specific clinical reasoning instructions.
- Automated diagnosis, guideline interpretation or error detection.
- LLM-based summarisation or compression.
- Clinical severity scoring or derived physiological risk scores.
- Agent memory across sequential windows.
- Agent tool execution and collaboration message handling.
- Selection of the validation cohort.
- Retrospective reviewer interface.
- Model fine-tuning or embedding-based retrieval.

## Known Residual Risks

- **Incomplete source documentation.** A missing clinical record can be mistaken by downstream agents for a missing intervention. Explicit missingness labels mitigate but do not eliminate this risk.
- **Late charting.** A decision may become visible after its clinical time. D5 preserves both timestamps but cannot recover unavailable documentation.
- **Information loss through budgets.** Optional evidence omitted by selection may matter clinically. Omission reporting and A7 sensitivity experiments address this risk.
- **Trend simplification.** First, last and extrema may hide important intermediate variation.
- **Note limitations.** MIMIC-IV-Note does not provide the full set of physician and nursing progress notes.
- **Prompt injection.** Clinical notes may contain adversarial instructions. Escaping and untrusted-source labelling reduce structural risks but do not guarantee model compliance.
- **Backend sensitivity.** Equal clinical content does not guarantee equal interpretation across models.
- **Oversized windows.** Some windows may contain more mandatory decisions than the allowed budget. These must fail explicitly rather than produce incomplete evidence.

## Related

- `docs/roadmap.md`: D1, D2, D4, D5, D8, A1, A2, A3 and A7.
- `docs/spec_snapshot_engine.md`: visibility and redaction.
- `docs/spec_decision_window.md`: decision categories, actions and temporal rules.
- `docs/spec_llm_backbone.md`: token counting and context enforcement.
- `docs/backends/gpu_budget.md`: measured local model context limits.
- `docs/literature_review.md`: information order and quantity sensitivity.
