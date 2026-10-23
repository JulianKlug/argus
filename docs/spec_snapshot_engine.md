# Spec: Leak-free patient snapshot engine (MIMIC-IV)

## Context

ARGUS agents judge whether ICU care at time `t` was erroneous, delayed or omitted.
Any record not available to clinicians at `t` (a lab not yet resulted, a discharge
diagnosis, the end of an infusion) gives agents hindsight and invalidates the
validation study. Every agent consumes patient data through this engine, so it is
built first and its leak guarantees are tested before any agent exists.

## Current State

Verified 2026-10-23.

- Repo `argus`: no code (README, LICENSE, `.gitignore`, `local_docs/`).
- Source: MIMIC-IV 3.1 at `/mnt/hdd1/datasets/mimiciv_3.1/{hosp,icu,note}`, 18 GB csv.gz,
  `SHA256SUMS.txt` present. Disk 97% full (54 GB free).
- Target disk: `/mnt/data1` NVMe, 682 GB free.
- Tooling: `uv`, DuckDB and pyarrow installed; no Postgres server; system Python 3.9.

Measured `charttime` → `storetime` lag (first 300k rows per table):

| Table | storetime null | storetime < charttime | lag min p50 / p90 / p99 |
|---|---|---|---|
| labevents | 1.7% | 0% | 73 / 342 / 1293 |
| chartevents | 0.3% | 9.8% | 16 / 126 / 378 |
| inputevents | 0% | 5.2% | 6 / 290 / 1223 |
| outputevents | 0% | 14.3% | 17 / 101 / 306 |
| emar | 0% | 3.4% | 1 / 13 / 260 |
| radiology | 0% | ~0% | 131 / 977 / 6310 |
| discharge | 0% | 0.8% | 1120 / 6702 / 35403 |

Consequence: `charttime` filtering leaks lab results by hours; `storetime` alone
shows events before they happened in 10–14% of rows. Rule: `max(charttime, storetime)`.

## Proposed Change

```
 agents (out of scope)
        │  get_snapshot(subject_id, at)
        ▼
 ┌──────────────────────┐   public: SnapshotEngine, PatientSnapshot,
 │ engine.py            │   ClinicalRecord, Demographics, CurrentEncounter,
 │ assemble snapshot    │   Domain, errors
 └─────────┬────────────┘
           ▼
 ┌──────────────────────┐   per-table availability + redaction rules
 │ availability.py      │   (single source of truth for leak policy)
 └─────────┬────────────┘
           ▼
 ┌──────────────────────┐   DuckDB driver: rows of one table for one subject
 │ store.py (MimicStore)│
 └─────────┬────────────┘
           ▼
   Parquet, sorted by subject_id   ◄── scripts/build_store.py (one-time)
   /mnt/data1/klug/argus/mimiciv_3.1/
```

Each layer calls only the one below. Agents never see table names, DuckDB or paths.

### Decisions (locked)

| # | Decision |
|---|---|
| D1 | Row availability = `max(clinical_time, storetime)`; `storetime` null → `clinical_time` |
| D2/D6 | Input `subject_id`. **Current** = admission (`hadm_id`) with `admittime <= at < dischtime`. Split by time: `event_time >= current.admittime` → `current`, else `past` (handles null `hadm_id`) |
| D3 | Prior-admission discharge notes included when available; current-admission discharge notes hard-banned |
| D5 | Interval rows visible from `starttime`; if `endtime > at`: `endtime` null, amount/value prorated linearly, status null, `is_truncated = True` |
| D7 | `at` must fall inside an ICU stay (`intime <= at < outtime`), else `NotInIcuStayError` |
| D9 | One-time csv.gz → Parquet on NVMe, queried with DuckDB |
| D10 | Domain-organized snapshot; MIMIC specifics stay inside engine |
| D12 | Labs/cultures drawn but not resulted: row visible, result fields null, `is_pending = True` |
| — | Date-only columns available at `date + 24h` |
| — | Inclusive cutoff: `available_time <= at` |

### Implementation Details

#### Public API (`src/argus/data/__init__.py`)

```python
class SnapshotEngine:
    def __init__(self, store_dir: Path) -> None: ...
    def get_snapshot(self, subject_id: int, at: datetime) -> PatientSnapshot: ...

@dataclass(frozen=True)
class Demographics:
    sex: str
    age_at_t: int                  # anchor_age + (at.year - anchor_year)
    race: str | None
    language: str | None
    insurance: str | None
    marital_status: str | None

@dataclass(frozen=True)
class CurrentEncounter:
    hadm_id: int
    admittime: datetime
    admission_type: str
    admission_location: str | None
    edregtime: datetime | None
    stay_id: int
    icu_intime: datetime
    first_careunit: str

@dataclass(frozen=True)
class ClinicalRecord:
    def get(self, domain: Domain) -> pd.DataFrame: ...   # empty frame with schema if no rows

@dataclass(frozen=True)
class PatientSnapshot:
    subject_id: int
    at: datetime
    demographics: Demographics
    current_encounter: CurrentEncounter
    current: ClinicalRecord        # event_time >= current_encounter.admittime
    past: ClinicalRecord           # event_time <  current_encounter.admittime

class Domain(Enum):
    ENCOUNTERS, CHARTED_OBSERVATIONS, LABS, MICROBIOLOGY, ORDERS,
    MEDICATION_ORDERS, MEDICATION_ADMINISTRATIONS, INPUTS, INPUT_INGREDIENTS,
    OUTPUTS, PROCEDURES, DIAGNOSES, NOTES, OUTPATIENT_MEASUREMENTS

class SnapshotError(Exception): ...
class SubjectNotFoundError(SnapshotError): ...
class NotInIcuStayError(SnapshotError): ...
class AmbiguousEncounterError(SnapshotError): ...   # >1 admission contains at
```

`at`: naive `datetime` (MIMIC shifted time). Timezone-aware → `ValueError`.

#### Common columns (every domain DataFrame)

| Column | Type | Meaning |
|---|---|---|
| `source_table` | str | MIMIC table name (audit only) |
| `source_row` | int64 | `_row` from store (audit only) |
| `hadm_id` | Int64 | nullable |
| `stay_id` | Int64 | nullable |
| `event_time` | datetime64 | clinical time (charttime / starttime / ordertime / collection) |
| `available_time` | datetime64 | when row became visible; always `<= at` |
| `is_pending` | bool | result not yet available (labs, micro) |
| `is_truncated` | bool | interval clipped at `at` |

Plus `label` (joined from `d_items` / `d_labitems` / `d_icd_*`) and table-specific
columns below. `caregiver_id`, `provider_id`, `order_provider_id`, `enter_provider_id` dropped.

#### Availability rules (`availability.py`) — the core of this spec

`ct` = clinical time, `st` = storetime, `EOD(d)` = `d + 24h`.

| Table | Domain | Row visible when | Redacted at `at` |
|---|---|---|---|
| chartevents | CHARTED_OBSERVATIONS | `max(charttime, st) <= at` | — |
| datetimeevents | CHARTED_OBSERVATIONS | `max(charttime, st) <= at` | — |
| labevents | LABS | `charttime <= at` | if `max(charttime, st) > at`: `value, valuenum, valueuom, ref_range_*, flag, comments` null; `is_pending` |
| microbiologyevents | MICROBIOLOGY | collection `coalesce(charttime, EOD(chartdate)) <= at` | if `max(collection, coalesce(st, EOD(storedate))) > at` or both null: `org_*, isolate_num, quantity, ab_*, dilution_*, interpretation, comments` null; `is_pending`; pending rows deduplicated on `(micro_specimen_id, test_itemid)` |
| emar (+emar_detail) | MEDICATION_ADMINISTRATIONS | `max(charttime, st) <= at` | — |
| prescriptions | MEDICATION_ORDERS | `max(starttime, poe.ordertime) <= at` (no poe → starttime) | `stoptime > at` → null |
| pharmacy | MEDICATION_ORDERS | `coalesce(entertime, starttime) <= at` | `stoptime`, `expirationdate`, `verifiedtime` `> at` → null; `status` null unless `stoptime <= at` |
| poe (+poe_detail) | ORDERS | `ordertime <= at` | `order_status` dropped; `discontinued_by_poe_id` null if that order's `ordertime > at` |
| inputevents | INPUTS | `starttime <= at` | if `endtime > at`: `endtime, statusdescription` null; `amount *= (at-start)/(end-start)`; `is_truncated` |
| ingredientevents | INPUT_INGREDIENTS | `starttime <= at` | as inputevents |
| procedureevents | PROCEDURES | `starttime <= at` | if `endtime > at`: `endtime, statusdescription` null; `value *= (at-start)/(end-start)`; `is_truncated` |
| outputevents | OUTPUTS | `max(charttime, st) <= at` | — |
| radiology (+detail) | NOTES | `max(charttime, st) <= at` | — |
| discharge (+detail) | NOTES | `max(charttime, st) <= at` **and** `hadm_id != current.hadm_id` | — |
| omr | OUTPATIENT_MEASUREMENTS | `EOD(chartdate) <= at` | — |
| diagnoses_icd | DIAGNOSES | owning admission `dischtime <= at` | `event_time = dischtime` |
| drgcodes | DIAGNOSES | owning admission `dischtime <= at` | `event_time = dischtime` |
| procedures_icd | PROCEDURES | owning admission `dischtime <= at` | coded post-discharge |
| hcpcsevents | PROCEDURES | owning admission `dischtime <= at` | billing, post-discharge |
| admissions | ENCOUNTERS + header | `admittime <= at` | current admission: `dischtime, deathtime, discharge_location, hospital_expire_flag` null; `edouttime > at` → null |
| icustays | ENCOUNTERS + header | `intime <= at` | `outtime > at` → `outtime, los, last_careunit` null |
| transfers | ENCOUNTERS | `intime <= at` | `outtime > at` → null |
| services | ENCOUNTERS | `transfertime <= at` | — |
| patients | Demographics | — | `dod`, `anchor_year_group` dropped; `age_at_t` computed |

Not exposed: `provider`, `caregiver` (ID lists); `d_*` used only for labels.

#### Store build (`scripts/build_store.py`)

```
uv run python scripts/build_store.py \
    --source /mnt/hdd1/datasets/mimiciv_3.1 \
    --dest   /mnt/data1/klug/argus/mimiciv_3.1
```

1. Verify every source file against `SHA256SUMS.txt`; abort on mismatch.
2. Per table: DuckDB `COPY (SELECT row_number() OVER () - 1 AS _row, * FROM read_csv(...) ORDER BY subject_id) TO '<dest>/<module>/<table>.parquet'`, timestamps typed as `TIMESTAMP`.
3. Write `<dest>/manifest.json`: source checksum, source row count, parquet row count per table.
4. Idempotent: skip a table whose manifest entry matches.

`_row` = 0-based line index in source CSV; stable across rebuilds.

#### Project setup

`pyproject.toml` (uv, Python 3.12): `duckdb`, `pyarrow`, `pandas`; dev: `pytest`, `hypothesis`.
Store location from `ARGUS_STORE_DIR` env var; tests needing real data are marked
`@pytest.mark.real_data` and skipped when unset.

## Acceptance Criteria

1. `build_store.py` aborts with non-zero exit on any SHA256 mismatch.
2. `manifest.json` parquet row count equals source row count for all 35 source files (hosp 22, icu 9, note 4).
3. Unknown `subject_id` → `SubjectNotFoundError`; `at` outside any ICU stay → `NotInIcuStayError`; tz-aware `at` → `ValueError`.
4. Fixture: every table in the rules table has a sentinel row with `available_time = at + 1 min`; zero sentinels returned.
5. Fixture: row with `available_time == at` is returned.
6. Fixture: lab drawn `at - 30 min`, stored `at + 30 min` → returned, `is_pending`, `valuenum` null; at `at + 31 min` → `valuenum` set, not pending.
7. Fixture: infusion 10:00–16:00, amount 600, `at` = 12:00 → `amount == 200`, `endtime` null, `statusdescription` null, `is_truncated`.
8. Fixture: for the current admission, `dischtime`, `deathtime`, `discharge_location`, `hospital_expire_flag`, current icustay `outtime`/`los`/`last_careunit` are null; no `dod` anywhere; zero `diagnoses_icd`, `drgcodes`, `procedures_icd`, `hcpcsevents`, discharge-note rows with current `hadm_id`.
9. Fixture: discharge note of current admission with planted `storetime < at` is still not returned.
10. Real data: 200 seeded-random `(subject_id, at)` inside ICU stays → every returned row has `available_time <= at` and `event_time <= at`.
11. Real data monotonicity: 50 subjects × 5 increasing `at` → `(source_table, source_row)` set at `at_i` ⊆ set at `at_{i+1}`; non-pending, non-truncated rows have identical values.
12. Real data: every `current` row `event_time >= current_encounter.admittime`; every `past` row `<`.
13. Real data: p95 `get_snapshot` latency ≤ 2 s over the 200-sample run (warm cache), reported with the command used.
14. `argus.data.__all__` lists only the public names in this spec; all other modules/functions are `_`-prefixed or unexported.
15. Each test states its card: `protects=…; fails_when=…; no_claim=…`.
16. No degradation of existing functionality (none exists).

## Testing Plan

| Layer | What | Count |
|---|---|---|
| Unit | `availability.py` rule per table: visible / not visible / boundary / redaction | +25 |
| Unit | proration, pending dedup, EOD date rule, age_at_t | +6 |
| Integration (fixture store) | criteria 3–9 via `SnapshotEngine` on synthetic Parquet | +9 |
| Integration (real store) | criteria 10–13, `real_data` marker | +4 |
| Build | checksum abort, row-count manifest, idempotency | +3 |

## Rules for the implementing agent

1. Acceptance criteria are the `protects / fails_when / no_claim` profile of each test: the observable it protects, the wrong result it rejects, and what green does not prove. Never "make the tests pass".
2. Real code and real tests land in the same change; a placeholder, a stub assertion or a test deferred to a follow-up is not done.
3. Any change to a test, validator, CI step, timeout, tolerance, threshold, snapshot or suppression pragma is reported on its own line in your summary with its justification, before any other result.
4. A reported command names the command and its observed result (exit status, counts, the failing line); a command you did not run is not reported, and stderr is never silenced in a command you cite.
5. Your report is a claim until the parent re-executes it: say what you verified, what you reused, and what you could not check.
6. Leak policy lives only in `availability.py`. No other module filters on time.

## Rollback Plan

New code only; revert the PR. Store is derived: `rm -rf /mnt/data1/klug/argus/mimiciv_3.1`.
Source CSVs are read-only and never modified.

## Effort Estimate

| Component | Human | CC |
|---|---|---|
| Project setup + store build script + manifest | 4 h | 20 min |
| Store build run (one-time, chartevents dominates) | ~1 h wall | ~1 h wall |
| `store.py` driver | 2 h | 10 min |
| `availability.py` (25 table rules) | 8 h | 45 min |
| `engine.py` + dataclasses + errors | 4 h | 20 min |
| Fixture store builder + fixture tests | 6 h | 40 min |
| Real-data property, monotonicity, latency tests | 3 h | 20 min |
| **Total** | **~3.5 days** | **~3.5 h incl. build** |

## Files Reference

| File | Change |
|---|---|
| `pyproject.toml` | New: uv project, deps, pytest markers |
| `scripts/build_store.py` | New: checksum, csv.gz → Parquet, manifest |
| `src/argus/data/__init__.py` | New: public exports |
| `src/argus/data/engine.py` | New: `SnapshotEngine` |
| `src/argus/data/snapshot.py` | New: `PatientSnapshot`, `ClinicalRecord`, `Demographics`, `CurrentEncounter` |
| `src/argus/data/domains.py` | New: `Domain` enum, column schemas, table → domain map |
| `src/argus/data/availability.py` | New: per-table rules |
| `src/argus/data/store.py` | New: DuckDB driver |
| `src/argus/data/errors.py` | New: exceptions |
| `tests/data/conftest.py` | New: fixture store builder with sentinel rows |
| `tests/data/test_availability.py` | New |
| `tests/data/test_engine_fixture.py` | New |
| `tests/data/test_engine_real.py` | New (`real_data`) |
| `tests/data/test_build_store.py` | New |

## Out of Scope

- Rendering snapshots to LLM prompt text.
- Derived features: trends, scores (SOFA), unit normalization.
- Cohort and window sampling (which patients, which `at`).
- Real-time / streaming ingestion.
- Other datasets (eICU, HiRID); the domain contract allows them later.
- MIMIC-III.

## Known Residual Risks

- Retrospective ICD coding timing unknown: prior-admission codes assumed available at that admission's `dischtime`.
- `storetime` null (1.7% of labs) falls back to `charttime`: may expose a result early.
- Linear proration hides titration inside one infusion row.
- MIMIC-IV notes lack nursing/physician progress notes; text input is radiology + prior discharge summaries only.

## Related

- `local_docs/gist.md`, `local_docs/literature_review.md` §8 point 3.
