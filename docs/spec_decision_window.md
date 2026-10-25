# Spec: Decision window — clinician decisions in (at − X, at]

## Context

Agents judge whether decisions taken in a retrospective window (default 6 h) were
erroneous, and whether something was omitted. They need the decisions of the window
separated from the evidence:

- **Medications & fluids**: ordered, started, titrated, stopped; dose / rate.
- **Device settings**: ventilator, dialysis/CRRT, ECMO, VAD, IABP, Impella, pacemaker.
  First value in the window and every change (keeping a setting is a decision).
- **Test orders**: labs, microbiology, imaging, ECG/echo, monitoring. Never results.
- **Procedures**: lines, airway (intubation, extubation, tracheostomy), drains, restraints, ...
- **Consults** and **patient movements** (admission, transfer, discharge orders).
- **Equipment removals**: lines, catheters, drains, pacer wires (planned or for a complication;
  accidental removals are not decisions).

All categories are always returned; which ones are evaluated is configured in the evaluation
layer, not here.

Omission judgments are only as good as coverage: an unmapped item reads as "not done".
Hence an explicit catalog, clinician-reviewed, with a coverage test.

## Design

```
 agents
   │ get_window(subject_id, at, window=6h)          get_snapshot(...) unchanged
   ▼
 engine.py ──► availability.Timeline   visible rows at `at` (leak policy, unchanged)
     │
     └──────► decisions.py   pure: visible frames + window start + catalog → decision frames
                   ▲
                   └── catalog.py ◄── decision_catalog.csv   (MIMIC keys → category)
```

- `decisions.py` receives only rows already visible at `at`; it cannot leak. It compares
  times with the window start only.
- Snapshot spec rule 6 amended: leak policy (`at`) only in `availability.py`; window logic
  only in `decisions.py`.
- MIMIC itemids, order types and drug strings never leave `argus.data` (D10).

### Public API additions

```python
class DecisionCategory(Enum): MEDICATION, DEVICE_SETTING, TEST_ORDER, PROCEDURE, CONSULT, PATIENT_MOVEMENT,
                              EQUIPMENT_REMOVAL
class DecisionAction(Enum):   ORDER, START, CHANGE, CONTINUE, STOP

@dataclass(frozen=True)
class DecisionWindow:
    snapshot: PatientSnapshot      # evidence at `at`; also contains the decision rows
    start: datetime                # at - window, exclusive
    def get(self, category: DecisionCategory) -> pd.DataFrame

SnapshotEngine.get_window(subject_id, at, window=timedelta(hours=6)) -> DecisionWindow
```

Decision frame: common columns (`event_time` = decision time; `(source_table, source_row)`
joins back to the snapshot) + `action`, `subcategory`, `order_table`, `order_row` + the
source row's own columns. One row per decision. `window <= 0` → `ValueError`.

### Rules

In window iff `start < event_time <= at`.

| Category | Source | Decision |
|---|---|---|
| MEDICATION | prescriptions, `drug_type != BASE` | ORDER at `coalesce(poe.ordertime, starttime)` |
| MEDICATION | inputevents main component (diluents skipped), chained per `(stay_id, itemid, ordercategoryname)` | segment starting > 10 min after the previous ended START; continuing segment with different rate CHANGE; `statusdescription` Stopped/Paused STOP at `endtime` |
| MEDICATION | poe Medications, `D/C` or `H` | STOP; label = discontinued drug |
| any | poe other types per catalog `(order_type, order_subtype)` (Lab → TEST_ORDER at order time, Consults → CONSULT, ADT → PATIENT_MOVEMENT) | New ORDER, Change CHANGE, D/C or H STOP |
| any (settings, PCA/epidural/inhaled meds, restraints) | chartevents items per catalog | per `(stay_id, itemid)`: first in window START (no prior value in stay), CONTINUE (equal), CHANGE (different); later in-window change CHANGE |
| any (ventilation, dialysis, lines, airway, procedures) | procedureevents per catalog | START at `starttime`; STOP at visible `endtime` (> 1 min) for DEVICE_SETTING / MEDICATION intervals |
| EQUIPMENT_REMOVAL | chartevents "<line> Discontinued" / "Reason Discontinued" | STOP each charting in window; value `Unplanned…` (accident) skipped |
| EQUIPMENT_REMOVAL | procedureevents removal items (chest tube removed, pacer wires D/C); catheter interval end (Foley) | STOP at `starttime` / `endtime`. Line interval ends not emitted: the charted discontinuation records them (87% overlap within 2 h) |
| TEST_ORDER | labevents, one row per `specimen_id` | ORDER at draw time; label = tests; result columns absent |
| TEST_ORDER | microbiologyevents, one row per `micro_specimen_id` | ORDER at collection; label = specimen + tests; result columns absent |

Not decisions: emar (execution), poe Medications New/Change (duplicates prescriptions), tests charted
in chartevents/procedureevents (imaging, cultures, ECG: execution of poe orders), alarm limits,
charted copies of POE orders (line / restraints ordered, consult checkboxes, called out, transfers):
the POE order is the decision. CRRT anticoagulation (heparin, citrate) is a DEVICE_SETTING.
A lab appears twice: poe Lab order (order time, no test names) and labevents specimen (draw time, test names);
MIMIC has no key between them.

### Prescription ↔ inputevents link

No shared key. Segment → prescription with same `hadm_id`, same normalized drug,
`rx.starttime − 1 h <= segment.starttime <= coalesce(rx.stoptime, ∞)`. Tie → route
(drip prefers `IV DRIP`, else `IV`), then latest `rx.starttime`. Unlinked → `order_*` null.
Measured on 500 ICU stays: 86–96% of drip/bolus segments fall inside an active same-drug
prescription.

### Catalog (`src/argus/data/decision_catalog.csv`)

| Column | Values |
|---|---|
| `source` | `d_items`, `inputevents_ordercategory`, `poe` |
| `group` | review unit: `linksto/d_items.category` (e.g. `chartevents/Routine Vital Signs`), poe `order_type` |
| `key` | itemid; ordercategoryname; `order_type|order_subtype` |
| `label` | human-readable |
| `category` | MEDICATION, DEVICE_SETTING, TEST_ORDER, NOT_DECISION |
| `subcategory` | e.g. VENTILATOR, DIALYSIS, ECMO, DRIP, FLUID_BOLUS, IMAGING |
| `drug` | normalized drug name (link key) |
| `status` | DRAFT (no evidence), AGREED (evidence agrees), DISPUTED (evidence disagrees or unsure), REVIEWED (clinician) |
| `evidence` | what each independent source said |

`scripts/draft_decision_catalog.py --store DIR --out CSV` drafts every key observed in the
store, then grades it against `scripts/catalog_evidence/`:
- `mimic_code.csv`: items of MIT-LCP/mimic-code concepts (vital signs, ventilator settings,
  CRRT, vasopressors…), category read from the concept column (`tidal_volume_set` vs `_observed`).
- `model_labels.csv`: blind per-row model labels with confidence, made without the draft.

Rerunning keeps REVIEWED rows. Unlisted keys are treated as NOT_DECISION; the real-data
coverage test makes that case empty.

Review: DISPUTED rows one by one; other rows by `group` (filter, confirm, set REVIEWED).

Prescriptions are not catalogued: every non-BASE prescription is a medication order. Their
drug string is normalized in code (first word found among ICU item drugs, else first word)
for the infusion link.

## Acceptance Criteria

1. Every decision row has `start < event_time <= at` and its `(source_table, source_row)` in the snapshot.
2. Vent setting unchanged from before window → one CONTINUE; changed in window → CHANGE; first ever → START.
3. Infusion: first segment START, rate change CHANGE, same-rate bag change none, diluent none, restart after gap START, Stopped STOP at endtime.
4. Drip segment links to its active prescription; PRN bolus + drip both active → drip chosen by route.
5. Prescription ordered before window, started in window → not a decision.
6. Lab and micro decisions carry no result columns, one row per specimen; pending results invisible.
7. emar rows never decisions.
8. Event at exactly `at − window` excluded; at `at` included.
9. Real data: catalog covers every itemid (chartevents/inputevents/procedureevents), poe `(type, subtype)`, inputevents ordercategory in the store.
10. Real data: criterion 1 on 200 seeded samples; p95 `get_window` ≤ 2 s warm.
11. `argus.data.__all__` gains exactly `DecisionWindow`, `DecisionCategory`, `DecisionAction`.
12. Charted non-setting decisions (PCA rate, restraints) follow criterion 2 under their catalog category.
13. poe Lab → TEST_ORDER, Consults → CONSULT, ADT → PATIENT_MOVEMENT, at order time.
14. Planned / complication line removal → EQUIPMENT_REMOVAL STOP; unplanned removal → none; Foley interval end → EQUIPMENT_REMOVAL STOP.

## Residual Risks

- Decisions documented after `at` are invisible → read as omissions (late charting).
- 4–14% of infusion segments unlinked (heuristic link, no key).
- Lab order time unknown; draw time used. Ordered-not-yet-drawn labs invisible.
- Setting CONTINUE compares with the last value in the same stay, even after a long gap.
- Catalog AGREED rows rest on agreement of the heuristic draft with mimic-code and a model, not on a clinician.
