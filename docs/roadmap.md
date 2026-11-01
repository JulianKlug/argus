# ARGUS — Roadmap

Plan: `local_docs/gist.md`. Design inputs: `literature_review.md` §8 (cited as L1–L12).
Month 1 = 2026-10 (first commit).

## Goal

Feasibility study: agents scan sequential 6 h windows of 50 MIMIC-IV ICU patients; two clinicians
adjudicate every flag.

| Outcome | Measure |
|---|---|
| Primary | PPV of flags, interrater κ, error category distribution, clinical severity |
| Secondary | False-positive patterns, per-agent contribution (ablation), latency and cost per window (L12) |

## Dependencies

```
 store ─► snapshot ─► decision window ─► rendering ─► agents ─► harness ─► study run ─► review ─► analysis
   D1         D1            D2              D5        A2–A5      A1          V2          V3–V4     R1–R2
                                                        ▲                     ▲            ▲
 LLM backbone (D4) ─────────────────────────────────────┤                     │            │
 taxonomy + harm scale (D7) ─── expert, adjudicator ────┘                     │            │
 cohort (D6) ─────────────────────────────────────────────────────────────────┘            │
 taxonomy + harm scale (D7) ─── review form ───────────────────────────────────────────────┘
```

## Phases

### P1 Data & design — M1–3 (Oct–Dec 2026)

| Id | Deliverable | Exit criterion | Depends on | Spec | Status |
|---|---|---|---|---|---|
| D1 | Leak-free snapshot engine (L1, L3) | Snapshot criteria green on real store | — | `spec_snapshot_engine.md` | ✅ |
| D2 | Decision window + catalog | Window criteria green on real store | D1 | `spec_decision_window.md` | ✅ |
| D3 | MIMIC-IV-Note integration | 35 tables in store; note leak tests green | D1 | `spec_snapshot_engine.md` | ✅ |
| D4 | LLM backbone (L11) | Model chosen: local or verified zero-retention; GPU budget; version pinned | — | `spec_llm_backbone.md` | |
| D5 | Window rendering to prompt text | Token count per domain measured on real windows; budget fixed for trends, notes, past admissions | D2, D4 | `spec_rendering.md` | **next** |
| D6 | Cohort | Inclusion criteria; seeded draw of 50 validation + disjoint development patients; windows per patient | D1 | `spec_cohort.md` | |
| D7 | Error taxonomy + harm scale (L7, L8) | Commission / omission / delay × domain; NCC MERP; labels true error / reasonable alternative / spurious / cannot determine | — | `spec_taxonomy.md` | |

### P2 Agents & internal testing — M4–7 (Jan–Apr 2027)

| Id | Deliverable | Exit criterion | Depends on | Spec |
|---|---|---|---|---|
| A1 | Run harness: orchestrator, structured outputs, run log | Rerun reproduces log; log holds prompt hash, model version, latency, cost per window (L12) | D4 | `spec_harness.md` |
| A2 | Summary agent | Memory across sequential windows; summaries checked on development cohort | A1, D5 | `spec_agents.md` |
| A3 | Expert agent (L9) | LLM + deterministic checks (dose ranges, bundle timing); output typed by D7 taxonomy | A2, D7 | `spec_agents.md` |
| A4 | Research agent (L2) | Retrieval over guideline corpus anchored to care date | A3 | `spec_agents.md` |
| A5 | Adjudication agent | Likelihood, severity, escalate yes/no per flag | A4, D7 | `spec_agents.md` |
| A6 | Baselines (L5) | Rule-based ICU trigger tool (Resar 2006); single LLM without agents | A1, D7 | `spec_baselines.md` |
| A7 | Internal evaluation (L4) | Injected-error paired set on development cohort; ablation switches per agent | A5, A6, D6 | `spec_internal_eval.md` |

### P3 Retrospective validation — M8–10 (May–Jul 2027)

| Id | Deliverable | Exit criterion | Depends on |
|---|---|---|---|
| V1 | Frozen system + preregistered protocol | Metrics, sample, stop rules registered; code and model tagged | A7 |
| V2 | Study run | All windows of 50 validation patients processed; logs complete | V1, D6 |
| V3 | Clinician review (L6) | Review tool: blinded first pass, then agent rationale; 2 reviewers + consensus | V2, D7 |
| V4 | Unflagged-window review (L4) | Random unflagged sample reviewed; sensitivity estimable | V2 |

### P4 Analysis & dissemination — M11–12 (Aug–Sep 2027)

| Id | Deliverable | Exit criterion | Depends on |
|---|---|---|---|
| R1 | Analysis (L10) | Outcomes table with CIs, κ, false-positive patterns, ablation, baselines | V3, V4 |
| R2 | Dissemination | Manuscript; code release; derived labels released under PhysioNet rules | R1 |

## Open decisions

| Decision | Options | Blocks |
|---|---|---|
| LLM hosting | Local model; verified zero-retention endpoint | D4 |
| Sensitivity estimate | Unflagged sample; injected errors; RealICU labels (if released) | A7, V4 |
| Reviewer blinding | Blinded first pass; rationale shown | V3 |
| Guideline corpus | Source, versions by year | A4 |

## Risks

| Risk | Mitigation |
|---|---|
| Late charting read as omission (L8) | "Cannot determine" label; residual risk logged in `spec_decision_window.md` |
| Few true errors at n=50 (L10) | Report CIs; framed as feasibility |
| Window exceeds context | D5 measures before agent design |
| LLM order/quantity sensitivity (Hager 2024) | D5 fixes rendering order; A7 tests it |
| Reviewer time | Size V3/V4 sample from V2 flag count before review |
