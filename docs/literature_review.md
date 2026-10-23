# ARGUS — Literature Review

Scope: prior work relevant to an agentic LLM system detecting erroneous, delayed or omitted
treatment in ICU patients from multimodal time-series + text (plan: `gist.md`).
Searched: Oct 2026. Items marked *preprint* are not peer reviewed.

## 1. Bottom line

No published system does what ARGUS proposes end-to-end
(multi-agent LLM + ICU time series + error/omission detection + clinician-adjudicated validation).
Pieces exist, and two works are close enough to shape the design:

| Work | Overlap with ARGUS | Gap |
|---|---|---|
| **RealICU** (Shen et al., arXiv 2605.13542, 2026, *preprint*) | MIMIC-IV ICU trajectories, windowed (30 min), physician *hindsight* labels, includes a **"Red Flag" unsafe-action task** and a structured-memory agent (ICU-Evo) | Benchmark, not an error-surveillance system. Red Flags = prospective "actions to avoid", not audit of actions taken. ICU-Evo has no guideline retrieval, no adjudication, no PPV. Its annotation *did* grade recorded actions (best-practice / acceptable / potentially harmful / omitted) — possible ready-made error labels if released |
| **SAFE-AI** (Trujillo et al., PLOS Digit Health 2026) | LLM agent pipeline detecting patient safety events (delay, overdose) in charts; 2 expert reviewers + consensus | Pediatric EMS cardiac arrest only, 2 error types; LLM only extracts, deterministic expert ontology decides |

Everything else covers a sub-problem (medication-only, text-only, or non-ICU).

## 2. Burden of ICU error (motivation)

- **Critical Care Safety Study** — Rothschild et al., Crit Care Med 2005;33:1694. ~20% of MICU/CCU
  admissions had an adverse event, ~45% preventable; medication (wrong dose) most common;
  >90% during routine care, not emergencies.
- **ICU Trigger Tool** — Resar et al., Crit Care Med 2006. 23 triggers, 54 hospitals, nurse screening +
  physician confirmation. The manual, retrospective gold standard ARGUS would automate.
- **Trigger-tool PPV systematic review** — PMC12622303 (2025). Baseline PPVs for rule-based triggers;
  useful benchmark for ARGUS's PPV.
- **Errors of omission** — Herasevich & Gajic (ICU Manag Pract 2007): omissions more frequent than
  commissions in ICU; early rule-based "data sniffers" for real-time surveillance.

## 3. Non-LLM automated error detection

| Work | Method | Result | Relevance |
|---|---|---|---|
| MedAware — Schiff et al., JAMIA 2017; Rozenblum et al., 2020 | ML outlier detection on prescriptions vs. patient context | 747,985 patients; ~75% of sampled alerts valid; 300-chart sample (PPV only) | Closest commercial precedent; same PPV-only limitation as ARGUS plan |
| ED Trigger Tool, multicenter (Ann Emerg Med 2025) | Computerized EHR trigger query + review | AEs in 44% of triggered visits | Rule-based comparator |
| Real-time VAE detection — Shenoy et al., ICHE 2018 | Monitor + EHR + micro rules | Removes human misclassification | Shows bedside time series usable for surveillance |
| TREWS — Adams et al., Nat Med 2022 | ML sepsis alert, prospective | Lower mortality when alert acted on fast | Delay-of-treatment framing; prospective evidence model |
| AI Clinician — Komorowski et al., Nat Med 2018 | RL policy on MIMIC-III sepsis | Mortality lowest when clinicians matched policy | "Deviation from optimal policy" ≈ error signal; MIMIC precedent |

## 4. LLM-based error / safety-event detection

| Work | Setting | Design | Key numbers |
|---|---|---|---|
| **SAFE-AI** (PLOS Digit Health 2026) | Peds EMS charts, n=300 | LLM extraction → expert ontology → deterministic code | Delay acc 0.94, recall 0.98, **precision 0.69**; abstract figures (97.9%/91.6%) disagree with Table 2 |
| **MEDEC / MEDIQA-CORR** (Ben Abacha et al., Findings ACL 2025) | 3,848 clinical texts, 5 error types | Benchmark, injected errors | Physicians still beat LLMs |
| **Paired evaluation critique** (Zhang & Beheshti, MLHC 2026) | MEDEC-style sets, 15 LLMs | Paired clean/erroneous notes | 13/15 models below random pairwise discrimination despite "moderate" F1 |
| Ong et al., RAG-LLM CDSS (Cell Rep Med 2025; arXiv 2402.01741) | 91 prescribing scenarios, 16 specialties | GPT-4 + RAG, pharmacist co-pilot | Co-pilot best: accuracy 61%, precision 0.57 |
| NHS medication safety review (Normand et al., arXiv 2512.21127, *preprint*) | Real primary-care EHR, n=277 | LLM review, expert grading | Sens 100%, spec 83%, **fully correct only 47%**; failures = context, not knowledge |
| **Rx-LLM** (medRxiv 2025, *preprint*) | ICU medication management | Benchmark by 5 critical-care pharmacists | Best accuracy 70%, high task variance |
| MEDIC — Pais et al., Nat Med 2024 | Online pharmacy directions | Fine-tuned LLM + rule guardrails, deployed | Near-misses −33% |
| Oncology discharge summaries (JCO CCI 2025) | Synthetic vignettes, injected errors | o4-mini, Gemini 2.5 Pro | 98% detection vs 48% specialists (synthetic) |
| LLM bleeding-event chart review (PubMed 42600978, 2026) | Critically ill, high bleeding risk | GPT-4o-mini + adjudication | 11.5% of flagged events not confirmed |
| ED missed-diagnosis eTrigger + LLM (medRxiv 2025.10.06, *preprint*) | ED | Rule eTrigger → LLM adjudicator, 2-physician review | Two-stage design analogous to ARGUS expert→adjudicator |
| Multi-LLM disagreement (Wittlinger et al., medRxiv 2026, *preprint*) | Endoscopy reports | 4 local LLMs, disagreement score | Lowest-agreement 6.5% held ~80% of errors; idea usable for adjudication agent |

Recurring findings: high recall, mediocre precision; failures driven by context and temporal
anchoring, not missing medical knowledge; hybrid (LLM extraction + deterministic rules) beats pure LLM
on narrow tasks.

## 5. Multi-agent clinical LLM systems

- **MDAgents** (Kim et al., NeurIPS 2024), **MedAgents** (Tang et al., 2024): adaptive multi-agent
  collaboration for medical QA; not EHR/time series.
- **EHRAgent** (Shi et al., EMNLP 2024): LLM writes code to query EHR tables — relevant for the
  summarizer agent querying MIMIC instead of reading serialized data.
- **SepsisAgent** (arXiv 2605.14723, 2026, *preprint*): LLM + clinical world model on MIMIC-IV sepsis;
  reports guideline adherence and unsafe-action metrics.
- **Multi-agent sepsis management** (Healthc Inform Res): 3 agents incl. guideline-compliance agent;
  single MIMIC-IV case — anecdotal.
- **Stroke multi-agent framework** (JMIR 2026): omission rate 38.5% → 24.5% vs single LLM. Evidence that
  agent decomposition reduces omissions.
- **Hager et al., Nat Med 2024** (MIMIC-IV-Ext-CDM): LLMs poor at guideline adherence and lab
  interpretation; sensitive to information order and quantity. Directly relevant to the expert and
  research agents.
- **Agent taxonomy survey** (arXiv 2602.04813, 2026): treatment planning/prescription is the least
  mature agent area (~59% gaps).

## 6. Time-series reasoning with LLMs

| Model | Notes | Fit for ARGUS |
|---|---|---|
| **TimeOmni-1** (Guan et al., ICLR 2026; arXiv 2509.24803) | Qwen2.5-7B, SFT + RL; perception/forecast/decision on TSR-Suite | General-domain, no clinical evaluation. Unproven on irregular, sparse ICU data |
| **OpenTSLM** (Langer et al., ICML 2026; arXiv 2510.02410) | Time-series encoder + LLM; ECG-QA, sleep, HAR | Medical, but dense waveforms, not charted vitals/labs |
| **CLIR-Bench** (arXiv 2607.09880, *preprint*) | QA over irregular clinical time series | Candidate eval for summarizer agent |
| Temporal reasoning on MIMIC-III ICU stays (Findings EMNLP 2025) | Multi-day summarization from chartevents/labs | Direct precedent for summarizer agent |
| **RealICU / ICU-Evo** | Structured memory across windows | Addresses anchoring across sequential windows |

Charted ICU data (hourly vitals, sparse labs) may not need a time-series encoder; textual
serialization with computed trends is the cheaper baseline TimeOmni-1 must beat.

## 7. Data governance

- **PhysioNet LLM policy** (Sept 2025, supersedes 2023): zero data retention, no training,
  no human review by providers; local models strongly recommended; unverifiable services prohibited.
  The DUA forbids sharing data via APIs otherwise.

## 8. Implications for the ARGUS plan

1. **Use MIMIC-IV, not MIMIC-III.** MIMIC-III (2001–2012) mixes CareVue/MetaVision; reliable
   medication administration timing exists only for MetaVision. MIMIC-IV (2008–2022) is current,
   matches RealICU, SepsisAgent, Hager et al., and enables direct benchmarking.
2. **Anchor guidelines to care date.** The research agent must judge against guidelines in force at
   the time of care (e.g. SSC 2012 vs 2021), or "errors" will be anachronistic.
3. **Prevent look-ahead leakage.** Only data and notes timestamped before each 6-h window ends may be
   visible. Discharge summaries leak outcome.
4. **PPV alone is insufficient** (same limit as MedAware, flagged in adjudication literature).
   Add: clinician review of a random sample of unflagged windows (sensitivity estimate), and/or
   injected-error paired evaluation (Zhang & Beheshti 2026), and/or RealICU Red Flag labels.
5. **Add baselines.** Rule-based ICU trigger tool (Resar 2006) and single-LLM without agents.
   Agent ablation is planned; a non-LLM baseline is not.
6. **Blind reviewers or measure anchoring.** Reviewers reading the agent's rationale are prone to
   automation bias. Consider a blinded first pass, then rationale.
7. **Use a standard harm scale** (NCC MERP or AHRQ Common Formats) and an error taxonomy
   (commission / omission / delay; medication, ventilation, fluids, monitoring) for comparability.
8. **Distinguish documentation gaps from care errors.** MIMIC lacks clinician intent; many "omissions"
   will be undocumented actions. Add a "cannot determine" label.
9. **Hybrid over pure LLM** for checkable rules (dose ranges, bundle timing): SAFE-AI and MEDIC show
   deterministic logic improves precision. Reserve LLM reasoning for context-dependent judgments.
10. **Sample size.** 50 patients × 6-h windows yields few true errors per category given
    ~20% AE incidence; report CIs and treat as feasibility, as planned.
11. **Local LLM** (or verified zero-retention endpoint) per PhysioNet policy; budget GPUs.
12. **"Near real time"** is untested retrospectively; report per-window latency and cost.

## 9. Sources

- RealICU — https://arxiv.org/abs/2605.13542
- SAFE-AI — https://pmc.ncbi.nlm.nih.gov/articles/PMC13585220 ; https://pubmed.ncbi.nlm.nih.gov/41445638/
- Rothschild 2005 — https://psnet.ahrq.gov/issue/critical-care-safety-study-incidence-and-nature-adverse-events-and-serious-medical-errors
- Resar 2006 ICU trigger tool — https://psnet.ahrq.gov/issue/trigger-tool-identify-adverse-events-intensive-care-unit
- Trigger tool PPV review — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12622303/
- Herasevich & Gajic — https://healthmanagement.org/c/icu/issuearticle/100161
- ED Trigger Tool — https://www.annemergmed.com/article/S0196-0644(25)00993-X/fulltext
- VAE detection — https://www.cambridge.org/core/journals/infection-control-and-hospital-epidemiology/article/abs/realtime-automated-detection-of-ventilatorassociated-events-avoiding-missed-detections-misclassifications-and-false-detections-due-to-human-error/E81521C81E314CF9F1D0E031DE9F027C
- MedAware — https://www.mobihealthnews.com/news/harvard-study-almost-800k-lives-shows-medaware-technology-reduces-medication-error ; https://www.medaware.com/medical-research
- MEDEC — https://arxiv.org/abs/2412.19260v2
- Paired evaluation — https://arxiv.org/abs/2608.16643
- Ong et al. — https://arxiv.org/abs/2402.01741 ; https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(25)00396-9
- NHS medication review — https://arxiv.org/abs/2512.21127
- Rx-LLM — https://www.medrxiv.org/content/medrxiv/early/2025/12/30/2025.12.01.25341004.full.pdf
- MEDIC — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11186789/
- Oncology error detection — https://ascopubs.org/doi/10.1200/CCI-25-00194
- Bleeding chart review — https://pubmed.ncbi.nlm.nih.gov/42600978/
- ED eTrigger + LLM — https://www.medrxiv.org/content/10.1101/2025.10.06.25337201.full.pdf
- Multi-LLM disagreement — https://www.medrxiv.org/content/10.64898/2026.05.04.26352392v1
- SepsisAgent — https://arxiv.org/html/2605.14723v1
- Multi-agent sepsis — https://e-hir.org/journal/view.php?number=1252
- Stroke multi-agent — https://pmc.ncbi.nlm.nih.gov/articles/PMC13423760/
- Hager et al. — https://www.medrxiv.org/content/10.1101/2024.01.26.24301810 ; https://physionet.org/content/mimic-iv-ext-cdm
- Agent taxonomy — https://arxiv.org/html/2602.04813v1
- TimeOmni-1 — https://arxiv.org/abs/2509.24803
- OpenTSLM — https://arxiv.org/abs/2510.02410
- CLIR-Bench — https://arxiv.org/pdf/2607.09880
- MIMIC-III temporal reasoning — https://aclanthology.org/2025.findings-emnlp.1128.pdf
- PhysioNet LLM policy — https://physionet.org/news/post/llm-responsible-use/
- Cited from prior knowledge, not re-fetched: MDAgents (NeurIPS 2024), MedAgents (2024),
  EHRAgent (arXiv 2401.07128), TREWS (Nat Med 2022), AI Clinician (Nat Med 2018),
  Schiff et al. JAMIA 2017.
