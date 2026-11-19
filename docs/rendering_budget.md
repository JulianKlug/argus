# D5 rendering budget: real-window measurement

Aggregates only: no patient text, no identifiers. Spec: `spec_rendering.md`.

## Run

| Item | Value |
|---|---|
| Command | `ARGUS_STORE_DIR=… uv run python scripts/measure_rendering.py --subjects-out <outside repo> --out <json> --workers 12` |
| Seed | 20261116 (random: seed, heavy: seed + 1) |
| Config digest | `9e2a904b68be1169f32c18dba6f16ffddd1bad5f49d6e93a347dab13d39e84b2` |
| Store manifest SHA256 | `bef00cf1f4256863abd8bb9a68c1a61ba49d3904edbbe6f5eb73719eca0593bb` (MIMIC-IV 3.1 + Note) |
| qwen27 | `22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643` |
| mistral24 | `810b24690f536970d4ad962dafda4db0a2d8b48b3a877639d537a3348b3558a3` |
| medgemma27 | `852e6fdb039823b0cf87695ee4546f749897d7b14d22cc4fd46cc335b1c3295c` |
| Sample | 200 random valid windows; 50 heavy windows = top 50 of 2,000 candidates by visible current-admission records (threshold 24,727 records) |
| Excluded from D6 validation | 347 subjects, list in a local file outside Git |
| Rerun | Same sample after header admission context, notes order and `poe` filtering (incl. `contentless_orders`); figures below are from the rerun |

Token counts are exact GGUF tokenizer counts (D4), no inference.

## Results

Tokens as median / p95 / p99 / max. Counter order: qwen27 · mistral24 · medgemma27.

### Mandatory sections (random, n = 200)

| Section | qwen27 | mistral24 | medgemma27 |
|---|---|---|---|
| Header | 217 / 341 / 457 / 536 | 220 / 347 / 474 / 545 | 212 / 333 / 443 / 531 |
| Decisions | 586 / 2344 / 2804 / 3510 | 620 / 2504 / 2966 / 3657 | 589 / 2358 / 2790 / 3478 |
| Coverage | 83 / 92 / 93 / 93 | 82 / 89 / 90 / 90 | 81 / 91 / 92 / 92 |

Heavy windows: decisions max 2750 / 2948 / 2742; header max 600 / 620 / 581 (services and care units). Mandatory failures: **0 / 250**.

### Optional sections, unbudgeted vs budgeted (mistral24, strictest)

| Section | Quota | Unbudgeted random | Budgeted random | Windows hitting quota (random / heavy) | Omitted items, median / max (random) |
|---|---:|---|---|---|---|
| Current state | 3,000 | 1042 / 1696 / 1888 / 2021 | same | 0 / 0 | 0 / 0 |
| Trends | 2,500 | 848 / 1738 / 2280 / 2398 | same | 0 / 0 | 0 / 0 |
| Notes | 3,000 | 3623 / 52929 / 135988 / 274765 | 2534 / 2919 / 2968 / 2970 | 97 / 36 | 0 / 182 |
| Past admissions | 1,500 | 246 / 4412 / 14927 / 31180 | 246 / 1494 / 1500 / 1500 | 31 / 5 | 0 / 1234 |
| Recent evidence | remainder | 22921 / 44130 / 54450 / 76435 | 9491 / 13293 / 13816 / 13860 | 162 / 50 | 464 / 2369 |
| **Total** | 16,000 | 34054 / 85649 / 174222 / 325482 | 15983 / 15998 / 16000 / 16000 | — | — |

Heavy windows (mistral24): current state max 2573, trends max 2294, recent evidence unbudgeted median 36,724.

mistral24 is the binding tokenizer: its budgeted total sits at 16,000; qwen27 at ~15,600, medgemma27 at ~15,200.

### Unbudgeted tokens per domain (mistral24, random)

| Domain | median | p95 | max |
|---|---:|---:|---:|
| Charted observations | 51,103 | 799,085 | 4,026,702 |
| Labs | 14,802 | 175,773 | 508,047 |
| Medication orders | 6,554 | 62,698 | 283,344 |
| Orders | 5,350 | 47,127 | 174,641 |
| Notes | 3,986 | 60,124 | 283,780 |
| Inputs | 2,174 | 35,780 | 163,015 |
| Input ingredients | 1,927 | 30,883 | 156,294 |
| Medication administrations | 1,563 | 121,242 | 799,522 |
| Outputs | 558 | 8,041 | 53,675 |
| Encounters | 388 | 2,420 | 13,819 |
| Procedures | 341 | 2,857 | 11,185 |
| Microbiology | 328 | 6,260 | 37,062 |
| Diagnoses | 276 | 4,396 | 37,959 |
| Outpatient measurements | 87 | 3,500 | 10,918 |

Whole stay (current + past), all records: the full record never fits any configured context.

### Decision lines per category (mistral24, random)

| Category | median | p95 | max |
|---|---:|---:|---:|
| Medication | 230 | 1,091 | 2,122 |
| Device setting | 90 | 806 | 1,284 |
| Test order | 155 | 633 | 1,306 |
| Procedure | 0 | 124 | 201 |
| Patient movement | 0 | 58 | 89 |
| Consult | 0 | 24 | 79 |
| Equipment removal | 0 | 1 | 92 |

### Core variable presence (current admission, all 250 windows)

Above 90 %: heart rate, respiratory rate, SpO2, temperature, GCS (3), NIBP mean and systolic, creatinine, urea nitrogen, potassium, sodium, glucose, hemoglobin, platelets, WBC. INR 87 %, lactate 70 %, blood gas pH / pO2 / pCO2 69–70 %, bilirubin 62 %, FiO2 57 %, PEEP set 51 %, arterial BP mean / systolic 42–43 %, ventilator mode 37 %.

Low rates follow case mix (ventilation, arterial line); `NOT RECORDED` is then informative. List kept.

### Other

- Reference ID collisions across the sample: **0**.
- Render time (random): p50 0.80 s, p95 1.23 s. `get_window()`: p50 1.28 s, p95 1.63 s.

## Decision

Initial budgets **confirmed** and fixed in `config/rendering.toml`:

| Component | Tokens | Rationale |
|---|---:|---|
| Total | 16,000 | Mandatory content max 4.3 k; leaves ≥ 12 k for evidence; fits mistral24's 32,768 context with prompt framing and 4,096 output tokens |
| Current state | 3,000 | Max observed 2,573: never truncated |
| Trends | 2,500 | Max observed 2,398: never truncated; p99 2,280 |
| Notes | 3,000 | Binding in 48 % of random windows; notes are long-tailed (p95 53 k), shortening keeps impressions |
| Past admissions | 1,500 | Binding in 16 %; breadth-first fill keeps every admission's leading codes |
| Recent evidence | remainder | ~9.5 k median, ~14 k max; omits a median 464 lower-priority records |

Quotas are caps, not reservations: tokens a quota section leaves unused go to recent evidence.

Residual risks: recent evidence is budget-limited in most windows (charted observations dominate the unbudgeted volume and have lowest priority but one); omissions are reported in Coverage. Budget sensitivity belongs to A7 arms.
