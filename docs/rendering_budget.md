# D5 rendering budget: real-window measurement

Aggregates only: no patient text, no identifiers. Spec: `spec_rendering.md`.

## Run

| Item | Value |
|---|---|
| Command | `ARGUS_STORE_DIR=… uv run python scripts/measure_rendering.py --subjects-out <outside repo> --out <json> --workers 12` |
| Seed | 20261116 (random: seed, heavy: seed + 1) |
| Config digest | `c54073b86a9eeed3da8f7fe924cd06e500f79b161642225282ed0415425a3fc8` |
| Store manifest SHA256 | `bef00cf1f4256863abd8bb9a68c1a61ba49d3904edbbe6f5eb73719eca0593bb` (MIMIC-IV 3.1 + Note) |
| qwen27 | `22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643` |
| mistral24 | `810b24690f536970d4ad962dafda4db0a2d8b48b3a877639d537a3348b3558a3` |
| medgemma27 | `852e6fdb039823b0cf87695ee4546f749897d7b14d22cc4fd46cc335b1c3295c` |
| Sample | 200 random valid windows; 50 heavy windows = top 50 of 2,000 candidates by visible current-admission records (threshold 24,727 records) |
| Excluded from D6 validation | 347 subjects, list in a local file outside Git |
| Rerun | Same sample after PR #2 review fixes: header admission context, notes order, `poe` filtering, test results kept beside `TEST_ORDER` decisions, every differing availability time shown (`documented`, incl. notes and latest trend point), per-domain omissions |
| Full aggregates | `rendering_budget.json` (all counters, sections, domains, categories; no identifiers) |

Token counts are exact GGUF tokenizer counts (D4), no inference.

## Results

Token counts per counter: qwen27 · mistral24 · medgemma27. mistral24 is binding: budgeted total median 15,966 (qwen27 15,621, medgemma27 15,271).

Mandatory failures: **0 / 250**. Reference ID collisions: **0**.

### Sections, random (n = 200)

Tokens median / p95 / p99 / max.

| Section | Quota | Mode | qwen27 | mistral24 | medgemma27 |
|---|---:|---|---|---|---|
| Coverage | mandatory | budgeted | 101 / 132 / 142 / 144 | 98 / 131 / 141 / 143 | 99 / 130 / 139 / 141 |
| Current state | 3,000 | unbudgeted | 1,214 / 1,817 / 2,039 / 2,119 | 1,254 / 1,926 / 2,115 / 2,235 | 1,192 / 1,780 / 1,982 / 2,066 |
| Current state | 3,000 | budgeted | 1,214 / 1,817 / 2,039 / 2,119 | 1,254 / 1,926 / 2,115 / 2,235 | 1,192 / 1,780 / 1,982 / 2,066 |
| Decisions | mandatory | budgeted | 586 / 2,345 / 2,804 / 3,510 | 620 / 2,505 / 2,966 / 3,657 | 589 / 2,359 / 2,790 / 3,478 |
| Header | mandatory | budgeted | 216 / 341 / 457 / 536 | 220 / 347 / 474 / 545 | 212 / 333 / 443 / 531 |
| Notes | 3,000 | unbudgeted | 3,882 / 56,586 / 143,815 / 284,742 | 3,690 / 54,301 / 137,976 / 276,912 | 3,664 / 53,906 / 136,283 / 271,789 |
| Notes | 3,000 | budgeted | 2,694 / 2,990 / 3,000 / 3,000 | 2,554 / 2,922 / 2,940 / 2,949 | 2,548 / 2,847 / 2,874 / 2,879 |
| Past admissions | 1,500 | unbudgeted | 240 / 4,369 / 14,757 / 30,509 | 246 / 4,412 / 14,928 / 31,180 | 224 / 4,088 / 14,045 / 29,283 |
| Past admissions | 1,500 | budgeted | 240 / 1,458 / 1,475 / 1,478 | 246 / 1,494 / 1,500 / 1,500 | 224 / 1,374 / 1,390 / 1,399 |
| Recent evidence | remainder | unbudgeted | 27,460 / 53,587 / 62,429 / 91,426 | 28,298 / 55,403 / 64,354 / 94,355 | 26,984 / 53,008 / 61,546 / 89,941 |
| Recent evidence | remainder | budgeted | 8,764 / 12,405 / 12,834 / 12,960 | 9,030 / 12,766 / 13,311 / 13,374 | 8,594 / 12,234 / 12,644 / 12,774 |
| Trends | 3,600 | unbudgeted | 1,204 / 2,455 / 3,105 / 3,375 | 1,217 / 2,482 / 3,144 / 3,427 | 1,194 / 2,448 / 3,089 / 3,363 |
| Trends | 3,600 | budgeted | 1,204 / 2,455 / 3,105 / 3,375 | 1,217 / 2,482 / 3,144 / 3,427 | 1,194 / 2,448 / 3,089 / 3,363 |
| **Total** | 16,000 | unbudgeted | 39,436 / 101,413 / 186,906 / 338,804 | 40,382 / 102,291 / 182,049 / 332,283 | 38,910 / 99,578 / 178,284 / 324,163 |
| **Total** | 16,000 | budgeted | 15,621 / 15,774 / 15,855 / 15,903 | 15,966 / 15,995 / 15,999 / 16,000 | 15,271 / 15,391 / 15,413 / 15,442 |

| Optional section | Windows hitting quota | Items available | Selected | Omitted |
|---|---:|---|---|---|
| Current state | 0 | 35 / 55 / 69 / 74 | 35 / 55 / 69 / 74 | 0 / 0 / 0 / 0 |
| Notes | 99 | 7 / 72 / 138 / 185 | 3 / 8 / 12 / 14 | 0 / 70 / 135 / 183 |
| Past admissions | 31 | 9 / 174 / 577 / 1,282 | 9 / 57 / 59 / 60 | 0 / 123 / 528 / 1,234 |
| Recent evidence | 170 | 824 / 1,595 / 1,865 / 2,616 | 258 / 372 / 406 / 436 | 559 / 1,339 / 1,709 / 2,455 |
| Trends | 0 | 9 / 18 / 24 / 25 | 9 / 18 / 24 / 25 | 0 / 0 / 0 / 0 |

### Sections, heavy (n = 50)

Tokens median / p95 / p99 / max.

| Section | Quota | Mode | qwen27 | mistral24 | medgemma27 |
|---|---:|---|---|---|---|
| Coverage | mandatory | budgeted | 119 / 139 / 143 / 143 | 119 / 138 / 142 / 142 | 116 / 137 / 140 / 140 |
| Current state | 3,000 | unbudgeted | 1,496 / 2,479 / 2,650 / 2,718 | 1,546 / 2,602 / 2,788 / 2,875 | 1,468 / 2,417 / 2,578 / 2,660 |
| Current state | 3,000 | budgeted | 1,496 / 2,479 / 2,650 / 2,718 | 1,546 / 2,602 / 2,788 / 2,875 | 1,468 / 2,417 / 2,578 / 2,660 |
| Decisions | mandatory | budgeted | 789 / 2,597 / 2,735 / 2,750 | 872 / 2,757 / 2,933 / 2,948 | 812 / 2,609 / 2,731 / 2,742 |
| Header | mandatory | budgeted | 282 / 475 / 551 / 600 | 286 / 483 / 569 / 620 | 274 / 464 / 534 / 581 |
| Notes | 3,000 | unbudgeted | 10,931 / 45,724 / 110,492 / 163,867 | 10,216 / 43,543 / 106,737 / 158,722 | 10,140 / 43,159 / 104,878 / 155,566 |
| Notes | 3,000 | budgeted | 2,922 / 2,998 / 3,000 / 3,000 | 2,792 / 2,885 / 2,898 / 2,898 | 2,757 / 2,854 / 2,865 / 2,869 |
| Past admissions | 1,500 | unbudgeted | 13 / 4,537 / 11,347 / 17,405 | 14 / 4,548 / 11,439 / 17,521 | 9 / 4,259 / 10,786 / 16,562 |
| Past admissions | 1,500 | budgeted | 13 / 1,437 / 1,471 / 1,499 | 14 / 1,482 / 1,487 / 1,490 | 9 / 1,346 / 1,385 / 1,400 |
| Recent evidence | remainder | unbudgeted | 43,643 / 74,095 / 85,796 / 89,054 | 45,078 / 76,519 / 88,203 / 91,537 | 43,036 / 73,126 / 84,511 / 87,803 |
| Recent evidence | remainder | budgeted | 8,686 / 12,232 / 12,721 / 12,741 | 9,013 / 12,577 / 13,054 / 13,136 | 8,514 / 12,020 / 12,324 / 12,440 |
| Trends | 3,600 | unbudgeted | 1,472 / 2,629 / 3,107 / 3,251 | 1,489 / 2,674 / 3,160 / 3,302 | 1,458 / 2,611 / 3,093 / 3,234 |
| Trends | 3,600 | budgeted | 1,472 / 2,629 / 3,107 / 3,251 | 1,489 / 2,674 / 3,160 / 3,302 | 1,458 / 2,611 / 3,093 / 3,234 |
| **Total** | 16,000 | unbudgeted | 62,649 / 102,734 / 197,078 / 278,040 | 63,233 / 103,928 / 196,285 / 275,797 | 60,742 / 100,117 / 190,209 / 267,551 |
| **Total** | 16,000 | budgeted | 15,638 / 15,817 / 15,855 / 15,861 | 15,976 / 15,997 / 15,999 / 15,999 | 15,291 / 15,406 / 15,425 / 15,433 |

| Optional section | Windows hitting quota | Items available | Selected | Omitted |
|---|---:|---|---|---|
| Current state | 0 | 37 / 64 / 78 / 83 | 37 / 64 / 78 / 83 | 0 / 0 / 0 / 0 |
| Notes | 36 | 28 / 69 / 113 / 152 | 3 / 13 / 15 / 15 | 20 / 65 / 111 / 149 |
| Past admissions | 5 | 0 / 174 / 456 / 700 | 0 / 49 / 56 / 62 | 0 / 125 / 408 / 653 |
| Recent evidence | 50 | 1,302 / 2,166 / 2,364 / 2,433 | 240 / 338 / 349 / 357 | 1,008 / 2,017 / 2,259 / 2,351 |
| Trends | 0 | 11 / 20 / 23 / 24 | 11 / 20 / 23 / 24 | 0 / 0 / 0 / 0 |

### Unbudgeted tokens per domain (random, current + past)

median / p95 / max.

| Domain | qwen27 | mistral24 | medgemma27 |
|---|---|---|---|
| Charted observations | 65,304 / 1,067,660 / 5,227,143 | 67,118 / 1,093,786 / 5,362,519 | 64,526 / 1,054,771 / 5,174,756 |
| Labs | 18,910 / 229,581 / 670,682 | 19,308 / 233,891 / 683,296 | 18,866 / 229,044 / 668,662 |
| Medication orders | 6,408 / 62,905 / 278,256 | 6,602 / 64,453 / 285,088 | 6,311 / 62,009 / 274,102 |
| Orders | 5,192 / 45,559 / 169,254 | 5,350 / 47,127 / 174,641 | 5,055 / 44,972 / 165,392 |
| Notes | 4,356 / 66,660 / 297,542 | 4,198 / 64,757 / 290,145 | 4,167 / 64,259 / 284,853 |
| Inputs | 2,042 / 34,267 / 156,738 | 2,174 / 35,781 / 163,015 | 2,000 / 33,851 / 154,647 |
| Input ingredients | 1,787 / 28,903 / 146,386 | 1,927 / 30,883 / 156,294 | 1,777 / 28,796 / 145,941 |
| Medication administrations | 1,833 / 145,109 / 999,437 | 1,882 / 148,905 / 1,023,454 | 1,846 / 144,126 / 990,201 |
| Outputs | 664 / 10,311 / 70,735 | 706 / 10,849 / 74,173 | 666 / 10,323 / 70,785 |
| Encounters | 370 / 2,312 / 13,127 | 388 / 2,420 / 13,819 | 370 / 2,300 / 13,099 |
| Microbiology | 352 / 7,660 / 43,053 | 383 / 8,190 / 45,819 | 333 / 7,235 / 41,299 |
| Procedures | 330 / 2,938 / 12,125 | 344 / 3,029 / 12,493 | 324 / 2,918 / 12,036 |
| Diagnoses | 270 / 4,272 / 36,901 | 276 / 4,397 / 37,959 | 262 / 4,168 / 36,297 |
| Outpatient measurements | 120 / 4,928 / 15,382 | 122 / 5,015 / 15,684 | 121 / 4,952 / 15,526 |

### Recent evidence omitted per domain (random, budgeted)

Items, median / p95 / max.

| Domain | Omitted |
|---|---|
| Charted observations | 504 / 1,194 / 2,219 |
| Encounters | 0 / 0 / 1 |
| Inputs | 0 / 47 / 102 |
| Labs | 0 / 97 / 151 |
| Medication administrations | 0 / 28 / 50 |
| Medication orders | 0 / 72 / 153 |
| Microbiology | 0 / 1 / 4 |
| Orders | 0 / 21 / 46 |
| Outputs | 0 / 22 / 36 |
| Procedures | 0 / 4 / 7 |

### Decision lines per category (random)

median / p95 / max.

| Category | qwen27 | mistral24 | medgemma27 |
|---|---|---|---|
| Consult | 0 / 22 / 72 | 0 / 24 / 79 | 0 / 21 / 66 |
| Device setting | 83 / 739 / 1,162 | 90 / 807 / 1,284 | 87 / 763 / 1,218 |
| Equipment removal | 0 / 1 / 84 | 0 / 2 / 92 | 0 / 1 / 82 |
| Medication | 222 / 1,056 / 2,062 | 230 / 1,091 / 2,122 | 219 / 1,028 / 2,035 |
| Patient movement | 0 / 51 / 80 | 0 / 58 / 89 | 0 / 50 / 79 |
| Procedure | 0 / 112 / 184 | 0 / 124 / 201 | 0 / 106 / 174 |
| Test order | 149 / 605 / 1,278 | 155 / 633 / 1,306 | 145 / 597 / 1,246 |

### Core variable presence (current admission, all 250 windows)

Above 90 %: heart rate, respiratory rate, SpO2, temperature, GCS (3), NIBP mean and systolic, creatinine, urea nitrogen, potassium, sodium, glucose, hemoglobin, platelets, WBC. INR 87 %, lactate 70 %, blood gas pH / pO2 / pCO2 69–70 %, bilirubin 62 %, FiO2 57 %, PEEP set 51 %, arterial BP mean / systolic 42–43 %, ventilator mode 37 %.

Low rates follow case mix (ventilation, arterial line); `NOT RECORDED` is then informative. List kept.

### Other

- Render time (random): p50 0.82 s, p95 1.30 s. `get_window()`: p50 1.25 s, p95 1.57 s.
- Availability times on every differing line (default `documented_delay_minutes = 0`) add ~20 % to current state and ~45 % to trends (incl. the latest point); recent evidence omits a median 559 records (464 without availability times). A positive threshold is an A7 compression arm.

## Decision

Budgets fixed in `config/rendering.toml`; initial values confirmed except trends (2,500 → 3,600):

| Component | Tokens | Rationale |
|---|---:|---|
| Total | 16,000 | Mandatory content max 4.4 k; leaves ≥ 11 k for evidence; fits mistral24's 32,768 context with prompt framing and 4,096 output tokens |
| Current state | 3,000 | Max observed 2,875 (heavy): never truncated |
| Trends | 3,600 | Availability times and the latest point raised the max to 3,427; 3,600 keeps ~5 % headroom, never truncated. At 2,750, 1–5 trends (last in list: blood gases) were cut in 10 / 250 windows. Cap, so unused tokens go to recent evidence |
| Notes | 3,000 | Binding in 50 % of random windows; long-tailed (p95 54 k); shortening keeps impressions; D9 summaries address the rest |
| Past admissions | 1,500 | Binding in 16 %; breadth-first fill keeps every admission's leading codes |
| Recent evidence | remainder | ~9.0 k median; omits a median 559 records, charted observations 504 of them (lowest priority but one) |

Quotas are caps, not reservations: tokens a quota section leaves unused go to recent evidence.

Residual risks: recent evidence is budget-limited in most windows; Coverage reports omissions per domain. Whether the selection policy impairs detection of specific error categories is an A7 question; token compliance alone does not establish information sufficiency.
