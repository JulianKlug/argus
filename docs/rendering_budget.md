# D5 rendering budget: real-window measurement

Aggregates only: no patient text, no identifiers. Spec: `spec_rendering.md`.

## Run

| Item | Value |
|---|---|
| Command | `ARGUS_STORE_DIR=… uv run python scripts/measure_rendering.py --subjects-out <outside repo> --out <json> --workers 12` |
| Seed | 20261116 (random: seed, heavy: seed + 1) |
| Config digest | `a343c7173c2409ab2c82d6305d45b00a3571c0144de1838dd4edc45c5d46204f` |
| Store manifest SHA256 | `bef00cf1f4256863abd8bb9a68c1a61ba49d3904edbbe6f5eb73719eca0593bb` (MIMIC-IV 3.1 + Note) |
| qwen27 | `22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643` |
| mistral24 | `810b24690f536970d4ad962dafda4db0a2d8b48b3a877639d537a3348b3558a3` |
| medgemma27 | `852e6fdb039823b0cf87695ee4546f749897d7b14d22cc4fd46cc335b1c3295c` |
| Sample | 200 random valid windows; 50 heavy windows = top 50 of 2,000 candidates by visible current-admission records (threshold 24,727 records) |
| Excluded from D6 validation | 347 subjects, list in a local file outside Git |
| Rerun | Same sample after PR #2 review fixes: header admission context, notes order, `poe` filtering, test results kept beside `TEST_ORDER` decisions, `documented` availability times, per-domain omissions |
| Full aggregates | `rendering_budget.json` (all counters, sections, domains, categories; no identifiers) |

Token counts are exact GGUF tokenizer counts (D4), no inference.

## Results

Token counts per counter: qwen27 · mistral24 · medgemma27. mistral24 is binding: budgeted total median 15,963 (qwen27 15,586, medgemma27 15,221).

Mandatory failures: **0 / 250**. Reference ID collisions: **0**.

### Sections, random (n = 200)

Tokens median / p95 / p99 / max.

| Section | Quota | Mode | qwen27 | mistral24 | medgemma27 |
|---|---:|---|---|---|---|
| Coverage | mandatory | budgeted | 98 / 132 / 141 / 144 | 95 / 131 / 140 / 143 | 97 / 130 / 138 / 141 |
| Current state | 3,000 | unbudgeted | 1,134 / 1,729 / 1,913 / 2,002 | 1,178 / 1,834 / 2,005 / 2,118 | 1,112 / 1,691 / 1,856 / 1,949 |
| Current state | 3,000 | budgeted | 1,134 / 1,729 / 1,913 / 2,002 | 1,178 / 1,834 / 2,005 / 2,118 | 1,112 / 1,691 / 1,856 / 1,949 |
| Decisions | mandatory | budgeted | 586 / 2,345 / 2,804 / 3,510 | 620 / 2,505 / 2,966 / 3,657 | 589 / 2,359 / 2,790 / 3,478 |
| Header | mandatory | budgeted | 216 / 341 / 457 / 536 | 220 / 347 / 474 / 545 | 212 / 333 / 443 / 531 |
| Notes | 3,000 | unbudgeted | 3,838 / 55,214 / 141,827 / 282,595 | 3,624 / 52,929 / 135,988 / 274,765 | 3,618 / 52,534 / 134,295 / 269,642 |
| Notes | 3,000 | budgeted | 2,664 / 2,992 / 2,998 / 3,000 | 2,534 / 2,919 / 2,968 / 2,970 | 2,526 / 2,848 / 2,870 / 2,884 |
| Past admissions | 1,500 | unbudgeted | 240 / 4,369 / 14,757 / 30,509 | 246 / 4,412 / 14,928 / 31,180 | 224 / 4,088 / 14,045 / 29,283 |
| Past admissions | 1,500 | budgeted | 240 / 1,458 / 1,475 / 1,478 | 246 / 1,494 / 1,500 / 1,500 | 224 / 1,374 / 1,390 / 1,399 |
| Recent evidence | remainder | unbudgeted | 25,295 / 47,986 / 56,879 / 81,910 | 26,132 / 49,782 / 58,993 / 84,805 | 24,880 / 47,601 / 56,128 / 80,425 |
| Recent evidence | remainder | budgeted | 9,020 / 12,651 / 13,101 / 13,178 | 9,313 / 13,097 / 13,574 / 13,773 | 8,833 / 12,439 / 12,909 / 12,952 |
| Trends | 2,500 | unbudgeted | 975 / 1,929 / 2,498 / 2,578 | 974 / 1,933 / 2,505 / 2,606 | 961 / 1,920 / 2,477 / 2,566 |
| Trends | 2,500 | budgeted | 975 / 1,929 / 2,412 / 2,456 | 974 / 1,933 / 2,419 / 2,473 | 961 / 1,920 / 2,390 / 2,440 |
| **Total** | 16,000 | unbudgeted | 36,708 / 91,252 / 181,036 / 333,905 | 37,740 / 94,161 / 176,132 / 327,366 | 36,048 / 89,542 / 172,414 / 319,264 |
| **Total** | 16,000 | budgeted | 15,586 / 15,750 / 15,820 / 15,855 | 15,963 / 15,993 / 15,999 / 16,000 | 15,221 / 15,344 / 15,387 / 15,414 |

| Optional section | Windows hitting quota | Items available | Selected | Omitted |
|---|---:|---|---|---|
| Current state | 0 | 35 / 55 / 69 / 74 | 35 / 55 / 69 / 74 | 0 / 0 / 0 / 0 |
| Notes | 97 | 7 / 72 / 138 / 185 | 3 / 9 / 12 / 14 | 0 / 70 / 135 / 182 |
| Past admissions | 31 | 9 / 174 / 577 / 1,282 | 9 / 57 / 59 / 60 | 0 / 123 / 528 / 1,234 |
| Recent evidence | 169 | 824 / 1,595 / 1,865 / 2,616 | 284 / 427 / 456 / 491 | 526 / 1,277 / 1,666 / 2,419 |
| Trends | 3 | 9 / 18 / 24 / 25 | 9 / 18 / 23 / 23 | 0 / 0 / 1 / 2 |

### Sections, heavy (n = 50)

Tokens median / p95 / p99 / max.

| Section | Quota | Mode | qwen27 | mistral24 | medgemma27 |
|---|---:|---|---|---|---|
| Coverage | mandatory | budgeted | 118 / 139 / 143 / 143 | 118 / 138 / 142 / 142 | 116 / 137 / 140 / 140 |
| Current state | 3,000 | unbudgeted | 1,400 / 2,335 / 2,498 / 2,604 | 1,458 / 2,455 / 2,639 / 2,761 | 1,376 / 2,260 / 2,438 / 2,546 |
| Current state | 3,000 | budgeted | 1,400 / 2,335 / 2,498 / 2,604 | 1,458 / 2,455 / 2,639 / 2,761 | 1,376 / 2,260 / 2,438 / 2,546 |
| Decisions | mandatory | budgeted | 789 / 2,597 / 2,735 / 2,750 | 872 / 2,757 / 2,933 / 2,948 | 812 / 2,609 / 2,731 / 2,742 |
| Header | mandatory | budgeted | 282 / 475 / 551 / 600 | 286 / 483 / 569 / 620 | 274 / 464 / 534 / 581 |
| Notes | 3,000 | unbudgeted | 10,514 / 44,807 / 108,863 / 161,726 | 9,808 / 42,685 / 105,109 / 156,581 | 9,736 / 42,242 / 103,249 / 153,425 |
| Notes | 3,000 | budgeted | 2,921 / 2,995 / 2,998 / 2,998 | 2,786 / 2,888 / 2,899 / 2,899 | 2,751 / 2,855 / 2,861 / 2,862 |
| Past admissions | 1,500 | unbudgeted | 13 / 4,537 / 11,347 / 17,405 | 14 / 4,548 / 11,439 / 17,521 | 9 / 4,259 / 10,786 / 16,562 |
| Past admissions | 1,500 | budgeted | 13 / 1,437 / 1,471 / 1,499 | 14 / 1,482 / 1,487 / 1,490 | 9 / 1,346 / 1,385 / 1,400 |
| Recent evidence | remainder | unbudgeted | 39,552 / 66,243 / 77,076 / 79,829 | 41,004 / 68,750 / 79,423 / 82,251 | 38,848 / 65,563 / 75,790 / 78,578 |
| Recent evidence | remainder | budgeted | 9,009 / 12,596 / 12,913 / 12,945 | 9,375 / 12,956 / 13,327 / 13,441 | 8,836 / 12,246 / 12,524 / 12,621 |
| Trends | 2,500 | unbudgeted | 1,154 / 2,091 / 2,477 / 2,546 | 1,158 / 2,118 / 2,507 / 2,574 | 1,146 / 2,072 / 2,463 / 2,529 |
| Trends | 2,500 | budgeted | 1,154 / 2,091 / 2,434 / 2,461 | 1,158 / 2,118 / 2,466 / 2,493 | 1,146 / 2,072 / 2,420 / 2,444 |
| **Total** | 16,000 | unbudgeted | 57,541 / 94,605 / 186,973 / 266,020 | 58,621 / 95,734 / 186,116 / 263,698 | 56,290 / 91,988 / 180,103 / 255,531 |
| **Total** | 16,000 | budgeted | 15,614 / 15,807 / 15,824 / 15,829 | 15,980 / 15,996 / 15,998 / 15,998 | 15,245 / 15,388 / 15,410 / 15,421 |

| Optional section | Windows hitting quota | Items available | Selected | Omitted |
|---|---:|---|---|---|
| Current state | 0 | 37 / 64 / 78 / 83 | 37 / 64 / 78 / 83 | 0 / 0 / 0 / 0 |
| Notes | 36 | 28 / 69 / 113 / 152 | 3 / 13 / 16 / 16 | 20 / 65 / 111 / 149 |
| Past admissions | 5 | 0 / 174 / 456 / 700 | 0 / 49 / 56 / 62 | 0 / 125 / 408 / 653 |
| Recent evidence | 50 | 1,302 / 2,166 / 2,364 / 2,433 | 272 / 385 / 391 / 391 | 980 / 1,960 / 2,221 / 2,306 |
| Trends | 1 | 11 / 20 / 23 / 24 | 11 / 20 / 23 / 23 | 0 / 0 / 1 / 1 |

### Unbudgeted tokens per domain (random, current + past)

median / p95 / max.

| Domain | qwen27 | mistral24 | medgemma27 |
|---|---|---|---|
| Charted observations | 58,510 / 931,053 / 4,647,963 | 60,280 / 956,527 / 4,780,559 | 57,732 / 918,164 / 4,595,576 |
| Labs | 17,668 / 217,498 / 660,277 | 18,078 / 221,751 / 672,752 | 17,618 / 216,961 / 658,257 |
| Medication orders | 6,388 / 62,864 / 278,200 | 6,582 / 64,413 / 285,032 | 6,291 / 61,969 / 274,046 |
| Orders | 5,192 / 45,559 / 169,254 | 5,350 / 47,127 / 174,641 | 5,055 / 44,972 / 165,392 |
| Notes | 4,285 / 65,283 / 294,884 | 4,130 / 63,379 / 287,473 | 4,096 / 62,881 / 282,195 |
| Inputs | 2,042 / 34,267 / 156,738 | 2,174 / 35,781 / 163,015 | 2,000 / 33,851 / 154,647 |
| Input ingredients | 1,787 / 28,903 / 146,386 | 1,927 / 30,883 / 156,294 | 1,777 / 28,796 / 145,941 |
| Medication administrations | 1,540 / 121,329 / 790,251 | 1,586 / 125,118 / 814,268 | 1,550 / 120,347 / 781,015 |
| Outputs | 586 / 9,194 / 61,101 | 631 / 9,731 / 64,539 | 589 / 9,205 / 61,151 |
| Encounters | 370 / 2,312 / 13,127 | 388 / 2,420 / 13,819 | 370 / 2,300 / 13,099 |
| Microbiology | 352 / 7,660 / 43,053 | 383 / 8,190 / 45,819 | 333 / 7,235 / 41,299 |
| Procedures | 330 / 2,938 / 12,125 | 344 / 3,029 / 12,493 | 324 / 2,918 / 12,036 |
| Diagnoses | 270 / 4,272 / 36,901 | 276 / 4,397 / 37,959 | 262 / 4,168 / 36,297 |
| Outpatient measurements | 120 / 4,928 / 15,382 | 122 / 5,015 / 15,684 | 121 / 4,952 / 15,526 |

### Recent evidence omitted per domain (random, budgeted)

Items, median / p95 / max.

| Domain | Omitted |
|---|---|
| Charted observations | 478 / 1,163 / 2,183 |
| Encounters | 0 / 0 / 1 |
| Inputs | 0 / 41 / 102 |
| Labs | 0 / 97 / 151 |
| Medication administrations | 0 / 28 / 50 |
| Medication orders | 0 / 72 / 153 |
| Microbiology | 0 / 1 / 4 |
| Orders | 0 / 21 / 46 |
| Outputs | 0 / 18 / 34 |
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

- Render time (random): p50 0.79 s, p95 1.24 s. `get_window()`: p50 1.26 s, p95 1.61 s.
- `documented` availability times added ~10 % to current state and trends and ~0.4 k to median unbudgeted recent evidence; recent evidence omits a median 526 records (464 before).

## Decision

Initial budgets **confirmed** and fixed in `config/rendering.toml`:

| Component | Tokens | Rationale |
|---|---:|---|
| Total | 16,000 | Mandatory content max 4.4 k; leaves ≥ 11 k for evidence; fits mistral24's 32,768 context with prompt framing and 4,096 output tokens |
| Current state | 3,000 | Max observed 2,761 (heavy): never truncated |
| Trends | 2,500 | Binding in 3 / 200 random and 1 / 50 heavy windows, at most 2 trends dropped; latest values stay in current state. Raise to 2,750 if A7 shows trend loss matters |
| Notes | 3,000 | Binding in 49 % of random windows; long-tailed (p95 53 k); shortening keeps impressions; D9 summaries address the rest |
| Past admissions | 1,500 | Binding in 16 %; breadth-first fill keeps every admission's leading codes |
| Recent evidence | remainder | ~9.3 k median; omits a median 526 records, charted observations 478 of them (lowest priority but one) |

Quotas are caps, not reservations: tokens a quota section leaves unused go to recent evidence.

Residual risks: recent evidence is budget-limited in most windows; Coverage reports omissions per domain. Budget sensitivity belongs to A7 arms.
