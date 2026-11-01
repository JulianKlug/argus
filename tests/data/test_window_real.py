"""Real MIMIC-IV store: catalog coverage, decision window properties, latency (spec criteria 9-10).

Run: ARGUS_STORE_DIR=/mnt/data1/klug/argus/mimiciv_3.1 uv run pytest -m real_data -s tests/data/test_window_real.py
"""

import random
import time
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import pytest

from argus.data import AmbiguousEncounterError, DecisionCategory, DecisionWindow, NotInIcuStayError
from argus.data.catalog import POE_KEY_SEPARATOR, DecisionCatalog, Source
from argus.data.domains import EVENT_TIME, ORDER_ROW, SOURCE_ROW, SOURCE_TABLE
from test_engine_real import (
    LATENCY_P95_SECONDS, MAX_REJECTED_SHARE, SAMPLE_SIZE, SEED, WARMUP_CALLS, _all_rows, _instant, _stays,
    engine, pytestmark, store_dir,  # noqa: F401 (fixtures and markers reused)
)

# Observed keys per catalog source: SQL over the store returning one text column.
OBSERVED_KEYS = {
    Source.D_ITEMS: [
        "SELECT DISTINCT itemid::VARCHAR FROM '{store}/icu/chartevents.parquet'",
        "SELECT DISTINCT itemid::VARCHAR FROM '{store}/icu/inputevents.parquet'",
        "SELECT DISTINCT itemid::VARCHAR FROM '{store}/icu/procedureevents.parquet'",
    ],
    Source.INPUT_ORDER_CATEGORY: [
        "SELECT DISTINCT ordercategoryname FROM '{store}/icu/inputevents.parquet' WHERE ordercategoryname IS NOT NULL",
    ],
    Source.POE: [
        f"SELECT DISTINCT order_type || '{POE_KEY_SEPARATOR}' || coalesce(order_subtype, '') "
        "FROM '{store}/hosp/poe.parquet'",
    ],
}


@pytest.mark.parametrize("source", list(Source))
def test_catalog_covers_every_observed_key(store_dir, source):
    """protects=spec criterion 9: no item, order type or drug silently read as 'not a decision' by omission;
    fails_when=store holds a key absent from decision_catalog.csv (new MIMIC version, edited catalog);
    no_claim=correct category of each key (clinician review)."""
    observed = set()
    for sql in OBSERVED_KEYS[source]:
        observed |= {row[0] for row in duckdb.sql(sql.format(store=store_dir)).fetchall()}

    missing = observed - DecisionCatalog().keys(source)

    assert not missing, sorted(missing)[:20]


@dataclass(frozen=True)
class _WindowSample:
    windows: list[DecisionWindow]
    seconds: list[float]
    rejected: int


@pytest.fixture(scope="module")
def window_sample(engine, store_dir) -> _WindowSample:  # noqa: F811
    """SAMPLE_SIZE seeded (subject, at) inside ICU stays, get_window timed after warm-up."""
    rng = random.Random(SEED + 2)
    stays = _stays(store_dir)

    for stay in rng.sample(stays, WARMUP_CALLS):
        try:
            engine.get_window(stay.subject_id, _instant(stay, 0.5).to_pydatetime())
        except (AmbiguousEncounterError, NotInIcuStayError):
            pass

    windows, seconds, rejected = [], [], 0
    while len(windows) < SAMPLE_SIZE:
        stay = rng.choice(stays)
        at = _instant(stay, rng.random()).to_pydatetime()
        start = time.perf_counter()
        try:
            window = engine.get_window(stay.subject_id, at)
        except (AmbiguousEncounterError, NotInIcuStayError):
            rejected += 1
            continue
        seconds.append(time.perf_counter() - start)
        windows.append(window)
    return _WindowSample(windows, seconds, rejected)


def test_decisions_are_visible_rows_inside_window(window_sample):
    """protects=spec criterion 10: every decision row is a snapshot row and start < event_time <= at;
    fails_when=a rule derives a decision from a hidden row or a time outside the window on real data;
    no_claim=completeness of decisions (catalog review) or link precision."""
    assert window_sample.rejected <= MAX_REJECTED_SHARE * SAMPLE_SIZE

    counts = {category: 0 for category in DecisionCategory}
    for window in window_sample.windows:
        rows = _all_rows(window.snapshot)
        visible = set(zip(rows[SOURCE_TABLE], rows[SOURCE_ROW]))
        start, at = pd.Timestamp(window.start), pd.Timestamp(window.snapshot.at)

        for category in DecisionCategory:
            frame = window.get(category)
            counts[category] += len(frame)
            keys = set(zip(frame[SOURCE_TABLE], frame[SOURCE_ROW]))
            assert keys <= visible, (window.snapshot.subject_id, window.snapshot.at, category)
            inside = (frame[EVENT_TIME] > start) & (frame[EVENT_TIME] <= at)
            assert inside.all(), (window.snapshot.subject_id, window.snapshot.at, category)

    print(f"\ndecisions over {SAMPLE_SIZE} windows: " + ", ".join(f"{c.name}={n}" for c, n in counts.items()))


def test_infusion_link_rate(window_sample):
    """protects=visibility of the prescription link quality on real data (reported, threshold-free);
    fails_when=no infusion decision is ever linked (link rule broken end to end);
    no_claim=link precision (needs clinician sample review)."""
    frames = [w.get(DecisionCategory.MEDICATION) for w in window_sample.windows]
    medication = pd.concat(frames, ignore_index=True)
    infusions = medication[medication[SOURCE_TABLE] == "inputevents"]

    linked = infusions[ORDER_ROW].notna().mean()
    print(f"\ninfusion decisions: {len(infusions)}, linked to a prescription: {linked:.1%}")
    assert linked > 0


def test_window_latency_p95(window_sample):
    """protects=get_window p95 <= 3 s warm (snapshot criterion 13 budget, decisions included);
    fails_when=decision rules add enough per-call work to exceed the budget;
    no_claim=cold-cache latency or concurrent load."""
    p50, p95, worst = np.percentile(window_sample.seconds, [50, 95, 100])
    print(f"\nget_window latency n={len(window_sample.seconds)} p50={p50:.3f}s p95={p95:.3f}s max={worst:.3f}s")
    assert p95 <= LATENCY_P95_SECONDS
