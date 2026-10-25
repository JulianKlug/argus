"""Real MIMIC-IV store: leak properties, monotonicity, split, latency (criteria 10-13).

Run: ARGUS_STORE_DIR=/mnt/data1/klug/argus/mimiciv_3.1 uv run pytest -m real_data -s tests/data
"""

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from argus.data import AmbiguousEncounterError, Domain, NotInIcuStayError, PatientSnapshot, SnapshotEngine
from argus.data.domains import AVAILABLE_TIME, EVENT_TIME, IS_PENDING, IS_TRUNCATED, SOURCE_ROW, SOURCE_TABLE

STORE_ENV = "ARGUS_STORE_DIR"
SEED = 20261023
SAMPLE_SIZE = 200
MONOTONIC_SUBJECTS = 50
MONOTONIC_STEPS = 5
WARMUP_CALLS = 5
LATENCY_P95_SECONDS = 2.0
MAX_REJECTED_SHARE = 0.05  # overlapping admissions / ICU outside admission

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not os.environ.get(STORE_ENV), reason=f"{STORE_ENV} unset"),
]


@dataclass(frozen=True)
class _Stay:
    subject_id: int
    intime: pd.Timestamp
    outtime: pd.Timestamp


def _stays(store_dir: Path) -> list[_Stay]:
    """All ICU stays, in stay_id order (deterministic sampling)."""
    rows = duckdb.sql(
        f"SELECT subject_id, intime, outtime FROM '{store_dir}/icu/icustays.parquet' "
        f"WHERE outtime > intime ORDER BY stay_id"
    ).fetchall()
    return [_Stay(s, pd.Timestamp(i), pd.Timestamp(o)) for s, i, o in rows]


def _instant(stay: _Stay, fraction: float) -> pd.Timestamp:
    """Minute-rounded instant at `fraction` of the stay, kept strictly before outtime."""
    at = (stay.intime + fraction * (stay.outtime - stay.intime)).floor("min")
    return max(at, stay.intime)


def _all_rows(snapshot: PatientSnapshot) -> pd.DataFrame:
    frames = [record.get(d) for record in (snapshot.current, snapshot.past) for d in Domain]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def store_dir() -> Path:
    return Path(os.environ[STORE_ENV])


@pytest.fixture(scope="module")
def engine(store_dir) -> SnapshotEngine:
    return SnapshotEngine(store_dir)


@dataclass(frozen=True)
class _Sample:
    snapshots: list[PatientSnapshot]
    seconds: list[float]
    rejected: int


@pytest.fixture(scope="module")
def sample(engine, store_dir) -> _Sample:
    """SAMPLE_SIZE seeded (subject, at) inside ICU stays, timed after warm-up."""
    rng = random.Random(SEED)
    stays = _stays(store_dir)

    for stay in rng.sample(stays, WARMUP_CALLS):
        try:
            engine.get_snapshot(stay.subject_id, _instant(stay, 0.5).to_pydatetime())
        except (AmbiguousEncounterError, NotInIcuStayError):
            pass

    snapshots, seconds, rejected = [], [], 0
    while len(snapshots) < SAMPLE_SIZE:
        stay = rng.choice(stays)
        at = _instant(stay, rng.random()).to_pydatetime()
        start = time.perf_counter()
        try:
            snapshot = engine.get_snapshot(stay.subject_id, at)
        except (AmbiguousEncounterError, NotInIcuStayError):
            rejected += 1
            continue
        seconds.append(time.perf_counter() - start)
        snapshots.append(snapshot)
    return _Sample(snapshots, seconds, rejected)


def test_no_row_after_at(sample):
    """protects=criterion 10 on real data: available_time and event_time never after `at`;
    fails_when=any rule admits a future row for real MIMIC timing patterns;
    no_claim=field-level redaction correctness (fixture tests) or leaks inside text."""
    print(f"\nrejected (ambiguous / ICU outside admission): {sample.rejected}")
    assert sample.rejected <= MAX_REJECTED_SHARE * SAMPLE_SIZE

    for snapshot in sample.snapshots:
        rows = _all_rows(snapshot)
        at = pd.Timestamp(snapshot.at)
        late = rows[(rows[AVAILABLE_TIME] > at) | (rows[EVENT_TIME] > at)]
        assert late.empty, (snapshot.subject_id, snapshot.at, late[[SOURCE_TABLE, SOURCE_ROW]].head())


def test_current_past_split(sample):
    """protects=criterion 12 partition on current admittime;
    fails_when=a current row predates admission or a past row follows it;
    no_claim=hadm_id consistency of rows."""
    for snapshot in sample.snapshots:
        admittime = pd.Timestamp(snapshot.current_encounter.admittime)
        for domain in Domain:
            assert (snapshot.current.get(domain)[EVENT_TIME] >= admittime).all(), (snapshot.subject_id, domain)
            assert (snapshot.past.get(domain)[EVENT_TIME] < admittime).all(), (snapshot.subject_id, domain)


def test_latency_p95(sample):
    """protects=criterion 13 p95 get_snapshot <= 2 s warm;
    fails_when=per-subject reads stop pruning (unsorted store, full scans);
    no_claim=cold-cache latency or concurrent load."""
    p50, p95, worst = np.percentile(sample.seconds, [50, 95, 100])
    print(f"\nlatency n={len(sample.seconds)} p50={p50:.3f}s p95={p95:.3f}s max={worst:.3f}s")
    assert p95 <= LATENCY_P95_SECONDS


def test_monotonic_in_at(engine, store_dir):
    """protects=criterion 11: advancing `at` never removes rows, settled rows never change;
    fails_when=a row disappears later (non-causal filter) or a settled value changes (unflagged redaction);
    no_claim=correctness of the values themselves."""
    rng = random.Random(SEED + 1)
    stays = _stays(store_dir)
    checked = 0
    while checked < MONOTONIC_SUBJECTS:
        stay = rng.choice(stays)
        fractions = sorted(rng.random() for _ in range(MONOTONIC_STEPS))
        try:
            snapshots = [engine.get_snapshot(stay.subject_id, _instant(stay, f).to_pydatetime()) for f in fractions]
        except (AmbiguousEncounterError, NotInIcuStayError):
            continue

        frames = [_all_rows(s).set_index([SOURCE_TABLE, SOURCE_ROW]) for s in snapshots]
        for earlier, later in zip(frames, frames[1:]):
            assert set(earlier.index) <= set(later.index), stay

            settled = earlier[~earlier[IS_PENDING] & ~earlier[IS_TRUNCATED]]
            columns = [c for c in settled.columns if c in later.columns]
            pd.testing.assert_frame_equal(
                settled[columns].sort_index(), later.loc[settled.index, columns].sort_index(), check_dtype=False,
            )
        checked += 1
