"""Renderer on real MIMIC-IV windows with stand-in counters: structure, budget, determinism on real text.

Run: ARGUS_STORE_DIR=/mnt/data1/klug/argus/mimiciv_3.1 uv run pytest -m real_data tests/rendering
Assertion messages carry counts and headings only, never patient text.
"""

import os
import random
import re
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from argus.data import AmbiguousEncounterError, NotInIcuStayError, SnapshotEngine
from argus.rendering import Renderer, RenderingBudgetError, Section
from argus.rendering.types import OPTIONAL_SECTIONS
from window_factory import COUNTERS, config

STORE_ENV = "ARGUS_STORE_DIR"
SEED = 20261117
WINDOWS = 20
HEADINGS = 8

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not os.environ.get(STORE_ENV), reason=f"{STORE_ENV} unset"),
]


@pytest.fixture(scope="module")
def windows():
    store = Path(os.environ[STORE_ENV])
    engine = SnapshotEngine(store)
    stays = duckdb.sql(
        f"SELECT subject_id, intime, outtime FROM '{store}/icu/icustays.parquet' WHERE outtime > intime ORDER BY stay_id"
    ).fetchall()
    rng = random.Random(SEED)

    result = []
    while len(result) < WINDOWS:
        subject, intime, outtime = rng.choice(stays)
        at = (pd.Timestamp(intime) + rng.random() * (pd.Timestamp(outtime) - pd.Timestamp(intime))).floor("min")
        try:
            result.append(engine.get_window(subject, max(at, pd.Timestamp(intime)).to_pydatetime()))
        except (AmbiguousEncounterError, NotInIcuStayError):
            continue
    return result


def test_real_windows_structure_budget_determinism(windows):
    """protects=acceptance 1, 11, 12, 15 on real text: eight headings only, no calendar date, budget kept
    under every counter, identical output on a second render; fails_when=real free text breaks structure or
    a real-only column leaks a timestamp; no_claim=token counts of the real tokenizers (measurement script)."""
    cfg = config()
    renderer = Renderer(cfg, {name: COUNTERS[name] for name in cfg.tokenizers})

    for index, window in enumerate(windows):
        try:
            result = renderer.render(window)
        except RenderingBudgetError:
            continue  # reported by the measurement script; fail-closed is the expected behaviour

        headings = [line for line in result.text.split("\n") if line.startswith("===")]
        assert len(headings) == HEADINGS, f"window {index}: {len(headings)} headings"
        assert not re.search(r"\b2\d{3}-\d{2}-\d{2}", result.text), f"window {index}: calendar date"
        assert max(result.total_tokens.values()) <= cfg.budget.total, f"window {index}: over total"
        for section in OPTIONAL_SECTIONS:
            assert result.omissions[section] >= 0
        assert renderer.render(window) == result, f"window {index}: not deterministic"
        assert set(result.section_tokens) == set(Section)
