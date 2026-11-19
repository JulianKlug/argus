"""Descriptive statistics of core variable series: count, first, min, max, change. No interpretation.

    observations of one variable in (at - lookback, at], numeric, not pending
        └─ split by unit (no conversion) ─► series with >= min_points ─► Trend

Ties (equal min or max values) resolve to the earliest point, then source row.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from argus.data.domains import AVAILABLE_TIME, EVENT_TIME, IS_PENDING, SOURCE_ROW, SOURCE_TABLE
from argus.rendering.types import SourceRow

VALUE_COLUMN = "valuenum"
UNIT_COLUMN = "valueuom"
NO_UNIT = ""


@dataclass(frozen=True)
class Point:
    time: pd.Timestamp
    value: float
    source: SourceRow
    known: pd.Timestamp | None = None  # available_time: when the value became known


@dataclass(frozen=True)
class Trend:
    unit: str
    points: tuple[Point, ...]  # chronological

    @property
    def first(self) -> Point:
        return self.points[0]

    @property
    def latest(self) -> Point:
        return self.points[-1]

    @property
    def minimum(self) -> Point:
        return min(self.points, key=lambda p: (p.value, p.time, p.source))

    @property
    def maximum(self) -> Point:
        return min(self.points, key=lambda p: (-p.value, p.time, p.source))

    @property
    def change(self) -> float:
        return self.latest.value - self.first.value

    @property
    def sources(self) -> tuple[SourceRow, ...]:
        return tuple(p.source for p in self.points)


def trends(observations: pd.DataFrame, at: datetime, lookback: timedelta, min_points: int) -> list[Trend]:
    """Series of one variable's rows, one per unit, ordered by unit; series below `min_points` dropped."""
    at = pd.Timestamp(at)
    rows = observations[
        (observations[EVENT_TIME] > at - lookback)
        & (observations[EVENT_TIME] <= at)
        & ~observations[IS_PENDING].astype(bool)
    ]
    values = pd.to_numeric(rows[VALUE_COLUMN], errors="coerce") if VALUE_COLUMN in rows else pd.Series(dtype=float)
    rows = rows.assign(_value=values)
    rows = rows[rows["_value"].notna()]

    units = rows[UNIT_COLUMN].astype(object).where(rows[UNIT_COLUMN].notna(), NO_UNIT) if UNIT_COLUMN in rows else NO_UNIT
    rows = rows.assign(_unit=units).sort_values([EVENT_TIME, SOURCE_TABLE, SOURCE_ROW])

    result = []
    for unit in sorted(rows["_unit"].unique()):
        series = rows[rows["_unit"] == unit]
        if len(series) < min_points:
            continue
        known = series[AVAILABLE_TIME] if AVAILABLE_TIME in series else [None] * len(series)
        points = tuple(
            Point(pd.Timestamp(t), float(v), SourceRow(table, int(row)), None if pd.isna(k) else pd.Timestamp(k))
            for t, v, table, row, k in zip(
                series[EVENT_TIME], series["_value"], series[SOURCE_TABLE], series[SOURCE_ROW], known,
            )
        )
        result.append(Trend(str(unit), points))
    return result
