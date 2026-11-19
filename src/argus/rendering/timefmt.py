"""Clinical times relative to the window end `at`. MIMIC dates are shifted: never render a calendar date.

    at - 2h15m        -> "T-02:15"
    at - 3d 4h15m     -> "T-3d 04:15"
    at                -> "T-00:00"
    at + 30m          -> "T+00:30"   (e.g. a charted planned time)

Offsets are floored to the minute, so the same instant always yields the same text.
"""

from datetime import datetime, timedelta

import pandas as pd

MINUTE = timedelta(minutes=1)
MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 24 * MINUTES_PER_HOUR
BEFORE = "-"
AFTER = "+"


def relative(time: datetime | pd.Timestamp, at: datetime | pd.Timestamp) -> str:
    """'T-02:15' style offset of `time` from `at`."""
    offset = pd.Timestamp(at) - pd.Timestamp(time)
    sign = BEFORE if offset >= timedelta(0) else AFTER
    return f"T{sign}{duration(abs(offset))}"


def duration(span: timedelta | pd.Timedelta) -> str:
    """'02:15' below one day, '3d 04:15' from one day on; floored to the minute."""
    if span < timedelta(0):
        raise ValueError("duration must be non-negative")

    minutes = pd.Timedelta(span) // MINUTE
    days, rest = divmod(minutes, MINUTES_PER_DAY)
    hours, minutes = divmod(rest, MINUTES_PER_HOUR)
    clock = f"{hours:02d}:{minutes:02d}"
    return f"{days}d {clock}" if days else clock


def day_number(start: datetime | pd.Timestamp, at: datetime | pd.Timestamp) -> int:
    """Day 1 = first 24 h after `start`."""
    return (pd.Timestamp(at) - pd.Timestamp(start)) // timedelta(days=1) + 1
