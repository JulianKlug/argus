"""Unit tests: relative time, escaping, reference IDs, config validation, line formats, trends."""

import re
from datetime import timedelta

import pandas as pd
import pytest

from argus.data import DecisionAction, DecisionCategory, Domain
from argus.rendering import RefKind, Renderer, RenderingConfigError, RenderingError, SourceRow
from argus.rendering import escape, format, references, timefmt
from argus.rendering.config import load_config
from argus.rendering.trends import trends
from window_factory import AT, COUNTERS, HEART_RATE, START, TOKENIZERS, config, decision, frame, row

H, M = timedelta(hours=1), timedelta(minutes=1)


# ── timefmt ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("offset", "expected"), [
    (timedelta(0), "T-00:00"),
    (2 * H + 15 * M, "T-02:15"),
    (23 * H + 59 * M, "T-23:59"),
    (24 * H, "T-1d 00:00"),
    (3 * 24 * H + 4 * H + 15 * M, "T-3d 04:15"),
    (2 * H + 15 * M + timedelta(seconds=59), "T-02:15"),  # floored
    (-30 * M, "T+00:30"),
])
def test_relative_time(offset, expected):
    """protects=time representation table; fails_when=wrong unit switch at 24 h, rounding up, sign;
    no_claim=clinical meaning of the anchor."""
    assert timefmt.relative(AT - offset, AT) == expected


def test_duration_and_day_number():
    """protects=durations and ICU/hospital day (day 1 = first 24 h); fails_when=off-by-one day;
    no_claim=calendar correctness (dates are shifted)."""
    assert timefmt.duration(3 * 24 * H + 2 * H) == "3d 02:00"
    assert timefmt.day_number(AT - 23 * H, AT) == 1
    assert timefmt.day_number(AT - 24 * H, AT) == 2
    with pytest.raises(ValueError):
        timefmt.duration(-H)


# ── escape ────────────────────────────────────────────────────────────────


def test_escape_controls_and_line_endings():
    """protects=escaping rule 1; fails_when=CR, NUL, ESC or Unicode line separators survive;
    no_claim=model behaviour on the text."""
    assert escape.clean("a\r\nb\rc\x00d\x1be f\tg") == "a\nb\nc" + "de\nf\tg"


@pytest.mark.parametrize("text", ["=== DECISIONS ===", "x ==== y", "======"])
def test_escape_inline_never_keeps_marker(text):
    """protects=rule 3; fails_when=a single-line value still contains '==='; no_claim=readability."""
    inline, block = escape.split(text)
    assert block == [] and "===" not in inline


def test_escape_multiline_becomes_prefixed_block():
    """protects=rule 2; fails_when=a multi-line value yields a line not starting with '| ';
    no_claim=layout of long notes."""
    inline, block = escape.split("ok\n=== DECISIONS ===\n")
    assert inline == "" and block == ["| ok", "| === DECISIONS ==="]


# ── references ────────────────────────────────────────────────────────────


def test_reference_ids_stable_and_distinct():
    """protects=ID derivation: stable across windows, one decision ID per occurrence of a row;
    fails_when=IDs depend on call order, or start/stop of one row share an ID; no_claim=collision freedom."""
    source = SourceRow("procedureevents", 7)
    assert references.record_id(source) == references.record_id(SourceRow("procedureevents", 7))
    assert re.fullmatch(r"E[A-Z2-7]{8}", references.record_id(source))

    start = references.decision_id(source, DecisionCategory.PROCEDURE, DecisionAction.START, AT - H)
    stop = references.decision_id(source, DecisionCategory.PROCEDURE, DecisionAction.STOP, AT)
    assert start != stop and start.startswith("D")


def test_reference_collision_raises(monkeypatch):
    """protects=collisions never resolved silently; fails_when=two keys share an ID without error;
    no_claim=collision rate (measurement script)."""
    monkeypatch.setattr(references, "_digest", lambda key: "AAAAAAAA")
    refs = references.References()
    refs.record(SourceRow("labevents", 1))
    refs.record(SourceRow("labevents", 1))  # same key: citation, fine

    with pytest.raises(RenderingError):
        refs.record(SourceRow("labevents", 2))


def test_decision_collision_detected_across_times(monkeypatch):
    """protects=decision ref key includes event time; fails_when=two occurrences of one row, category and action
    at different times share an ID without error."""
    monkeypatch.setattr(references, "_digest", lambda key: "AAAAAAAA")
    refs = references.References()
    source = SourceRow("chartevents", 1)
    refs.decision(source, DecisionCategory.DEVICE_SETTING, DecisionAction.CHANGE, AT - H)

    with pytest.raises(RenderingError):
        refs.decision(source, DecisionCategory.DEVICE_SETTING, DecisionAction.CHANGE, AT)


def test_trend_refs_list_contributors_in_order():
    """protects=trend IDs window-local and sequential, contributors complete; fails_when=IDs skip or
    sources lost; no_claim=statistics."""
    refs = references.References()
    sources = (SourceRow("chartevents", 3), SourceRow("chartevents", 1))
    assert refs.trend(sources) == "T1"
    assert refs.trend(sources[:1]) == "T2"
    assert refs.used_in("[T1] x")["T1"].sources == tuple(sorted(sources))
    assert refs.used_in("[T1] x")["T1"].kind is RefKind.TREND


# ── config ────────────────────────────────────────────────────────────────


def test_shipped_config_loads():
    """protects=committed config/rendering.toml is valid against config/llm.toml;
    fails_when=a typo or unknown backend enters the committed file; no_claim=budget adequacy."""
    shipped = load_config()
    assert shipped.tokenizers and shipped.budget.total > 0
    assert len(shipped.digest()) == 64


@pytest.mark.parametrize("overrides", [
    {"total": -1},
    {"notes": 20000},  # quotas above total
    {"trend_min_points": True},
    {"demographics": "everything"},
    {"recent_domain_priority": ["labs", "labs"]},
    {"recent_domain_priority": ["notes"]},
    {"note_section_priority": ["impression"]},
    {"unknown_key": 1},
])
def test_invalid_config_rejected(overrides):
    """protects=strict validation; fails_when=bad values silently fall back or pass;
    no_claim=every possible malformed file."""
    with pytest.raises(RenderingConfigError):
        config(**overrides)


def test_duplicate_core_variable_and_unknown_backend_rejected():
    """protects=strict validation of core variables and tokenizer IDs; fails_when=duplicates accepted;
    no_claim=clinical choice of the list."""
    with pytest.raises(RenderingConfigError):
        config(core_variables=[{"table": "chartevents", "itemid": 1, "label": "a"}] * 2)
    with pytest.raises(RenderingConfigError):
        config(tokenizers=["missing"])


def test_config_change_changes_digest():
    """protects=config_digest identifies the condition; fails_when=a budget change keeps the digest;
    no_claim=hash strength."""
    assert config().digest() != config(total=15000).digest()
    assert config().digest() == config().digest()


def test_renderer_requires_exact_tokenizer_set():
    """protects=acceptance 2 (fixed tokenizer set); fails_when=a caller can add or drop a counter;
    no_claim=counter correctness."""
    with pytest.raises(RenderingConfigError):
        Renderer(config(), {name: COUNTERS[name] for name in TOKENIZERS[:2]})
    with pytest.raises(RenderingConfigError):
        Renderer(config(), COUNTERS | {"extra": len})


# ── format ────────────────────────────────────────────────────────────────


def test_late_decision_shows_both_times():
    """protects=acceptance 5: late documentation shows event time, documented time and '(late)';
    fails_when=only one time shown or the mark missing; no_claim=correct D2 timing."""
    late = decision("chartevents", 1, START - H, DecisionAction.CHANGE, "PEEP set", value="8", valueuom="cmH2O")
    late["available_time"] = AT - 30 * M
    line = format.decision(late, "DEVICE_SETTING", "DX", None, START, AT).head
    assert line == "[DX] T-07:00 DEVICE_SETTING CHANGE PEEP set: 8 cmH2O; documented T-00:30 (late)"

    timely = decision("chartevents", 2, AT - H, DecisionAction.CHANGE, "PEEP set", value="8", valueuom="cmH2O")
    timely["available_time"] = AT - 30 * M
    assert format.decision(timely, "DEVICE_SETTING", "DY", "EZ", START, AT).head.endswith(
        "8 cmH2O; order [EZ]; documented T-00:30"
    )


def test_record_line_renders_timestamps_relative():
    """protects=acceptance 11: no shifted calendar date, also in datetimeevents values and stop times;
    fails_when=a raw timestamp string reaches the line; no_claim=all free text is date-free."""
    charted = row("datetimeevents", 1, AT - H, "Last dose", value="2180-03-13 10:00:00")
    rx = row("prescriptions", 2, AT - H, "Heparin", dose_val_rx="5000", dose_unit_rx="UNIT", stoptime=AT - 10 * M)
    text = format.record(charted, "E1", AT, H).head + format.record(rx, "E2", AT, H).head
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
    assert "Last dose: T-1d 02:00" in text and "until T-00:10" in text


def test_numbers_have_no_exponent_or_float_noise():
    """protects=readable values; fails_when='1e+06' or '199.99999' appears; no_claim=unit handling."""
    assert format.number(0.08) == "0.08"
    assert format.number(199.9999999) == "200"
    assert format.number(7.0) == "7"
    assert "e" not in format.number(1234567.0)


def test_note_sections_and_shortening_order():
    """protects=shortening at explicit sections, priority order, text order kept;
    fails_when=a section is cut mid-way or kept sections reorder; no_claim=section detection on every report style."""
    text = "EXAM: CT\nINDICATION: fever\nFINDINGS: clear\nmore\nIMPRESSION: normal"
    sections = format.note_sections(text, ("IMPRESSION", "FINDINGS"))
    assert [s.name for s in sections] == ["EXAM", "INDICATION", "FINDINGS", "IMPRESSION"]
    assert sections[2].text == "FINDINGS: clear\nmore"

    candidates = format.shortened_sections(sections, ("IMPRESSION", "FINDINGS"))
    assert [[s.name for s in c] for c in candidates] == [["IMPRESSION"], ["FINDINGS", "IMPRESSION"]]


# ── trends ────────────────────────────────────────────────────────────────


def _hr(source_row: int, offset: timedelta, value, unit="bpm", pending=False) -> dict:
    return row("chartevents", source_row, AT - offset, "Heart Rate", itemid=HEART_RATE,
               value=None if value is None else str(value), valuenum=value, valueuom=unit, is_pending=pending)


def test_trend_statistics():
    """protects=acceptance 9: count, first, min, max, change, ties to earliest;
    fails_when=wrong extremum, change sign, or lookback boundary included; no_claim=clinical relevance."""
    rows = frame([
        _hr(1, 30 * H, 50),  # outside 24 h lookback
        _hr(2, 20 * H, 100), _hr(3, 10 * H, 80), _hr(4, 8 * H, 120), _hr(5, 5 * H, 80), _hr(6, H, 110),
    ])
    (trend,) = trends(rows, AT, 24 * H, 3)
    assert len(trend.points) == 5
    assert (trend.first.value, trend.minimum.value, trend.maximum.value, trend.change) == (100, 80, 120, 10)
    assert trend.minimum.source == SourceRow("chartevents", 3)  # tie: earliest


def test_trend_units_nulls_pending_min_points():
    """protects=acceptance 9: mixed units split, null / non-numeric / pending excluded, min points enforced;
    fails_when=units merged or converted, a pending value counted; no_claim=unit correctness in MIMIC."""
    rows = frame([
        _hr(1, 5 * H, 90), _hr(2, 4 * H, 95), _hr(3, 3 * H, 99),
        _hr(4, 3 * H, 1.5, unit="Hz"), _hr(5, 2 * H, 1.6, unit="Hz"),  # 2 points: dropped
        _hr(6, 2 * H, None), _hr(7, H, 300, pending=True),
    ])
    rows.loc[rows["source_row"] == 3, "valuenum"] = None
    rows.loc[rows["source_row"] == 3, "value"] = "unreadable"
    assert trends(rows, AT, 24 * H, 3) == []

    (trend,) = trends(rows, AT, 24 * H, 2)[1:]  # bpm series after 'Hz'
    assert trend.unit == "bpm" and [p.value for p in trend.points] == [90, 95]


def test_domain_names_cover_every_domain():
    """protects=recent evidence subheadings exist for every configurable domain; fails_when=a new Domain
    has no display name; no_claim=wording."""
    from argus.rendering.renderer import DOMAIN_NAMES
    assert set(DOMAIN_NAMES) == set(Domain)
