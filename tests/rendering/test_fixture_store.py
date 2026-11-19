"""D1 + D2 + D5 over the synthetic fixture store: planted future rows never reach the text (acceptance 3)."""

import importlib.util
import re
from pathlib import Path

import pytest

from argus.data import SnapshotEngine
from argus.rendering import Renderer
from window_factory import COUNTERS, config

_DATA_FIXTURES = Path(__file__).resolve().parents[1] / "data" / "conftest.py"


def _fixtures():
    """tests/data/conftest.py under its own module name (both test dirs have a conftest)."""
    spec = importlib.util.spec_from_file_location("data_fixtures", _DATA_FIXTURES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fixture_window(tmp_path_factory):
    fixtures = _fixtures()
    root = tmp_path_factory.mktemp("mimic")
    data = fixtures.fixture_data()
    fixtures.write_source(root / "source", data)
    fixtures.build_store.build(root / "source", root / "store", [])
    window = SnapshotEngine(root / "store").get_window(fixtures.SUBJECT, fixtures.AT.to_pydatetime(), fixtures.WINDOW)
    return window, data.marked[fixtures.Kind.SENTINEL], data.marked[fixtures.Kind.BOUNDARY]


def test_future_rows_never_rendered(fixture_window):
    """protects=acceptance 3: every rendered fact derives from the snapshot; sentinel rows (available after `at`)
    are never cited, boundary rows (available at `at`) are; fails_when=renderer reads anything but the window;
    no_claim=D1 visibility rules themselves (tests/data)."""
    window, sentinels, boundaries = fixture_window
    cfg = config()
    result = Renderer(cfg, {name: COUNTERS[name] for name in cfg.tokenizers}).render(window)

    cited = {(s.table, s.row) for ref in result.refs.values() for s in ref.sources}
    assert not cited & sentinels
    assert cited & boundaries
    assert "CVICU" not in result.text  # sentinel ICU stay
    assert not re.search(r"\b2\d{3}-\d{2}-\d{2}", result.text)
