"""Build script: checksum abort, row-count manifest, idempotency, record order."""

import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

import build_store
from conftest import COLUMNS, FixtureData, fixture_data, write_source

SCRIPT = Path(build_store.__file__)
TABLE_COUNT = 35


def _run(source: Path, dest: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--source", str(source), "--dest", str(dest), *extra],
        capture_output=True, text=True,
    )


@pytest.fixture
def source(tmp_path) -> Path:
    path = tmp_path / "source"
    write_source(path, fixture_data())
    return path


def test_checksum_mismatch_aborts(source, tmp_path):
    """protects=criterion 1 corrupted source never converted;
    fails_when=build exits 0 or writes Parquet after a SHA256 mismatch;
    no_claim=detection of corruption that preserves the hash."""
    target = source / "hosp/labevents.csv.gz"
    target.write_bytes(target.read_bytes() + b"\0")

    result = _run(source, tmp_path / "store")

    assert result.returncode != 0
    assert "hosp/labevents.csv.gz" in result.stderr
    assert not list((tmp_path / "store").rglob("*.parquet"))


def test_missing_checksum_entry_aborts_unless_supplied(source, tmp_path):
    """protects=files absent from SHA256SUMS (MIMIC-IV-Note) are not trusted silently;
    fails_when=missing entry treated as ok, or --checksums file ignored;
    no_claim=authenticity of the supplied checksum file."""
    sums = source / build_store.CHECKSUM_FILE
    lines = sums.read_text().splitlines()
    note_lines = [line for line in lines if " note/" in line]
    sums.write_text("\n".join(line for line in lines if line not in note_lines) + "\n")

    assert _run(source, tmp_path / "store").returncode != 0

    extra = tmp_path / "note_sums.txt"
    extra.write_text("\n".join(note_lines) + "\n")
    assert _run(source, tmp_path / "store", "--checksums", str(extra)).returncode == 0


def test_manifest_row_counts_match_source(source, tmp_path):
    """protects=criterion 2 every source record lands in Parquet;
    fails_when=a table is missing from the manifest or counts differ from rows written;
    no_claim=real MIMIC counts (checked on the real build)."""
    data: FixtureData = fixture_data()
    manifest = build_store.build(source, tmp_path / "store", [])

    assert len(manifest) == TABLE_COUNT
    for table in COLUMNS:
        expected = len(data.rows.get(table, []))
        assert manifest[table]["source_rows"] == manifest[table]["parquet_rows"] == expected, table


def test_rebuild_skips_valid_tables(source, tmp_path):
    """protects=idempotent rebuild: unchanged tables not rewritten;
    fails_when=second run reconverts or alters a Parquet file;
    no_claim=partial-file recovery (tmp + rename)."""
    dest = tmp_path / "store"
    build_store.build(source, dest, [])
    mtimes = {p: p.stat().st_mtime_ns for p in dest.rglob("*.parquet")}

    result = _run(source, dest)

    assert result.returncode == 0
    assert result.stdout.count("skip ") == TABLE_COUNT
    assert {p: p.stat().st_mtime_ns for p in dest.rglob("*.parquet")} == mtimes


def test_row_index_is_source_order_and_file_sorted_by_subject(tmp_path):
    """protects=`_row` = 0-based record index (stable audit key), Parquet sorted by subject_id;
    fails_when=row_number follows sorted or parallel scan order, or multi-line text shifts indices;
    no_claim=stability across DuckDB versions."""
    data = FixtureData()
    for subject_id, text in [(3, "a"), (1, "line1\nline2"), (2, "b"), (1, "c")]:
        data.add("note/radiology", note_id=f"n{len(data.rows['note/radiology'])}", subject_id=subject_id,
                 hadm_id=1, note_type="RR", note_seq=1, text=text)
    write_source(tmp_path / "source", data)
    build_store.build(tmp_path / "source", tmp_path / "store", [])

    rows = duckdb.sql(f"SELECT _row, subject_id, note_id FROM '{tmp_path}/store/note/radiology.parquet'").fetchall()

    assert rows == [(1, 1, "n1"), (3, 1, "n3"), (2, 2, "n2"), (0, 3, "n0")]
    manifest = json.loads((tmp_path / "store" / build_store.MANIFEST_FILE).read_text())
    assert manifest["note/radiology"]["source_rows"] == 4


def test_free_text_tables_use_small_row_groups(tmp_path):
    """protects=one subject's note read decompresses few notes (latency criterion 13);
    fails_when=note text written in default 122,880-row groups;
    no_claim=actual latency (measured by the real_data latency tests)."""
    rows = build_store.FREE_TEXT_ROW_GROUP_SIZE + 1
    data = FixtureData()
    for i in range(rows):
        data.add("note/discharge", note_id=f"n{i}", subject_id=i, hadm_id=1, note_type="DS", note_seq=1, text="t")
    write_source(tmp_path / "source", data)
    build_store.build(tmp_path / "source", tmp_path / "store", [])

    sizes = duckdb.sql(
        f"SELECT DISTINCT row_group_id, row_group_num_rows "
        f"FROM parquet_metadata('{tmp_path}/store/note/discharge.parquet')"
    ).fetchall()

    assert max(size for _, size in sizes) <= build_store.FREE_TEXT_ROW_GROUP_SIZE
    assert sum(size for _, size in sizes) == rows
