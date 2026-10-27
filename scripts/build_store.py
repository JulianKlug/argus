"""One-time MIMIC-IV csv.gz -> Parquet conversion (spec D9).

    source/<module>/<table>.csv.gz ──sha256 check──► DuckDB ──► dest/<module>/<table>.parquet
                                                               dest/manifest.json

Each Parquet file gets `_row` (0-based record index in the source CSV) and is
sorted by `subject_id` so per-subject reads prune row groups.

Usage:
    uv run python scripts/build_store.py --source <mimic dir> --dest <store dir> \
        [--checksums <extra SHA256SUMS.txt> ...]
"""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import duckdb

MODULES = ("hosp", "icu", "note")
CSV_SUFFIX = ".csv.gz"
CHECKSUM_FILE = "SHA256SUMS.txt"
MANIFEST_FILE = "manifest.json"
ROW_COLUMN = "_row"
SUBJECT_COLUMN = "subject_id"
HASH_CHUNK_BYTES = 1 << 24
DUCKDB_MEMORY_LIMIT = "32GB"
TMP_SUFFIX = ".tmp"
UNSORTED_SUFFIX = ".unsorted.parquet"

EXIT_CHECKSUM_MISMATCH = 1
EXIT_ROW_COUNT_MISMATCH = 2

# Clinical and audit times; date-only columns are typed TIMESTAMP too (midnight).
TIMESTAMP_COLUMNS = frozenset({
    "admittime", "dischtime", "deathtime", "edregtime", "edouttime",
    "charttime", "storetime", "chartdate", "storedate", "scheduletime",
    "starttime", "stoptime", "endtime", "entertime", "verifiedtime", "expirationdate",
    "ordertime", "transfertime", "intime", "outtime", "dod",
})

# Codes and free text that the CSV sniffer may mistype (e.g. ICD "0010" -> 10).
TEXT_COLUMNS = frozenset({
    "icd_code", "drg_code", "hcpcs_cd", "code", "gsn", "ndc", "formulary_drug_cd",
    "value", "comments", "text", "field_value", "result_value", "flag", "priority",
    "ref_range_lower", "ref_range_upper", "quantity", "dilution_text", "dilution_value",
    "dose_val_rx", "form_val_disp", "doses_per_24_hrs", "prod_strength",
    "dose_due", "dose_given", "product_amount_given", "product_code",
    "infusion_rate", "prior_infusion_rate", "infusion_rate_adjustment_amount",
    "completion_interval", "restart_interval", "duration", "expiration_value",
    "fill_quantity", "basal_rate", "one_hr_max", "lockout_interval", "disp_sched",
    "event_txt", "medication", "drug", "isolate_num", "barcode_type",
})

# Quantities that are prorated or compared; sniffer may pick BIGINT on integer samples.
DOUBLE_COLUMNS = frozenset({
    "valuenum", "amount", "rate", "originalamount", "originalrate", "totalamount",
    "patientweight", "los",
})

# Identifiers and counters; sniffer types all-null samples as VARCHAR.
BIGINT_COLUMNS = frozenset({
    "subject_id", "hadm_id", "stay_id", "itemid", "seq_num", "icd_version",
    "labevent_id", "specimen_id", "microevent_id", "micro_specimen_id", "spec_itemid",
    "test_seq", "test_itemid", "org_itemid", "ab_itemid", "pharmacy_id", "emar_seq",
    "poe_seq", "orderid", "linkorderid", "transfer_id", "anchor_age", "anchor_year",
    "hospital_expire_flag", "note_seq", "field_ordinal",
})

# Per-table exceptions to the generic maps above.
TABLE_TYPES = {
    "icu/procedureevents": {"value": "DOUBLE"},
    "icu/outputevents": {"value": "DOUBLE"},
}

# Note text averages 1-10 kB: default 122,880-row groups make one subject's read
# decompress ~1 GB. 2048 rows cut per-subject note reads ~4x at equal file size.
FREE_TEXT_ROW_GROUP_SIZE = 2048
FREE_TEXT_TABLES = frozenset({"note/discharge", "note/radiology"})


def _read_checksums(path: Path) -> dict[str, str]:
    """Parse `<sha256> <path relative to source>` lines."""
    checksums = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        digest, relative = line.split(maxsplit=1)
        checksums[relative.strip()] = digest
    return checksums


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(source: Path) -> list[Path]:
    files = []
    for module in MODULES:
        files.extend(sorted((source / module).glob(f"*{CSV_SUFFIX}")))
    return files


def _table_key(source: Path, csv_path: Path) -> str:
    """e.g. 'hosp/admissions'."""
    return csv_path.relative_to(source).as_posix().removesuffix(CSV_SUFFIX)


def _verify_checksums(source: Path, files: list[Path], checksum_files: list[Path]) -> dict[str, str]:
    """Return {table_key: sha256}; exit non-zero on any mismatch or missing entry."""
    expected = {}
    for checksum_file in [source / CHECKSUM_FILE, *checksum_files]:
        expected |= _read_checksums(checksum_file)
    actual = {}
    failures = []
    for csv_path in files:
        relative = csv_path.relative_to(source).as_posix()
        digest = _sha256(csv_path)
        if expected.get(relative) != digest:
            failures.append(relative)
        actual[_table_key(source, csv_path)] = digest
        print(f"sha256 {relative} {'ok' if relative not in failures else 'MISMATCH'}", flush=True)

    if failures:
        print(f"checksum mismatch: {', '.join(failures)}", file=sys.stderr)
        sys.exit(EXIT_CHECKSUM_MISMATCH)
    return actual


def _header(csv_path: Path) -> list[str]:
    with gzip.open(csv_path, "rt") as handle:
        return handle.readline().strip().split(",")


def _read_csv_sql(csv_path: Path, columns: list[str], table_types: dict[str, str]) -> str:
    """read_csv() call with explicit types for time and text columns.

    preserve_insertion_order keeps output order == file order, so Parquet position is the
    record index. (parallel=false would buffer the whole table in the Parquet writer.)
    """
    types = {c: "TIMESTAMP" for c in columns if c in TIMESTAMP_COLUMNS}
    types |= {c: "VARCHAR" for c in columns if c in TEXT_COLUMNS}
    types |= {c: "DOUBLE" for c in columns if c in DOUBLE_COLUMNS}
    types |= {c: "BIGINT" for c in columns if c in BIGINT_COLUMNS}
    types |= table_types
    types_sql = ", ".join(f"'{c}': '{t}'" for c, t in types.items())
    return (
        f"read_csv('{csv_path}', header=true, types={{{types_sql}}})"
    )


def _manifest_entry_valid(entry: dict | None, digest: str, parquet_path: Path, con) -> bool:
    if entry is None or entry.get("sha256") != digest or not parquet_path.exists():
        return False

    rows = _parquet_rows(con, parquet_path)
    return rows == entry["parquet_rows"] == entry["source_rows"]


def _parquet_rows(con, parquet_path: Path) -> int:
    return con.execute(
        f"SELECT sum(row_group_num_rows) FROM (SELECT DISTINCT row_group_id, row_group_num_rows "
        f"FROM parquet_metadata('{parquet_path}'))"
    ).fetchone()[0] or 0


def _convert(con, csv_path: Path, parquet_path: Path, table_types: dict[str, str], parquet_options: str) -> dict:
    """Write one Parquet file atomically; return its manifest counts."""
    columns = _header(csv_path)
    read_sql = _read_csv_sql(csv_path, columns, table_types)
    order_sql = f"ORDER BY {SUBJECT_COLUMN}, {ROW_COLUMN}" if SUBJECT_COLUMN in columns else f"ORDER BY {ROW_COLUMN}"

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = parquet_path.with_suffix(parquet_path.suffix + TMP_SUFFIX)
    unsorted_path = parquet_path.with_suffix(UNSORTED_SUFFIX)

    # Pass 1 streams the CSV in file order; pass 2 numbers rows by Parquet position
    # (file_row_number streams, unlike row_number() OVER () which holds the table in RAM)
    # and sorts with spilling to temp_directory.
    con.execute(f"COPY (SELECT * FROM {read_sql}) TO '{unsorted_path}' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(
        f"COPY (SELECT file_row_number AS {ROW_COLUMN}, * EXCLUDE (file_row_number) "
        f"FROM read_parquet('{unsorted_path}', file_row_number=true) {order_sql}) "
        f"TO '{tmp_path}' (FORMAT parquet, COMPRESSION zstd{parquet_options})"
    )
    unsorted_path.unlink()

    # Independent pass over the source, so a dropped record shows as a count mismatch.
    source_rows = con.execute(f"SELECT count(*) FROM {read_sql}").fetchone()[0]
    parquet_rows = _parquet_rows(con, tmp_path)
    tmp_path.replace(parquet_path)
    return {"source_rows": source_rows, "parquet_rows": parquet_rows}


def _connect(dest: Path):
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{dest / '.duckdb_tmp'}'")
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order = true")
    con.execute("SET enable_progress_bar = false")
    return con


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build(source: Path, dest: Path, checksum_files: list[Path]) -> dict:
    """Verify, convert every table not already valid in the manifest, return the manifest.

    `checksum_files` extend `<source>/SHA256SUMS.txt`, e.g. MIMIC-IV-Note's own sums.
    """
    files = _source_files(source)
    digests = _verify_checksums(source, files, checksum_files)

    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / MANIFEST_FILE
    manifest = _load_manifest(manifest_path)
    con = _connect(dest)

    for csv_path in files:
        key = _table_key(source, csv_path)
        parquet_path = dest / f"{key}.parquet"
        if _manifest_entry_valid(manifest.get(key), digests[key], parquet_path, con):
            print(f"skip {key}", flush=True)
            continue

        print(f"convert {key}", flush=True)
        parquet_options = f", ROW_GROUP_SIZE {FREE_TEXT_ROW_GROUP_SIZE}" if key in FREE_TEXT_TABLES else ""
        counts = _convert(con, csv_path, parquet_path, TABLE_TYPES.get(key, {}), parquet_options)
        manifest[key] = {"sha256": digests[key], **counts}

        # Persist after each table so an interrupted build resumes.
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument(
        "--checksums", type=Path, action="append", default=[],
        help="extra SHA256SUMS file, paths relative to --source (repeatable)",
    )
    args = parser.parse_args()

    manifest = build(args.source, args.dest, args.checksums)
    mismatched = [k for k, v in manifest.items() if v["source_rows"] != v["parquet_rows"]]
    if mismatched:
        print(f"row count mismatch: {', '.join(mismatched)}", file=sys.stderr)
        sys.exit(EXIT_ROW_COUNT_MISMATCH)

    print(f"done: {len(manifest)} tables")


if __name__ == "__main__":
    main()
