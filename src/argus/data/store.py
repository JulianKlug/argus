"""DuckDB driver over the Parquet store: rows of one table for one subject.

No time logic here; leak policy lives in availability.py.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa

from argus.data.domains import LABEL, Table

PARQUET_SUFFIX = ".parquet"
ROW_COLUMN = "_row"
SUBJECT_COLUMN = "subject_id"

# Staff identifiers: not clinical information, never exposed.
DROPPED_COLUMNS = frozenset({
    "caregiver_id", "provider_id", "order_provider_id", "enter_provider_id", "admit_provider_id",
})

# Nullable pandas dtypes so concatenating tables never turns ids into floats.
_PANDAS_TYPES = {
    pa.int64(): pd.Int64Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.float64(): pd.Float64Dtype(),
    pa.bool_(): pd.BooleanDtype(),
}


@dataclass(frozen=True)
class _DictionaryLabel:
    """Label joined from a dictionary table, e.g. d_items.label on itemid."""

    dictionary: Table
    keys: tuple[str, ...]
    column: str


# Label source per table: a dictionary join or one of the table's own columns.
_LABELS: dict[Table, _DictionaryLabel | str] = {
    Table.CHARTEVENTS: _DictionaryLabel(Table.D_ITEMS, ("itemid",), "label"),
    Table.DATETIMEEVENTS: _DictionaryLabel(Table.D_ITEMS, ("itemid",), "label"),
    Table.INPUTEVENTS: _DictionaryLabel(Table.D_ITEMS, ("itemid",), "label"),
    Table.INGREDIENTEVENTS: _DictionaryLabel(Table.D_ITEMS, ("itemid",), "label"),
    Table.OUTPUTEVENTS: _DictionaryLabel(Table.D_ITEMS, ("itemid",), "label"),
    Table.PROCEDUREEVENTS: _DictionaryLabel(Table.D_ITEMS, ("itemid",), "label"),
    Table.LABEVENTS: _DictionaryLabel(Table.D_LABITEMS, ("itemid",), "label"),
    Table.DIAGNOSES_ICD: _DictionaryLabel(Table.D_ICD_DIAGNOSES, ("icd_code", "icd_version"), "long_title"),
    Table.PROCEDURES_ICD: _DictionaryLabel(Table.D_ICD_PROCEDURES, ("icd_code", "icd_version"), "long_title"),
    Table.MICROBIOLOGYEVENTS: "test_name",
    Table.EMAR: "medication",
    Table.EMAR_DETAIL: "product_description",
    Table.PRESCRIPTIONS: "drug",
    Table.PHARMACY: "medication",
    Table.POE: "order_type",
    Table.POE_DETAIL: "field_name",
    Table.RADIOLOGY: "note_type",
    Table.RADIOLOGY_DETAIL: "field_name",
    Table.DISCHARGE: "note_type",
    Table.DISCHARGE_DETAIL: "field_name",
    Table.OMR: "result_name",
    Table.DRGCODES: "description",
    Table.HCPCSEVENTS: "short_description",
    Table.ADMISSIONS: "admission_type",
    Table.ICUSTAYS: "first_careunit",
    Table.TRANSFERS: "careunit",
    Table.SERVICES: "curr_service",
}

_DICTIONARIES = (Table.D_ITEMS, Table.D_LABITEMS, Table.D_ICD_DIAGNOSES, Table.D_ICD_PROCEDURES)


class MimicStore:
    """Read-only access to the Parquet store built by scripts/build_store.py."""

    def __init__(self, store_dir: Path) -> None:
        self._dir = Path(store_dir)
        self._con = duckdb.connect()
        self._con.execute("SET parquet_metadata_cache = true")
        self._con.execute("SET enable_progress_bar = false")
        self._sql = {}

        # Dictionaries are small: load once into memory.
        for dictionary in _DICTIONARIES:
            self._con.execute(
                f"CREATE TABLE {dictionary.source_name} AS SELECT * FROM read_parquet('{self._path(dictionary)}')"
            )

    def rows(self, table: Table, subject_id: int) -> pd.DataFrame:
        """All rows of `table` for one subject, ordered by source row, with `label`."""
        sql = self._sql.get(table) or self._build_sql(table)
        arrow = self._con.sql(sql, params=[subject_id]).to_arrow_table()
        return arrow.to_pandas(types_mapper=_PANDAS_TYPES.get)

    def _path(self, table: Table) -> Path:
        return self._dir / f"{table.value}{PARQUET_SUFFIX}"

    def _build_sql(self, table: Table) -> str:
        """SELECT kept columns + label; cached per table."""
        source = f"read_parquet('{self._path(table)}')"
        columns = [r[0] for r in self._con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
        kept = ", ".join(f"t.{c}" for c in columns if c not in DROPPED_COLUMNS)

        label = _LABELS.get(table)
        join = ""
        label_sql = f"NULL::VARCHAR AS {LABEL}"
        if isinstance(label, str):
            label_sql = f"t.{label} AS {LABEL}"
        if isinstance(label, _DictionaryLabel):
            on = " AND ".join(f"t.{k} = d.{k}" for k in label.keys)
            join = f"LEFT JOIN {label.dictionary.source_name} d ON {on}"
            label_sql = f"d.{label.column} AS {LABEL}"

        sql = f"SELECT {kept}, {label_sql} FROM {source} t {join} WHERE t.{SUBJECT_COLUMN} = ? ORDER BY t.{ROW_COLUMN}"
        self._sql[table] = sql
        return sql
