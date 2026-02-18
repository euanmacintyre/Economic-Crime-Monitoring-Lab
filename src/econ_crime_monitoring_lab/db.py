from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes


DEFAULT_DB_PATH = Path("data/econ_crime_lab.db")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

CORE_SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS run_metadata (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        seed INTEGER NOT NULL,
        days INTEGER NOT NULL,
        version_hash TEXT,
        app_version TEXT,
        db_path TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS investigator_notes (
        run_id TEXT NOT NULL,
        alert_id TEXT NOT NULL,
        investigator_note TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (run_id, alert_id)
    )
    """,
]

INDEX_PLAN: dict[str, list[str]] = {
    "transactions_scored": ["run_id", "txn_id", "timestamp", "customer_id"],
    "alerts": ["run_id", "alert_id", "txn_id", "created_at", "customer_id"],
    "cases": ["run_id", "alert_id", "status", "opened_at"],
    "mi_weekly": ["run_id", "week_start"],
    "model_metrics_weekly": ["run_id", "week_start"],
    "feature_drift_weekly": ["run_id", "week_start", "feature"],
    "dq_summary": ["run_id", "check_name", "status"],
    "investigator_notes": ["run_id", "alert_id"],
    "run_metadata": ["created_at"],
}


def _validate_identifier(identifier: str) -> str:
    if not IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return identifier


def _quote_identifier(identifier: str) -> str:
    safe_identifier = _validate_identifier(identifier)
    return f'"{safe_identifier}"'


def _sql_type_for_dtype(dtype: object) -> str:
    if ptypes.is_bool_dtype(dtype):
        return "INTEGER"
    if ptypes.is_integer_dtype(dtype):
        return "INTEGER"
    if ptypes.is_float_dtype(dtype):
        return "REAL"
    if ptypes.is_datetime64_any_dtype(dtype):
        return "TEXT"
    if ptypes.is_string_dtype(dtype) or ptypes.is_object_dtype(dtype):
        return "TEXT"
    return "TEXT"


def _prepare_df_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if ptypes.is_datetime64_any_dtype(series.dtype):
            out[col] = series.map(lambda x: x.isoformat() if pd.notna(x) else None)
        elif ptypes.is_bool_dtype(series.dtype):
            out[col] = series.map(lambda x: int(x) if pd.notna(x) else None)
    return out


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        for statement in CORE_SCHEMA_SQL:
            conn.execute(statement)
        conn.commit()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    quoted_table = _quote_identifier(table_name)
    rows = conn.execute(f"PRAGMA table_info({quoted_table})").fetchall()

    columns: set[str] = set()
    for row in rows:
        if isinstance(row, sqlite3.Row):
            columns.add(str(row["name"]))
        else:
            columns.add(str(row[1]))
    return columns


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    _validate_identifier(table_name)
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1"
    row = conn.execute(query, (table_name,)).fetchone()
    return row is not None


def ensure_table_columns(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    _validate_identifier(table_name)

    if not _table_exists(conn, table_name):
        return

    quoted_table = _quote_identifier(table_name)
    existing_columns = _table_columns(conn, table_name)
    for col in df.columns:
        _validate_identifier(str(col))
        if col in existing_columns:
            continue

        quoted_col = _quote_identifier(str(col))
        sql_type = _sql_type_for_dtype(df[col].dtype)
        conn.execute(f"ALTER TABLE {quoted_table} ADD COLUMN {quoted_col} {sql_type}")
        existing_columns.add(str(col))


def _create_indexes(conn: sqlite3.Connection, table_name: str) -> None:
    _validate_identifier(table_name)
    if not _table_exists(conn, table_name):
        return

    quoted_table = _quote_identifier(table_name)
    cols = _table_columns(conn, table_name)
    for col in INDEX_PLAN.get(table_name, []):
        _validate_identifier(col)
        if col not in cols:
            continue

        idx_name = _validate_identifier(f"idx_{table_name}_{col}")
        quoted_col = _quote_identifier(col)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {idx_name} ON {quoted_table} ({quoted_col})"
        )


def write_table(
    df: pd.DataFrame,
    table_name: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    if_exists: str = "append",
) -> None:
    initialize_database(db_path)
    prepared_df = _prepare_df_for_sql(df)
    with get_connection(db_path) as conn:
        if if_exists == "append":
            ensure_table_columns(conn, table_name, prepared_df)
        prepared_df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        _create_indexes(conn, table_name)
        conn.commit()


def write_tables(
    table_map: dict[str, pd.DataFrame],
    db_path: Path | str = DEFAULT_DB_PATH,
    if_exists: str = "append",
) -> None:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        for table_name, df in table_map.items():
            prepared_df = _prepare_df_for_sql(df)
            if if_exists == "append":
                ensure_table_columns(conn, table_name, prepared_df)
            prepared_df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            _create_indexes(conn, table_name)
        conn.commit()


def read_table(
    table_name: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    run_id: str | None = None,
) -> pd.DataFrame:
    _validate_identifier(table_name)
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        if not _table_exists(conn, table_name):
            return pd.DataFrame()

        quoted_table = _quote_identifier(table_name)
        query = f"SELECT * FROM {quoted_table}"
        params: tuple[str, ...] = ()
        cols = _table_columns(conn, table_name)
        if run_id is not None and "run_id" in cols:
            query += " WHERE run_id = ?"
            params = (run_id,)

        return pd.read_sql_query(query, conn, params=params)


def delete_run_records(
    run_id: str,
    table_names: list[str],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    if not table_names:
        return

    initialize_database(db_path)
    with get_connection(db_path) as conn:
        for table_name in table_names:
            _validate_identifier(table_name)
            if not _table_exists(conn, table_name):
                continue
            cols = _table_columns(conn, table_name)
            if "run_id" in cols:
                quoted_table = _quote_identifier(table_name)
                conn.execute(f"DELETE FROM {quoted_table} WHERE run_id = ?", (run_id,))
        conn.commit()


def write_run_metadata(
    run_id: str,
    seed: int,
    days: int,
    version_hash: str,
    app_version: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    initialize_database(db_path)
    created_at = pd.Timestamp.now(tz="UTC").isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO run_metadata
            (run_id, created_at, seed, days, version_hash, app_version, db_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                seed,
                days,
                version_hash,
                app_version,
                str(db_path),
            ),
        )
        _create_indexes(conn, "run_metadata")
        conn.commit()


def list_run_ids(db_path: Path | str = DEFAULT_DB_PATH) -> list[str]:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        if not _table_exists(conn, "run_metadata"):
            return []
        rows = conn.execute(
            "SELECT run_id FROM run_metadata ORDER BY created_at DESC"
        ).fetchall()
    return [str(row[0]) for row in rows]


def get_latest_run_id(db_path: Path | str = DEFAULT_DB_PATH) -> str | None:
    run_ids = list_run_ids(db_path)
    return run_ids[0] if run_ids else None


def read_run_metadata(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        if not _table_exists(conn, "run_metadata"):
            return pd.DataFrame(
                columns=[
                    "run_id",
                    "created_at",
                    "seed",
                    "days",
                    "version_hash",
                    "app_version",
                    "db_path",
                ]
            )
        return pd.read_sql_query(
            "SELECT * FROM run_metadata ORDER BY created_at DESC",
            conn,
        )


def upsert_investigator_note(
    run_id: str,
    alert_id: str,
    note: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    initialize_database(db_path)
    updated_at = pd.Timestamp.now(tz="UTC").isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO investigator_notes (run_id, alert_id, investigator_note, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id, alert_id)
            DO UPDATE SET
                investigator_note = excluded.investigator_note,
                updated_at = excluded.updated_at
            """,
            (run_id, alert_id, note, updated_at),
        )
        _create_indexes(conn, "investigator_notes")
        conn.commit()


def read_investigator_notes(
    run_id: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    initialize_database(db_path)
    with get_connection(db_path) as conn:
        if not _table_exists(conn, "investigator_notes"):
            return pd.DataFrame(
                columns=["run_id", "alert_id", "investigator_note", "updated_at"]
            )
        query = "SELECT * FROM investigator_notes WHERE run_id = ?"
        return pd.read_sql_query(query, conn, params=(run_id,))

