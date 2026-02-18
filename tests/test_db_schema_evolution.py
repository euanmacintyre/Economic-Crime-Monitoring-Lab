from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from econ_crime_monitoring_lab.db import write_tables


def test_write_tables_adds_missing_run_id_column_for_existing_table(tmp_path: Path) -> None:
    db_path = tmp_path / "schema_evolution.db"

    legacy_customers = pd.DataFrame(
        [
            {
                "customer_id": "C000001",
                "created_at": "2025-01-01T00:00:00+00:00",
                "risk_band": "low",
                "home_country": "US",
            }
        ]
    )

    with sqlite3.connect(db_path) as conn:
        legacy_customers.to_sql("customers", conn, if_exists="replace", index=False)

    migrated_customers = pd.DataFrame(
        [
            {
                "run_id": "run-test-001",
                "customer_id": "C000002",
                "created_at": "2025-01-02T00:00:00+00:00",
                "risk_band": "medium",
                "home_country": "GB",
            }
        ]
    )

    write_tables({"customers": migrated_customers}, db_path=db_path, if_exists="append")

    with sqlite3.connect(db_path) as conn:
        schema_rows = conn.execute('PRAGMA table_info("customers")').fetchall()
        columns = [str(row[1]) for row in schema_rows]
        assert "run_id" in columns

        total_rows = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        assert int(total_rows) == 2

