from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_pipeline_run_creates_db_and_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "econ_crime_lab.db"
    reports_dir = tmp_path / "reports"

    cmd = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--days",
        "7",
        "--customers",
        "50",
        "--txns-per-day",
        "70",
        "--seed",
        "4",
        "--db-path",
        str(db_path),
        "--reports-dir",
        str(reports_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert db_path.exists()
    assert any(reports_dir.glob("*/mi_report.md"))

