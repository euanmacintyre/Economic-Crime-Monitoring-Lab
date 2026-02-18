#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from econ_crime_monitoring_lab import __version__
from econ_crime_monitoring_lab.alerts import build_alerts_and_cases
from econ_crime_monitoring_lab.config import Settings, get_settings
from econ_crime_monitoring_lab.data_gen import generate_synthetic_data
from econ_crime_monitoring_lab.db import (
    write_run_metadata,
    write_tables,
)
from econ_crime_monitoring_lab.features import build_transaction_features
from econ_crime_monitoring_lab.mi_report import build_mi_weekly, generate_weekly_mi_report
from econ_crime_monitoring_lab.model import train_and_score_model
from econ_crime_monitoring_lab.quality import build_dq_summary, validate_transactions
from econ_crime_monitoring_lab.rules import get_rule_definitions, score_rules

logger = logging.getLogger("econ_crime_pipeline")


@dataclass
class PipelineOutputs:
    run_id: str
    db_path: Path
    report_dir: Path


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_event(event: str, **kwargs: object) -> None:
    payload = json.dumps(kwargs, default=str, sort_keys=True)
    logger.info("%s | %s", event, payload)


@contextmanager
def stage_timer(stage_name: str, **context: object) -> Iterator[None]:
    log_event("stage_start", stage=stage_name, **context)
    started = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - started
        log_event("stage_complete", stage=stage_name, seconds=round(elapsed, 3), **context)


def parse_args(settings: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic economic crime monitoring pipeline."
    )
    parser.add_argument("--db-path", type=Path, default=settings.db_path)
    parser.add_argument("--reports-dir", type=Path, default=settings.reports_dir)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--days", type=int, default=settings.days)
    parser.add_argument("--customers", type=int, default=settings.n_customers)
    parser.add_argument(
        "--target-alert-rate",
        type=float,
        default=settings.target_alert_rate,
        help="Optional target proportion of transactions that become alerts (0..1).",
    )
    parser.add_argument("--txns-per-day", type=int, default=450)
    parser.add_argument("--tp-base-rate", type=float, default=0.38)
    parser.add_argument(
        "--fresh-db",
        action="store_true",
        help="Delete existing database file before pipeline run.",
    )
    return parser.parse_args()


def _resolve_version_hash() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip() or "unknown"
    except Exception:
        return "unknown"


def _run_id() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("run-%Y%m%d-%H%M%S")


def _save_raw_bundle(bundle: dict[str, pd.DataFrame], output_root: Path, run_id: str) -> Path:
    run_dir = output_root / "raw" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in bundle.items():
        frame.to_csv(run_dir / f"{name}.csv", index=False)
    return run_dir


def _save_processed_outputs(
    tables: dict[str, pd.DataFrame],
    output_root: Path,
    run_id: str,
) -> Path:
    processed_dir = output_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(processed_dir / f"{run_id}_{name}.csv", index=False)
        frame.to_csv(processed_dir / f"latest_{name}.csv", index=False)
    return processed_dir


def _attach_run_id(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "run_id", run_id)
    return out


def run_pipeline(args: argparse.Namespace) -> PipelineOutputs:
    run_id = _run_id()
    db_path: Path = Path(args.db_path)
    reports_dir: Path = Path(args.reports_dir)

    if args.fresh_db and db_path.exists():
        db_path.unlink()
        log_event("fresh_db_deleted", db_path=str(db_path.resolve()))

    data_root = PROJECT_ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    log_event(
        "pipeline_start",
        run_id=run_id,
        db_path=str(db_path.resolve()),
        reports_dir=str(reports_dir.resolve()),
        seed=args.seed,
        days=args.days,
        customers=args.customers,
    )

    with stage_timer("generate_synthetic_data"):
        bundle = generate_synthetic_data(
            days=args.days,
            seed=args.seed,
            txns_per_day=args.txns_per_day,
            n_customers=args.customers,
        )

    with stage_timer("persist_raw_csv"):
        raw_dir = _save_raw_bundle(bundle=bundle, output_root=data_root, run_id=run_id)

    with stage_timer("data_quality"):
        dq_summary = build_dq_summary(bundle["transactions"])
        validate_transactions(bundle["transactions"])

    with stage_timer("feature_engineering"):
        featured = build_transaction_features(
            transactions=bundle["transactions"],
            customers=bundle["customers"],
            merchants=bundle["merchants"],
            devices=bundle["devices"],
        )

    with stage_timer("rules_engine"):
        ruled = score_rules(featured)

    with stage_timer("model_training"):
        model_artifacts = train_and_score_model(ruled)

    with stage_timer("alert_generation"):
        alert_artifacts = build_alerts_and_cases(
            scored_df=model_artifacts.scored_transactions,
            tp_base_rate=args.tp_base_rate,
            seed=args.seed,
            target_alert_rate=args.target_alert_rate,
        )

    with stage_timer("weekly_mi_aggregation"):
        mi_weekly = build_mi_weekly(alerts=alert_artifacts.alerts, cases=alert_artifacts.cases)

    metrics = model_artifacts.metrics
    confusion = metrics["confusion_matrix"]
    model_metrics_table = pd.DataFrame(
        [
            {"metric": "auc", "value": metrics["auc"]},
            {"metric": "precision", "value": metrics["precision"]},
            {"metric": "recall", "value": metrics["recall"]},
            {"metric": "train_rows", "value": metrics["train_rows"]},
            {"metric": "test_rows", "value": metrics["test_rows"]},
            {"metric": "tn", "value": confusion["tn"]},
            {"metric": "fp", "value": confusion["fp"]},
            {"metric": "fn", "value": confusion["fn"]},
            {"metric": "tp", "value": confusion["tp"]},
        ]
    )

    with stage_timer("persist_sqlite_and_processed_csv"):
        tables = {
            "customers": _attach_run_id(bundle["customers"], run_id),
            "accounts": _attach_run_id(bundle["accounts"], run_id),
            "merchants": _attach_run_id(bundle["merchants"], run_id),
            "devices": _attach_run_id(bundle["devices"], run_id),
            "transactions_raw": _attach_run_id(bundle["transactions"], run_id),
            "transactions_features": _attach_run_id(featured, run_id),
            "transactions_scored": _attach_run_id(model_artifacts.scored_transactions, run_id),
            "alerts": _attach_run_id(alert_artifacts.alerts, run_id),
            "cases": _attach_run_id(alert_artifacts.cases, run_id),
            "rule_definitions": _attach_run_id(get_rule_definitions(), run_id),
            "model_metrics": _attach_run_id(model_metrics_table, run_id),
            "model_top_features": _attach_run_id(model_artifacts.top_features, run_id),
            "model_metrics_weekly": _attach_run_id(model_artifacts.model_metrics_weekly, run_id),
            "feature_drift_overall": _attach_run_id(model_artifacts.feature_drift_overall, run_id),
            "feature_drift_weekly": _attach_run_id(model_artifacts.feature_drift_weekly, run_id),
            "dq_summary": _attach_run_id(dq_summary, run_id),
            "mi_weekly": _attach_run_id(mi_weekly, run_id),
        }
        processed_dir = _save_processed_outputs(tables=tables, output_root=data_root, run_id=run_id)
        write_tables(tables, db_path=db_path, if_exists="append")
        write_run_metadata(
            run_id=run_id,
            seed=args.seed,
            days=args.days,
            version_hash=_resolve_version_hash(),
            app_version=__version__,
            db_path=db_path,
        )

    with stage_timer("build_report"):
        report = generate_weekly_mi_report(
            alerts=alert_artifacts.alerts,
            cases=alert_artifacts.cases,
            model_metrics=model_artifacts.metrics,
            top_features=model_artifacts.top_features,
            model_metrics_weekly=model_artifacts.model_metrics_weekly,
            feature_drift_weekly=model_artifacts.feature_drift_weekly,
            mi_weekly=mi_weekly,
            output_root=reports_dir,
        )

    log_event(
        "pipeline_complete",
        run_id=run_id,
        raw_dir=str(raw_dir.resolve()),
        processed_dir=str(processed_dir.resolve()),
        db_path=str(db_path.resolve()),
        report_dir=str(report.report_dir.resolve()),
        alerts_generated=len(alert_artifacts.alerts),
        cases_generated=len(alert_artifacts.cases),
    )

    print("Pipeline complete")
    print(f"Run ID: {run_id}")
    print(f"SQLite DB: {db_path.resolve()}")
    print(f"Report folder: {report.report_dir.resolve()}")

    return PipelineOutputs(run_id=run_id, db_path=db_path, report_dir=report.report_dir)


def main() -> None:
    configure_logging()
    settings = get_settings()
    args = parse_args(settings)
    run_pipeline(args)


if __name__ == "__main__":
    main()
