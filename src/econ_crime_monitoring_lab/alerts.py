from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from econ_crime_monitoring_lab.rules import get_rule_catalog


@dataclass
class AlertArtifacts:
    alerts: pd.DataFrame
    cases: pd.DataFrame


def _safe_json_loads(payload: str | float | None) -> list[dict[str, Any]]:
    if payload is None or (isinstance(payload, float) and np.isnan(payload)):
        return []
    if not isinstance(payload, str) or not payload.strip():
        return []
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def build_alerts(
    scored_df: pd.DataFrame,
    ml_threshold: float = 0.70,
    rules_threshold: float = 35.0,
    combined_threshold: float = 0.55,
    target_alert_rate: float | None = None,
) -> pd.DataFrame:
    df = scored_df.copy()

    rules_norm = (df["rules_risk_score"] / 100.0).clip(0, 1)
    df["combined_score"] = 0.55 * rules_norm + 0.45 * df["ml_score"]

    calibrated_combined_threshold = combined_threshold
    if target_alert_rate is not None and 0 < target_alert_rate < 1:
        quantile = max(0.0, min(1.0, 1.0 - target_alert_rate))
        calibrated_combined_threshold = float(df["combined_score"].quantile(quantile))

    alert_condition = (
        (df["ml_score"] >= ml_threshold)
        | (df["rules_risk_score"] >= rules_threshold)
        | (df["combined_score"] >= calibrated_combined_threshold)
    )

    alerts_src = df[alert_condition].copy().sort_values("timestamp").reset_index(drop=True)
    alerts_src["alert_id"] = [f"ALERT-{i:09d}" for i in range(1, len(alerts_src) + 1)]

    rule_catalog = get_rule_catalog()

    def _build_reason_codes(row: pd.Series) -> str:
        reasons: list[dict[str, Any]] = []

        if row["rule_hits"] != "none":
            for rule_name in [name.strip() for name in str(row["rule_hits"]).split(",")]:
                if not rule_name:
                    continue
                rule_info = rule_catalog.get(rule_name, {})
                reasons.append(
                    {
                        "type": "rule",
                        "code": rule_name,
                        "weight": rule_info.get("weight"),
                        "threshold": rule_info.get("threshold"),
                    }
                )

        if float(row["ml_score"]) >= ml_threshold:
            reasons.append(
                {
                    "type": "model",
                    "code": "ml_high_score",
                    "threshold": f"ml_score >= {ml_threshold:.2f}",
                    "value": round(float(row["ml_score"]), 4),
                }
            )

        if float(row["combined_score"]) >= calibrated_combined_threshold:
            reasons.append(
                {
                    "type": "combined",
                    "code": "combined_signal",
                    "threshold": f"combined_score >= {calibrated_combined_threshold:.2f}",
                    "value": round(float(row["combined_score"]), 4),
                }
            )

        return json.dumps(reasons)

    def _reason_summary(row: pd.Series) -> str:
        reasons = _safe_json_loads(str(row["reason_codes"]))
        if not reasons:
            return "Combined signal without explicit rule or model threshold."

        labels: list[str] = []
        for reason in reasons:
            code = str(reason.get("code", "unknown"))
            threshold = str(reason.get("threshold", ""))
            labels.append(f"{code} ({threshold})")
        return "; ".join(labels)

    def _severity(score: float) -> str:
        if score >= 0.80:
            return "high"
        if score >= 0.62:
            return "medium"
        return "low"

    alerts_src["reason_codes"] = alerts_src.apply(_build_reason_codes, axis=1)
    alerts_src["reason_summary"] = alerts_src.apply(_reason_summary, axis=1)
    alerts_src["severity"] = alerts_src["combined_score"].apply(_severity)
    alerts_src["created_at"] = alerts_src["timestamp"]

    if "ml_top_features" not in alerts_src.columns:
        alerts_src["ml_top_features"] = "[]"

    columns = [
        "alert_id",
        "txn_id",
        "reason_codes",
        "reason_summary",
        "severity",
        "created_at",
        "customer_id",
        "account_id",
        "channel",
        "country",
        "typology_label",
        "rules_risk_score",
        "ml_score",
        "combined_score",
        "rule_hits",
        "ml_top_features",
    ]
    return alerts_src[columns]


def simulate_case_queue(
    alerts: pd.DataFrame,
    tp_base_rate: float = 0.38,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if alerts.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "alert_id",
                "status",
                "opened_at",
                "assigned_at",
                "closed_at",
                "handling_time_hours",
            ]
        )

    statuses = ["new", "in_review", "closed_true_positive", "closed_false_positive"]
    status_probs = [0.10, 0.15, 0.30, 0.45]

    rows: list[dict[str, object]] = []
    for idx, row in alerts.reset_index(drop=True).iterrows():
        opened_at = pd.to_datetime(row["created_at"], utc=True)
        sampled_status = str(rng.choice(statuses, p=status_probs))

        if sampled_status in {"closed_true_positive", "closed_false_positive"}:
            sampled_status = (
                "closed_true_positive"
                if rng.random() < tp_base_rate
                else "closed_false_positive"
            )

        assigned_at = opened_at + pd.Timedelta(minutes=int(rng.integers(10, 180)))
        closed_at = pd.NaT
        handling_hours = np.nan

        if sampled_status in {"closed_true_positive", "closed_false_positive"}:
            handling_hours = float(np.clip(rng.gamma(shape=2.4, scale=5.0), 0.5, 120.0))
            closed_at = assigned_at + pd.Timedelta(hours=handling_hours)

        rows.append(
            {
                "case_id": f"CASE-{idx + 1:09d}",
                "alert_id": row["alert_id"],
                "status": sampled_status,
                "opened_at": opened_at,
                "assigned_at": assigned_at,
                "closed_at": closed_at,
                "handling_time_hours": handling_hours,
            }
        )

    return pd.DataFrame(rows)


def build_alerts_and_cases(
    scored_df: pd.DataFrame,
    tp_base_rate: float = 0.38,
    seed: int = 42,
    target_alert_rate: float | None = None,
) -> AlertArtifacts:
    alerts = build_alerts(scored_df=scored_df, target_alert_rate=target_alert_rate)
    cases = simulate_case_queue(alerts=alerts, tp_base_rate=tp_base_rate, seed=seed)
    return AlertArtifacts(alerts=alerts, cases=cases)

