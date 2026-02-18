from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from econ_crime_monitoring_lab.config import get_settings
from econ_crime_monitoring_lab.db import (
    get_latest_run_id,
    read_investigator_notes,
    read_run_metadata,
    read_table,
    upsert_investigator_note,
)


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _format_run_option(row: pd.Series) -> str:
    created = pd.to_datetime(row["created_at"], utc=True, errors="coerce")
    created_str = created.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(created) else "unknown"
    return f"{row['run_id']} ({created_str})"


def _safe_bar_chart(series: pd.Series, title: str) -> None:
    st.subheader(title)
    if series.empty:
        st.info("No data for current filter selection.")
        return
    st.bar_chart(series)


def _safe_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    st.subheader(title)
    if df.empty:
        st.info("No data for current filter selection.")
        return
    chart = df[[x_col, y_col]].dropna().sort_values(x_col).set_index(x_col)
    if chart.empty:
        st.info("No chartable points after filtering.")
        return
    st.line_chart(chart)


def _render_alert_detail(
    selected_alert_id: str,
    run_id: str,
    db_path: Path,
    alerts: pd.DataFrame,
    transactions: pd.DataFrame,
    notes_map: dict[str, str],
) -> None:
    selected_alert = alerts[alerts["alert_id"] == selected_alert_id]
    if selected_alert.empty:
        return

    alert_row = selected_alert.iloc[0]
    txn_row = transactions[transactions["txn_id"] == alert_row["txn_id"]]
    txn_context = txn_row.iloc[0] if not txn_row.empty else None

    st.markdown("### Alert Detail")
    st.markdown(f"**Alert ID:** `{selected_alert_id}`")
    st.markdown(f"**Reason Summary:** {alert_row['reason_summary']}")

    reason_codes = _parse_json_list(alert_row.get("reason_codes"))
    if reason_codes:
        st.markdown("**Reason Codes**")
        st.dataframe(pd.DataFrame(reason_codes), use_container_width=True, hide_index=True)

    model_features = _parse_json_list(alert_row.get("ml_top_features"))
    if model_features:
        st.markdown("**Top Model Feature Contributions**")
        st.dataframe(pd.DataFrame(model_features), use_container_width=True, hide_index=True)

    if txn_context is not None:
        st.markdown("**Transaction Context**")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Transaction Amount",
            f"{float(txn_context['amount']):,.2f}",
            help="Current transaction amount.",
        )
        c2.metric(
            "Customer Baseline Amount",
            f"{float(txn_context['customer_prev_mean_amount']):,.2f}",
            help="Average historical amount before this transaction.",
        )
        c3.metric(
            "Amount vs Baseline",
            f"{float(txn_context['amount_to_customer_mean']):.2f}x",
            help="Current amount divided by historical customer mean.",
        )

        customer_id = txn_context["customer_id"]
        txn_ts = pd.to_datetime(txn_context["timestamp"], utc=True, errors="coerce")
        if pd.notna(txn_ts):
            window_start = txn_ts - pd.Timedelta(hours=24)
            timeline = transactions[
                (transactions["customer_id"] == customer_id)
                & (transactions["timestamp"] >= window_start)
                & (transactions["timestamp"] <= txn_ts)
            ].sort_values("timestamp")
            st.markdown("**Customer Activity Timeline (Last 24h)**")
            if timeline.empty:
                st.info("No transactions found in the customer 24h window.")
            else:
                timeline_chart = timeline[["timestamp", "amount"]].set_index("timestamp")
                st.line_chart(timeline_chart)

    st.markdown("**Investigator Note**")
    note_key = f"note_{run_id}_{selected_alert_id}"
    default_note = notes_map.get(selected_alert_id, "")
    note_value = st.text_area(
        "Add or update note",
        value=default_note,
        key=note_key,
        height=120,
        help="Stored in SQLite against this run_id and alert_id.",
    )
    if st.button("Save note", key=f"save_{run_id}_{selected_alert_id}"):
        upsert_investigator_note(
            run_id=run_id,
            alert_id=selected_alert_id,
            note=note_value,
            db_path=db_path,
        )
        st.success("Investigator note saved.")


def main() -> None:
    settings = get_settings()

    st.set_page_config(page_title="Economic Crime Monitoring Lab", layout="wide")
    st.title("Economic Crime Monitoring Lab Dashboard")
    st.caption("Synthetic fraud + AML monitoring outputs from SQLite")

    st.sidebar.header("Configuration")
    db_path = Path(st.sidebar.text_input("SQLite path", value=str(settings.db_path)))

    if not db_path.exists():
        st.warning("Database file not found. Run `make run` first.")
        return

    run_meta = read_run_metadata(db_path=db_path)
    if run_meta.empty:
        st.warning("No run metadata found. Run `make run` first.")
        return

    run_meta["label"] = run_meta.apply(_format_run_option, axis=1)
    latest_run_id = get_latest_run_id(db_path=db_path)
    default_index = 0
    if latest_run_id is not None:
        latest_matches = run_meta.index[run_meta["run_id"] == latest_run_id]
        if len(latest_matches) > 0:
            default_index = int(latest_matches[0])

    selected_label = st.sidebar.selectbox(
        "Run",
        options=run_meta["label"].tolist(),
        index=default_index,
        help="Latest run is selected by default.",
    )
    selected_run_id = str(run_meta.loc[run_meta["label"] == selected_label, "run_id"].iloc[0])

    transactions = read_table("transactions_scored", db_path=db_path, run_id=selected_run_id)
    alerts = read_table("alerts", db_path=db_path, run_id=selected_run_id)
    cases = read_table("cases", db_path=db_path, run_id=selected_run_id)
    model_metrics_weekly = read_table(
        "model_metrics_weekly", db_path=db_path, run_id=selected_run_id
    )
    feature_drift_weekly = read_table(
        "feature_drift_weekly", db_path=db_path, run_id=selected_run_id
    )
    mi_weekly = read_table("mi_weekly", db_path=db_path, run_id=selected_run_id)
    dq_summary = read_table("dq_summary", db_path=db_path, run_id=selected_run_id)
    notes = read_investigator_notes(run_id=selected_run_id, db_path=db_path)

    if transactions.empty:
        st.warning("No scored transactions found for selected run.")
        return

    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], utc=True, errors="coerce")
    if not alerts.empty:
        alerts["created_at"] = pd.to_datetime(alerts["created_at"], utc=True, errors="coerce")
    if not cases.empty:
        cases["opened_at"] = pd.to_datetime(cases["opened_at"], utc=True, errors="coerce")

    max_ts = transactions["timestamp"].max()
    min_ts = transactions["timestamp"].min()
    default_start = max(min_ts, max_ts - pd.Timedelta(days=30)) if pd.notna(max_ts) else min_ts

    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_start.date(), max_ts.date()) if pd.notna(max_ts) else None,
        help="Filters charts and alert explorer. Defaults to last 30 days.",
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0], tz="UTC")
        end_date = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
    else:
        start_date = min_ts
        end_date = max_ts + pd.Timedelta(days=1)

    tx_filtered = transactions[
        (transactions["timestamp"] >= start_date) & (transactions["timestamp"] < end_date)
    ]
    alerts_filtered = (
        alerts[(alerts["created_at"] >= start_date) & (alerts["created_at"] < end_date)]
        if not alerts.empty
        else alerts
    )

    cases_filtered = cases
    if not cases.empty and not alerts_filtered.empty:
        cases_filtered = cases[cases["alert_id"].isin(alerts_filtered["alert_id"])]

    total_txns = len(tx_filtered)
    suspicious_rate = float(tx_filtered["is_suspicious"].mean()) if total_txns else 0.0
    alert_count = len(alerts_filtered)
    open_cases = (
        int(cases_filtered["status"].isin(["new", "in_review"]).sum())
        if not cases_filtered.empty
        else 0
    )
    tp_closed = (
        int((cases_filtered["status"] == "closed_true_positive").sum())
        if not cases_filtered.empty
        else 0
    )
    fp_closed = (
        int((cases_filtered["status"] == "closed_false_positive").sum())
        if not cases_filtered.empty
        else 0
    )
    closed_total = tp_closed + fp_closed
    tp_rate = tp_closed / closed_total if closed_total else 0.0
    avg_handling = (
        float(cases_filtered["handling_time_hours"].dropna().mean())
        if not cases_filtered.empty and cases_filtered["handling_time_hours"].notna().any()
        else 0.0
    )

    latest_auc = float("nan")
    if not model_metrics_weekly.empty:
        model_metrics_weekly["week_start"] = pd.to_datetime(
            model_metrics_weekly["week_start"], utc=True, errors="coerce"
        )
        weekly_in_range = model_metrics_weekly[
            (model_metrics_weekly["week_start"] >= start_date)
            & (model_metrics_weekly["week_start"] < end_date)
        ].sort_values("week_start")
        if not weekly_in_range.empty:
            latest_auc = float(weekly_in_range["auc"].dropna().iloc[-1])

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric(
        "Transactions",
        f"{total_txns:,}",
        help="Total transactions in the selected date range.",
    )
    k2.metric(
        "Suspicious Label Rate",
        f"{suspicious_rate:.2%}",
        help="Synthetic labelled suspicious rate for selected transactions.",
    )
    k3.metric(
        "Alerts",
        f"{alert_count:,}",
        help="Alerts generated by combined rules + ML decisioning.",
    )
    k4.metric(
        "Open Cases",
        f"{open_cases:,}",
        help="Cases currently in `new` or `in_review` statuses.",
    )
    k5.metric(
        "TP Rate (closed)",
        f"{tp_rate:.2%}",
        help="Share of closed cases marked true positive.",
    )
    k6.metric(
        "Avg Handling (hrs)",
        f"{avg_handling:.2f}",
        help="Average case handling time for cases with closure timestamps.",
    )
    k7.metric(
        "Latest Weekly AUC",
        f"{latest_auc:.3f}" if pd.notna(latest_auc) else "n/a",
        help="Most recent available weekly AUC for the selected date range.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        by_typology = (
            alerts_filtered.groupby("typology_label")
            .size()
            .rename("alert_count")
            .sort_values(ascending=False)
            if not alerts_filtered.empty
            else pd.Series(dtype=float)
        )
        _safe_bar_chart(by_typology, "Alerts by Typology")

    with c2:
        by_channel = (
            alerts_filtered.groupby("channel")
            .size()
            .rename("alert_count")
            .sort_values(ascending=False)
            if not alerts_filtered.empty
            else pd.Series(dtype=float)
        )
        _safe_bar_chart(by_channel, "Alerts by Channel")

    with c3:
        by_country = (
            alerts_filtered.groupby("country")
            .size()
            .rename("alert_count")
            .sort_values(ascending=False)
            .head(10)
            if not alerts_filtered.empty
            else pd.Series(dtype=float)
        )
        _safe_bar_chart(by_country, "Alerts by Country (Top 10)")

    st.subheader("Monitoring")
    weekly_auc = pd.DataFrame()
    if not model_metrics_weekly.empty:
        weekly_auc = model_metrics_weekly[
            (model_metrics_weekly["week_start"] >= start_date)
            & (model_metrics_weekly["week_start"] < end_date)
        ].copy()
    _safe_line_chart(weekly_auc, "week_start", "auc", "Model AUC Trend")

    if not feature_drift_weekly.empty:
        feature_drift_weekly["week_start"] = pd.to_datetime(
            feature_drift_weekly["week_start"], utc=True, errors="coerce"
        )
        drift_filtered = feature_drift_weekly[
            (feature_drift_weekly["week_start"] >= start_date)
            & (feature_drift_weekly["week_start"] < end_date)
        ]
        st.subheader("Feature Drift Proxy")
        if drift_filtered.empty:
            st.info("No drift records for selected range.")
        else:
            drift_feature = st.selectbox(
                "Drift feature",
                options=sorted(drift_filtered["feature"].unique()),
            )
            drift_chart = drift_filtered[drift_filtered["feature"] == drift_feature]
            _safe_line_chart(
                drift_chart,
                "week_start",
                "relative_shift",
                f"Weekly Relative Shift: {drift_feature}",
            )
            st.dataframe(
                drift_filtered.sort_values(["week_start", "feature"]),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Data Quality")
    if dq_summary.empty:
        st.info("No data quality summary available.")
    else:
        pass_count = int((dq_summary["status"] == "pass").sum())
        fail_count = int((dq_summary["status"] == "fail").sum())
        d1, d2 = st.columns(2)
        d1.metric("Checks Passed", f"{pass_count}")
        d2.metric("Checks Failed", f"{fail_count}")
        st.dataframe(dq_summary, use_container_width=True, hide_index=True)

    st.subheader("Alerts Explorer")
    if alerts_filtered.empty:
        st.info("No alerts in selected date range.")
        return

    notes_map = dict(zip(notes["alert_id"], notes["investigator_note"])) if not notes.empty else {}

    explorer = alerts_filtered.copy()
    if not cases_filtered.empty:
        explorer = explorer.merge(
            cases_filtered[["alert_id", "status", "handling_time_hours"]],
            on="alert_id",
            how="left",
        )
    else:
        explorer["status"] = "new"
        explorer["handling_time_hours"] = pd.NA

    explorer["investigator_note"] = explorer["alert_id"].map(notes_map).fillna("")

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    severity_values = sorted(explorer["severity"].dropna().unique())
    channel_values = sorted(explorer["channel"].dropna().unique())
    country_values = sorted(explorer["country"].dropna().unique())
    status_values = sorted(explorer["status"].dropna().unique())

    severity_filter = fcol1.multiselect("Severity", severity_values, default=severity_values)
    channel_filter = fcol2.multiselect("Channel", channel_values, default=channel_values)
    country_filter = fcol3.multiselect("Country", country_values, default=country_values)
    status_filter = fcol4.multiselect("Case Status", status_values, default=status_values)

    filtered = explorer[
        explorer["severity"].isin(severity_filter)
        & explorer["channel"].isin(channel_filter)
        & explorer["country"].isin(country_filter)
        & explorer["status"].isin(status_filter)
    ].sort_values("created_at", ascending=False)

    if filtered.empty:
        st.info("No alerts match the selected explorer filters.")
        return

    display_columns = [
        "alert_id",
        "txn_id",
        "created_at",
        "severity",
        "status",
        "channel",
        "country",
        "typology_label",
        "rules_risk_score",
        "ml_score",
        "combined_score",
        "reason_summary",
        "investigator_note",
    ]

    selected_alert_id: str | None = None
    try:
        selection = st.dataframe(
            filtered[display_columns],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="alerts_table",
        )
        if selection is not None and selection.selection.rows:
            selected_row = selection.selection.rows[0]
            selected_alert_id = str(filtered.iloc[selected_row]["alert_id"])
    except TypeError:
        st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True)

    if selected_alert_id is None:
        selected_alert_id = st.selectbox(
            "Alert detail selection",
            options=filtered["alert_id"].tolist(),
        )

    _render_alert_detail(
        selected_alert_id=selected_alert_id,
        run_id=selected_run_id,
        db_path=db_path,
        alerts=filtered,
        transactions=tx_filtered,
        notes_map=notes_map,
    )

    st.subheader("Weekly MI Table")
    if mi_weekly.empty:
        st.info("No weekly MI aggregate available.")
    else:
        mi_weekly["week_start"] = pd.to_datetime(mi_weekly["week_start"], utc=True, errors="coerce")
        mi_filtered = mi_weekly[
            (mi_weekly["week_start"] >= start_date) & (mi_weekly["week_start"] < end_date)
        ]
        st.dataframe(mi_filtered, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
