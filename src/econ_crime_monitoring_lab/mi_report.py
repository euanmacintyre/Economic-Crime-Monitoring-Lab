from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class ReportArtifacts:
    report_dir: Path
    markdown_path: Path
    html_path: Path


def _format_table_for_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data available._"

    table = df.copy().head(max_rows)
    for col in table.columns:
        if pd.api.types.is_datetime64_any_dtype(table[col]):
            table[col] = pd.to_datetime(table[col], utc=True).dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    cols = table.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in cols) + " |" for _, row in table.iterrows()]
    return "\n".join([header, sep, *rows])


def _save_chart(series: pd.Series, output_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if series.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
    else:
        series.plot(ax=ax, marker="o")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def build_mi_weekly(alerts: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    if alerts.empty:
        return pd.DataFrame(
            columns=[
                "week_start",
                "alert_count",
                "tp",
                "fp",
                "open_cases",
                "avg_handling_hours",
                "top_rule",
                "typology_mix",
            ]
        )

    alerts_work = alerts.copy()
    alerts_work["created_at"] = pd.to_datetime(alerts_work["created_at"], utc=True)
    alerts_work["week_start"] = alerts_work["created_at"].dt.to_period("W").apply(
        lambda x: x.start_time
    )

    cases_work = cases.copy()
    if not cases_work.empty:
        cases_work = cases_work.merge(
            alerts_work[["alert_id", "week_start"]], on="alert_id", how="left"
        )

    rows: list[dict[str, object]] = []
    for week_start, alert_group in alerts_work.groupby("week_start"):
        case_group = (
            cases_work[cases_work["week_start"] == week_start]
            if not cases_work.empty
            else pd.DataFrame()
        )

        tp = (
            int((case_group["status"] == "closed_true_positive").sum())
            if not case_group.empty
            else 0
        )
        fp = (
            int((case_group["status"] == "closed_false_positive").sum())
            if not case_group.empty
            else 0
        )
        open_cases = (
            int(case_group["status"].isin(["new", "in_review"]).sum())
            if not case_group.empty
            else 0
        )
        avg_handling = (
            float(case_group["handling_time_hours"].dropna().mean())
            if (not case_group.empty and case_group["handling_time_hours"].notna().any())
            else 0.0
        )

        exploded_rules = (
            alert_group["rule_hits"].fillna("none").str.split(",").explode().str.strip()
        )
        exploded_rules = exploded_rules[(exploded_rules != "") & (exploded_rules != "none")]
        top_rule = (
            str(exploded_rules.value_counts().index[0])
            if not exploded_rules.empty
            else "none"
        )

        typology_mix = (
            alert_group["typology_label"].value_counts(normalize=True).round(4).to_dict()
        )

        rows.append(
            {
                "week_start": pd.Timestamp(week_start),
                "alert_count": int(len(alert_group)),
                "tp": tp,
                "fp": fp,
                "open_cases": open_cases,
                "avg_handling_hours": avg_handling,
                "top_rule": top_rule,
                "typology_mix": json.dumps(typology_mix),
            }
        )

    return pd.DataFrame(rows).sort_values("week_start")


def generate_weekly_mi_report(
    alerts: pd.DataFrame,
    cases: pd.DataFrame,
    model_metrics: dict[str, object],
    top_features: pd.DataFrame,
    model_metrics_weekly: pd.DataFrame,
    feature_drift_weekly: pd.DataFrame,
    mi_weekly: pd.DataFrame,
    output_root: Path | str = Path("reports"),
) -> ReportArtifacts:
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    report_dir = Path(output_root) / today
    report_dir.mkdir(parents=True, exist_ok=True)

    alerts_by_typology = (
        alerts.groupby("typology_label")
        .size()
        .reset_index(name="alert_count")
        .sort_values("alert_count", ascending=False)
        if not alerts.empty
        else pd.DataFrame(columns=["typology_label", "alert_count"])
    )
    alerts_by_channel = (
        alerts.groupby("channel")
        .size()
        .reset_index(name="alert_count")
        .sort_values("alert_count", ascending=False)
        if not alerts.empty
        else pd.DataFrame(columns=["channel", "alert_count"])
    )
    alerts_by_country = (
        alerts.groupby("country")
        .size()
        .reset_index(name="alert_count")
        .sort_values("alert_count", ascending=False)
        if not alerts.empty
        else pd.DataFrame(columns=["country", "alert_count"])
    )

    outcomes = (
        cases["status"].value_counts().rename_axis("status").reset_index(name="count")
        if not cases.empty
        else pd.DataFrame(columns=["status", "count"])
    )

    top_rules = (
        alerts["rule_hits"].fillna("none").str.split(",").explode().str.strip().value_counts()
        .rename_axis("rule_name")
        .reset_index(name="trigger_count")
        if not alerts.empty
        else pd.DataFrame(columns=["rule_name", "trigger_count"])
    )
    if not top_rules.empty:
        top_rules = top_rules[top_rules["rule_name"] != "none"]

    model_summary = pd.DataFrame(
        [
            {"metric": "AUC", "value": model_metrics.get("auc", float("nan"))},
            {"metric": "Precision", "value": model_metrics.get("precision", float("nan"))},
            {"metric": "Recall", "value": model_metrics.get("recall", float("nan"))},
            {"metric": "Train Rows", "value": model_metrics.get("train_rows", float("nan"))},
            {"metric": "Test Rows", "value": model_metrics.get("test_rows", float("nan"))},
        ]
    )

    auc_chart_path = report_dir / "auc_trend.png"
    alert_chart_path = report_dir / "alerts_trend.png"
    _save_chart(
        series=model_metrics_weekly.set_index("week_start")["auc"]
        if not model_metrics_weekly.empty
        else pd.Series(dtype=float),
        output_path=auc_chart_path,
        title="Weekly AUC Trend",
        ylabel="AUC",
    )
    _save_chart(
        series=mi_weekly.set_index("week_start")["alert_count"]
        if not mi_weekly.empty
        else pd.Series(dtype=float),
        output_path=alert_chart_path,
        title="Weekly Alert Volume",
        ylabel="Alerts",
    )

    md_sections = [
        "# Weekly Economic Crime Monitoring MI Report",
        f"Report date: **{today}**",
        "",
        "## Alert Volumes by Typology",
        _format_table_for_markdown(alerts_by_typology),
        "",
        "## Alert Volumes by Channel",
        _format_table_for_markdown(alerts_by_channel),
        "",
        "## Alert Volumes by Country",
        _format_table_for_markdown(alerts_by_country),
        "",
        "## Investigation Outcomes",
        _format_table_for_markdown(outcomes),
        "",
        "## Top Triggering Rules",
        _format_table_for_markdown(top_rules),
        "",
        "## Model Metrics",
        _format_table_for_markdown(model_summary),
        "",
        "## Weekly Monitoring Summary",
        _format_table_for_markdown(mi_weekly),
        "",
        "## AUC by Week",
        _format_table_for_markdown(model_metrics_weekly),
        "",
        "## Weekly Drift Proxy",
        _format_table_for_markdown(feature_drift_weekly),
        "",
        "## Top ML Features",
        _format_table_for_markdown(top_features),
    ]

    markdown_path = report_dir / "mi_report.md"
    markdown_path.write_text("\n".join(md_sections), encoding="utf-8")

    html_path = report_dir / "mi_report.html"
    html_content = f"""
    <html>
      <head>
        <meta charset=\"utf-8\" />
        <title>Economic Crime Monitoring MI Report</title>
        <style>
          body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 32px;
          }}
          h1, h2, h3 {{ color: #1f2d3d; }}
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
          th {{ background: #f5f7fa; text-align: left; }}
          img {{ width: 100%; max-width: 860px; border: 1px solid #eee; margin-bottom: 18px; }}
        </style>
      </head>
      <body>
        <h1>Weekly Economic Crime Monitoring MI Report</h1>
        <p>Report date: <strong>{today}</strong></p>

        <h2>Alert Volumes by Typology</h2>
        {alerts_by_typology.to_html(index=False)}

        <h2>Alert Volumes by Channel</h2>
        {alerts_by_channel.to_html(index=False)}

        <h2>Alert Volumes by Country</h2>
        {alerts_by_country.to_html(index=False)}

        <h2>Investigation Outcomes</h2>
        {outcomes.to_html(index=False)}

        <h2>Top Triggering Rules</h2>
        {top_rules.to_html(index=False)}

        <h2>Model Metrics</h2>
        {model_summary.to_html(index=False)}

        <h2>Weekly Monitoring Summary</h2>
        {mi_weekly.to_html(index=False)}

        <h2>AUC Trend</h2>
        {model_metrics_weekly.to_html(index=False)}
        <img src=\"{auc_chart_path.name}\" alt=\"AUC trend\" />

        <h2>Weekly Alert Volume Trend</h2>
        <img src=\"{alert_chart_path.name}\" alt=\"Alert trend\" />

        <h2>Weekly Drift Proxy</h2>
        {feature_drift_weekly.to_html(index=False)}

        <h2>Top ML Features</h2>
        {top_features.to_html(index=False)}
      </body>
    </html>
    """
    html_path.write_text(html_content, encoding="utf-8")

    return ReportArtifacts(report_dir=report_dir, markdown_path=markdown_path, html_path=html_path)
