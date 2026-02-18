# Economic Crime Monitoring Lab

[![CI](https://github.com/<OWNER>/economic-crime-monitoring-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/<OWNER>/economic-crime-monitoring-lab/actions/workflows/ci.yml)

Synthetic transaction monitoring pipeline with alerting, management information (MI), and a dashboard. All data in this repository is synthetic, and the project is for defensive analytics only.

## What it does

The pipeline generates synthetic customer, account, and transaction data; runs data quality checks; engineers monitoring features; applies rules and a baseline model; creates an alert queue and case records; produces weekly MI outputs; and serves a Streamlit dashboard for review.

## Quickstart (macOS)

```bash
make install
make run
make dashboard
```

`make dashboard` starts the app at `http://localhost:8501`. Generated outputs are written locally and are not committed to the repository.

## Outputs

- Default SQLite database: `data/econ_crime_lab.db`
- Reports directory: `reports/YYYY-MM-DD/`
- Key tables:
  - `customers`
  - `accounts`
  - `transactions`
  - `alerts`
  - `cases`
  - `mi_weekly`
  - `model_metrics` (if present)

## How this maps to Economic Crime (Data)

- Controls and monitoring: rule-based controls, alert volume/rate monitoring, and false-positive review.
- MI: weekly reporting outputs and trend tracking for alerts and case outcomes.
- Governance: reproducible runs, logged assumptions and run metadata, and data quality checks.
- Threat awareness: simulated typologies (for example structuring-style behavior and unusual movement patterns) support pattern spotting and triage practice.

## Repository layout

```text
scripts/    # pipeline and utility entry points
src/        # econ_crime_monitoring_lab package
tests/      # automated tests
data/       # local synthetic data outputs
reports/    # dated MI outputs
```

## Notes and limitations

- Synthetic data only.
- This is not a fraud playbook.
- The baseline model is illustrative.
- This is a learning project and not production monitoring.

## Screenshots

![Dashboard screenshot](docs/dashboard.png)
![Alert detail screenshot](docs/alert_detail.png)

Add screenshots by saving images into `docs/` with those names.
