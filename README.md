# econ_crime_monitoring_lab

Synthetic-only economic crime monitoring lab for local demo and interview use.
The project simulates a fraud + AML monitoring stack end-to-end: synthetic transaction generation, controls, rules + ML scoring, alerting, case simulation, MI reporting, and investigator dashboarding.

## Quick Start (macOS, ~2 minutes)

```bash
make install
make run
make dashboard
```

What happens:
- `make run` creates outputs in local folders only:
  - `data/econ_crime_lab.db`
  - `reports/YYYY-MM-DD/mi_report.md`
  - `reports/YYYY-MM-DD/mi_report.html`
- `make dashboard` starts Streamlit and loads data from SQLite.

## Synthetic Data & Repo Hygiene

- All data is synthetic. No real customer data is used.
- Generated outputs are intentionally ignored by git (`.db`, CSV outputs, reports, caches).
- To reset local outputs and caches:

```bash
make clean
```

## What This Demonstrates (Economic Crime Data)

- Monitoring controls: schema validation, data quality checks, and run metadata.
- Management information (MI): weekly alert volumes, outcomes, and handling metrics.
- Alert explainability: rule hit context, model feature contributions, and investigator notes.

## Common Commands

```bash
make install
make run
make dashboard
make test
make lint
make clean
```

## Architecture

See `docs/architecture.md` for a mermaid system overview.
