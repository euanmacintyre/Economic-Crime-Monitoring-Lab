SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
VENV_BIN := $(VENV)/bin
VENV_PYTHON := $(VENV_BIN)/python

.PHONY: install run dashboard test lint clean

install:
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e .
	$(VENV_PYTHON) -m pip install -e ".[dev]"

run: install
	$(VENV_PYTHON) scripts/run_pipeline.py

dashboard: install
	$(VENV_BIN)/streamlit run src/econ_crime_monitoring_lab/dashboard.py

test: install
	$(VENV_PYTHON) -m pytest -q

lint: install
	$(VENV_BIN)/ruff check .
	$(VENV_BIN)/mypy src scripts

clean:
	bash scripts/clean_outputs.sh
