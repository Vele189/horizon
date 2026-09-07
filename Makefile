# The three commands a reviewer types. Nothing else in this repository is
# required reading to run it.
#
#   make setup    provision: venv, dependencies, Postgres, schema, dbt profile
#   make run      the pipeline, end to end
#   make test     pytest and dbt test
#
# Every target is idempotent. `make setup` twice is `make setup` once, and a
# half-finished setup is repaired by running it again rather than by reading
# this file to work out which step failed.

SHELL := /bin/bash
.DEFAULT_GOAL := help

VENV    ?= .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
COMPOSE ?= docker compose
DBT     := $(PY) dbt_analytics/dbt_env.py -- dbt
MODE    ?= daily

.PHONY: help setup run test lint clean venv deps database schema profile seed dashboard

help:
	@echo "make setup    — venv, dependencies, Postgres, bronze schema, dbt profile"
	@echo "make run      — the full pipeline (MODE=daily, or MODE=backfill)"
	@echo "make test     — pytest, then dbt test"
	@echo "make lint     — ruff"
	@echo "make dashboard— run the Streamlit app locally"
	@echo "make clean    — remove caches and the venv (leaves the database alone)"

# --- setup ------------------------------------------------------------------

setup: deps database schema profile seed
	@echo
	@echo "Setup complete. Next: make run"

$(VENV):
	python3 -m venv $(VENV)

venv: $(VENV)

# Upgrading pip first because resolving this dependency set on an old pip is
# slow enough that people assume it has hung.
deps: venv
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements-dev.txt
	@echo "dependencies installed"

# .env has no default and must not: docker-compose.yml refuses to start rather
# than standing up a warehouse with a blank password.
.env:
	@test -f .env || { \
	  cp .env.example .env; \
	  echo "created .env from .env.example — set POSTGRES_PASSWORD and DATABASE_URL, then re-run"; \
	  exit 1; \
	}

database: .env
	$(COMPOSE) up -d
	@echo -n "waiting for postgres "
	@for i in $$(seq 1 60); do \
	  if $(COMPOSE) exec -T postgres pg_isready -q; then echo "— ready"; exit 0; fi; \
	  echo -n "."; sleep 1; \
	done; \
	echo " — timed out"; exit 1

# The bronze DDL is `create ... if not exists` throughout, so re-applying is a
# no-op rather than an error.
schema: database
	$(PY) ingestion/apply_schema.py

# Generated, git-ignored, and holding env_var() lookups rather than values.
profile:
	$(PY) dbt_analytics/dbt_env.py --write-profile

# cities.yml is the registry; the seed is derived from it and regenerating is
# how the two are kept from drifting.
seed:
	$(PY) dbt_analytics/export_cities.py

# --- running ----------------------------------------------------------------

run:
	$(PY) run_pipeline.py --mode $(MODE)

dashboard:
	$(VENV)/bin/streamlit run dashboard/app.py

# --- checking ---------------------------------------------------------------

# Two suites, and both run. pytest covers the Python; dbt test covers the
# assertions that live in the warehouse, which pytest cannot see.
test:
	$(PY) -m pytest -q -m "not live"
	$(DBT) test --project-dir dbt_analytics

lint:
	$(VENV)/bin/ruff check .

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache dbt_analytics/target dbt_analytics/logs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@echo "caches and venv removed; the database and its volume are untouched"
