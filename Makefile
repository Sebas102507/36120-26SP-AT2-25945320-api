SHELL := /bin/bash

# Location of the experiments repository (override: make artefacts EXPERIMENTS=/path/to/repo)
EXPERIMENTS ?= ../36120-26SP-AT2-25945320-experiments
MODEL_DIR := models

.PHONY: install test test-api run artefacts deploy down clean env smoke

env:
	@if [[ ! -e .env ]]; then cp .env.example .env; fi

install: env
	uv sync --extra dev

# Unit tests only: no network access or artefacts needed.
test: install
	uv run pytest -vv -m "not api"

# Integration tests against a running API (live Render service by default).
# Local instance: API_BASE_URL=http://localhost:8080 make test-api
test-api: install
	API_TESTS_REQUIRED=1 uv run pytest tests/test_api.py -v

run: install
	PYTHONPATH=app uv run uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Replace models/ with the current production artefacts from the experiments repository:
# the fitted preprocessing pipeline and the promoted CCI and WHC bundles.
artefacts:
	@test -f $(EXPERIMENTS)/models/preprocessing/feature_pipeline.joblib || \
		(echo "Run the preprocessing notebook in $(EXPERIMENTS) first." && exit 1)
	@test -d $(EXPERIMENTS)/models/comfort_climate/production || \
		(echo "Run the cci_comparison notebook first." && exit 1)
	@test -d $(EXPERIMENTS)/models/weather_hazard/production || \
		(echo "Run the whc_comparison notebook first." && exit 1)
	rm -rf $(MODEL_DIR)/preprocessing $(MODEL_DIR)/comfort_climate $(MODEL_DIR)/weather_hazard
	mkdir -p $(MODEL_DIR)
	cp -R $(EXPERIMENTS)/models/preprocessing $(MODEL_DIR)/preprocessing
	cp -R $(EXPERIMENTS)/models/comfort_climate/production $(MODEL_DIR)/comfort_climate
	cp -R $(EXPERIMENTS)/models/weather_hazard/production $(MODEL_DIR)/weather_hazard
	@echo "Artefacts copied into $(MODEL_DIR)."

deploy: env
	docker compose build
	docker compose up -d

down:
	docker compose down

clean:
	@find . -name '__pycache__' -prune -exec rm -rf {} \;
	@find . -name '*.pyc' -delete
	rm -rf .pytest_cache build dist *.egg-info

# Smoke-test a running instance (local by default; override with URL=https://...onrender.com)
URL ?= http://localhost:8080
smoke:
	curl -fsS "$(URL)/health" && echo
	curl -fsS "$(URL)/predict/index/comfort_climate?date=2025-01-01" && echo
	curl -fsS "$(URL)/predict/category/weather_hazard?date=2025-01-01" && echo
