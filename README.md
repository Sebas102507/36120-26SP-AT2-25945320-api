# Weather Intelligence API

FastAPI service that serves two machine learning models for **Sydney, NSW**, trained on the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), and is
deployed on Render with Docker.

| Target | Type | Horizon | Endpoint |
|:--|:--|:--|:--|
| Climate Comfort Index (CCI, 0–100) | Regression | +1, +2, +3 days | `/predict/index/comfort_climate` |
| Weather Hazard Category (WHC) | Multiclass classification | exactly +7 days | `/predict/category/weather_hazard` |

| | |
|:--|:--|
| **Student name** | Juan Vargas T. |
| **Student ID** | 25945320 |
| **Subject** | 36120 Advanced Machine Learning Application — Assessment Task 2 |
| **Live service** | https://three6120-26sp-at2-25945320-api.onrender.com (interactive docs at [`/docs`](https://three6120-26sp-at2-25945320-api.onrender.com/docs)) |
| **API repository** | https://github.com/Sebas102507/36120-26SP-AT2-25945320-api |
| **Experiments repository** | https://github.com/Sebas102507/36120-26SP-AT2-25945320-experiments |
| **Custom package** | [`nucleus-ml`](https://test.pypi.org/project/nucleus-ml/) on TestPyPI (module `nucleus_ml.weather`) |

---

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The trained artefacts are already in
`models/`.

```sh
git clone https://github.com/Sebas102507/36120-26SP-AT2-25945320-api.git
cd 36120-26SP-AT2-25945320-api
make run        # creates .env, installs dependencies, starts the API on port 8080
```

Then open <http://localhost:8080/docs>, or run `make smoke` in a second terminal to call the main
endpoints.

Without `make`:

```sh
cp .env.example .env
uv sync --extra dev          # nucleus-ml from TestPyPI, everything else from PyPI
PYTHONPATH=app uv run uvicorn main:app --reload --port 8080
```

With Docker (the same image Render runs):

```sh
make deploy     # docker compose build + up on port 8080
make down       # stop it
```

## Endpoints

| Method | Path | Input | Description |
|:--|:--|:--|:--|
| GET | `/` | – | Project overview: title, description, version, student, endpoints, targets, GitHub link |
| GET | `/health` | – | Liveness check; answers even if a model failed to load |
| GET | `/predict/index/comfort_climate` | `date` (YYYY-MM-DD, required) | CCI for the three days after `date` |
| GET | `/predict/category/weather_hazard` | `date` (YYYY-MM-DD, required) | WHC for the day exactly seven days after `date` |
| GET | `/model-metadata` | – | Target, algorithm, horizon, features, feature importance and performance metrics per model |
| GET | `/monitoring/drift` | – | Whether recently served data still resembles the training data |

Accepted dates run from `PREDICTION_MIN_DATE` (default 2000-01-01) to about five days before
today, because the Open-Meteo archive lags real time. The model needs the 35 days before `date`,
which it downloads on each request.

### Sample calls (live service)

Base URL: **https://three6120-26sp-at2-25945320-api.onrender.com**

```sh
BASE=https://three6120-26sp-at2-25945320-api.onrender.com
```

For a local run, use `BASE=http://localhost:8080` instead. The free Render instance sleeps when
idle, so the first call can take up to a minute.

The response values below are illustrative; the live service returns the deployed models' actual
predictions and metadata.

#### `GET /` — project overview

```sh
curl "$BASE/"
```

```json
{
  "project": "Weather Intelligence API",
  "description": "Machine learning services for Sydney weather forecasting: ...",
  "version": "1.0.0",
  "student": "Juan Vargas T. (25945320)",
  "github": "https://github.com/Sebas102507/36120-26SP-AT2-25945320-api",
  "endpoints": ["/", "/health", "/predict/index/comfort_climate",
                "/predict/category/weather_hazard", "/model-metadata", "/monitoring/drift"],
  "prediction_targets": [
    {"target": "Climate Comfort Index (0-100)",
     "endpoint": "/predict/index/comfort_climate?date=YYYY-MM-DD",
     "prediction_type": "regression", "forecast_horizon": "+1, +2 and +3 days"},
    {"target": "Weather Hazard Category",
     "endpoint": "/predict/category/weather_hazard?date=YYYY-MM-DD",
     "prediction_type": "multiclass classification", "forecast_horizon": "+7 days"}
  ]
}
```

#### `GET /health` — health check

```sh
curl "$BASE/health"
```

```json
{"status": "healthy", "message": "Weather Intelligence API is running"}
```

#### `GET /predict/index/comfort_climate` — CCI for the next three days

```sh
curl "$BASE/predict/index/comfort_climate?date=2025-01-01"
```

```json
{
  "input_date": "2025-01-01",
  "predictions": {"comfort_climate": {"2025-01-02": 72, "2025-01-03": 76, "2025-01-04": 81}}
}
```

#### `GET /predict/category/weather_hazard` — WHC exactly seven days ahead

```sh
curl "$BASE/predict/category/weather_hazard?date=2025-01-01"
```

```json
{
  "input_date": "2025-01-01",
  "predictions": {"weather_hazard": {"2025-01-08": "Moderate Risk"}}
}
```

The category is one of `Low Risk`, `Moderate Risk`, `High Risk` or `Extreme Risk`.

#### `GET /model-metadata` — deployed models

```sh
curl "$BASE/model-metadata"
```

```json
[
  {
    "target": "comfort_climate",
    "experiment": "<champion experiment>",
    "prediction_type": "regression",
    "algorithm": "<algorithm>",
    "forecast_horizon": "1, 2, 3 Days",
    "features": ["cci", "temperature_2m", "..."],
    "feature_importance": {"cci": 0.21, "...": 0.0},
    "performance_metrics": {"mae": 0.0, "mae_h1": 0.0, "mae_h2": 0.0, "mae_h3": 0.0, "...": 0.0}
  },
  {
    "target": "weather_hazard",
    "experiment": "<champion experiment>",
    "prediction_type": "classification",
    "algorithm": "<algorithm>",
    "forecast_horizon": "7 Days",
    "features": ["whi", "wind_gusts_10m", "..."],
    "feature_importance": {"whi": 0.18, "...": 0.0},
    "performance_metrics": {"macro_f1": 0.0, "...": 0.0}
  }
]
```

#### `GET /monitoring/drift` — served data against training data

```sh
curl "$BASE/monitoring/drift"
```

```json
[
  {"target": "comfort_climate", "experiment": "<champion experiment>",
   "status": "insufficient data: at least 30 distinct dates must be served first",
   "rows_observed": 3, "features_checked": 90, "drifted_features": []},
  {"target": "weather_hazard", "experiment": "<champion experiment>",
   "status": "insufficient data: at least 30 distinct dates must be served first",
   "rows_observed": 3, "features_checked": 85, "drifted_features": []}
]
```

#### Invalid requests

```sh
curl "$BASE/predict/index/comfort_climate?date=01-01-2025"   # 400: wrong date format
curl "$BASE/predict/index/comfort_climate?date=2099-01-01"   # 400: outside the supported range
curl "$BASE/predict/category/weather_hazard"                 # 422: date is missing
curl "$BASE/predict/unknown"                                 # 404: unknown path
```

```json
{"detail": "Invalid date. Use the YYYY-MM-DD format."}
```

Every endpoint can also be tried in the browser at
[`/docs`](https://three6120-26sp-at2-25945320-api.onrender.com/docs) (Swagger UI).

### Errors

| Status | When | Example `detail` |
|:-:|:--|:--|
| 400 | `date` is not a valid `YYYY-MM-DD` date | `Invalid date. Use the YYYY-MM-DD format.` |
| 400 | `date` is outside the supported range, or within ~5 days of today | `Date … is outside the supported prediction range …` |
| 404 | Unknown path | `Not Found` |
| 405 | Wrong HTTP method (e.g. POST) | `Method Not Allowed` |
| 422 | `date` is missing | FastAPI validation error listing the missing parameter |
| 502 | Open-Meteo is unreachable or cannot provide complete history | `Open-Meteo observations could not be retrieved.` |
| 503 | A model failed to load at start-up | `Prediction models are unavailable. Check the server logs.` |
| 500 | Any unexpected error (logged on the server) | `Internal server error. Please try again later.` |

## How a prediction is made

Feature engineering is **not re-implemented here**. The API loads the preprocessing pipeline
fitted in the experiments repository and runs it unchanged, so a served feature row is built by
exactly the code the models were trained on (`nucleus_ml.weather`):

```text
GET /predict/...?date=D
  1. wx.fetch_for_serving(pipeline, D)     hourly Open-Meteo data for D and the 35 days before it
                                           (requested in UTC, converted to Sydney time)
  2. wx.features_for_date(pipeline, …, D)  the fitted pipeline:
       DailyAggregator                     hourly -> daily (mean / sum / max), CCI + WHI + WHC,
                                           incomplete days masked (DST-aware)
       WeatherFeatureBuilder               same-day values, lags, rolling stats, rain totals,
                                           gust maxima, deltas, seasonal sin/cos -> one row
  3. model.predict(row)                    each model receives only its own feature columns;
                                           CCI: one model per horizon, clipped to 0–100
                                           WHC: class 0–3 mapped to its category name
```

Three safeguards keep training and serving consistent:

- **Same code:** the pipeline object is pickled from `nucleus_ml`, so the API installs the same
  `nucleus-ml` version that fitted it.
- **Same columns:** at start-up each model's recorded feature list (models may use a subset,
  chosen during feature selection) must be produced by the pipeline, or loading fails.
- **No partial data:** if Open-Meteo cannot provide complete history, the endpoint returns 502
  rather than predicting from an incomplete row.

## Repository layout

```text
app/
├── main.py                  start-up: loads the pipeline and both model bundles
├── api/routes/weather.py    the endpoints, date validation, error codes
├── core/config.py           settings (environment variables) and artefact paths
├── models/weather.py        response schemas (pydantic)
└── services/
    ├── features.py          FeatureService (pipeline + Open-Meteo) and bundle loader
    ├── comfort_climate.py   CCI predictor (+1, +2, +3 days)
    └── weather_hazard.py    WHC predictor (+7 days)
models/                      production model artefacts (below)
tests/                       endpoint and service tests (no network or artefacts needed)
Dockerfile                   the production image (used by Render)
docker-compose.yml           local Docker run with live reload
render.yaml                  Render blueprint (Docker web service, /health check)
Makefile                     run, test, artefacts, deploy, smoke
pyproject.toml, uv.lock      dependencies
```

## Model artefacts (`models/`)

```text
models/
├── preprocessing/feature_pipeline.joblib    fitted nucleus_ml preprocessing pipeline
├── comfort_climate/                         promoted CCI bundle
│   ├── <champion>_h1.joblib, _h2.joblib, _h3.joblib
│   └── metadata.json, manifest.json, drift_reference.json, promotion.json
└── weather_hazard/                          promoted WHC bundle
    ├── <champion>_h7.joblib
    └── metadata.json, manifest.json, drift_reference.json, promotion.json
```

| File | Contents |
|:--|:--|
| `manifest.json` | Model files and each model's own feature columns |
| `metadata.json` | The document served by `/model-metadata` |
| `drift_reference.json` | Training distribution of every feature used, for `/monitoring/drift` |
| `promotion.json` | Audit record: ranking, shortlist, feature selection, test scores |

### How the champions were chosen (experiments repository)

| Stage | What happens |
|:-:|:--|
| 1–2 | Baselines, ablation and Optuna hyperparameter tuning for every algorithm, scored by walk-forward cross-validation on 2000–2024 (validation years 2021–2024) |
| 3 | Shortlist the best three tuned algorithms |
| 4 | Feature-family selection per horizon with the settings fixed, then a short re-tune |
| 5 | Champion: best mean CV score (CCI: MAE averaged over horizons; WHC: macro F1) |
| 6 | Final fit on all development data (2000–2024) |
| 7 | One evaluation on the untouched 2025 test year |
| 8 | Promotion to `production/`, then copied into this repository's `models/` |

### Updating the artefacts

After rerunning the notebooks in the experiments repository, either:

- **Manually:**
  ```sh
  make artefacts                                     # assumes ../36120-26SP-AT2-25945320-experiments
  make artefacts EXPERIMENTS=/path/to/experiments    # or point to it explicitly
  ```
- **Automatically:** push the experiments repository to `main`. Its
  `sync-production-models.yml` workflow commits the new artefacts to this repository's `models/`.

Then commit, push and redeploy.

## Dependencies

Declared in `pyproject.toml` and pinned in `uv.lock`:

- `fastapi`, `uvicorn` and `pydantic` for the service;
- `nucleus-ml` (from TestPyPI) for the preprocessing pipeline and model wrappers;
- `scikit-learn`, `xgboost`, `numpy`, `pandas` and `joblib`, needed to load the pickled models;
- `requests`, for Open-Meteo.

`nucleus-ml` and the modelling libraries must match the versions that produced the artefacts, or
unpickling can fail. After publishing a new `nucleus-ml` release, pin it exactly
(`"nucleus-ml==X.Y.Z"`) and run `uv lock`.

## Configuration

Environment variables (or `.env`, see `.env.example`):

| Variable | Default | Meaning |
|:--|:--|:--|
| `MODEL_DIR` | `./models` locally, `/models` in Docker | Folder containing the artefacts |
| `PREDICTION_MIN_DATE` / `PREDICTION_MAX_DATE` | `2000-01-01` / `2035-12-31` | Accepted input dates |
| `ARCHIVE_LAG_DAYS` | `5` | Most recent days rejected because the archive is not yet complete |
| `GITHUB_URL` | this repository | Shown by `/` |
| `PORT` | `8080` | Set automatically by Render |

## Tests

```sh
make test          # or: uv run pytest
```

The predictors and Open-Meteo are replaced by fakes, so the tests need neither network access
nor real artefacts. They cover every endpoint, input validation and error codes.

## Deploy to Render

1. Push this repository to GitHub, **including `models/`**.
2. In Render choose **New → Blueprint** and select the repository. `render.yaml` defines a Docker
   web service with `/health` as its health check and `MODEL_DIR=/models`.
3. Render builds the `Dockerfile`, which installs `nucleus-ml` from TestPyPI and copies
   `models/` into the image, and sets `PORT` automatically.
4. Check the deployment: `make smoke URL=https://three6120-26sp-at2-25945320-api.onrender.com`,
   or open `/docs`.

Every push to `main` triggers a new build, including the automatic artefact sync from the
experiments repository.

## Monitoring and retraining

`GET /monitoring/drift` compares the feature rows the API has served since start-up (one per
distinct date) with each model's `drift_reference.json`. A feature is flagged when more than 10%
of its recent values fall outside the 1st–99th percentile range seen in training.

| `status` | Meaning |
|:--|:--|
| `ok` | No feature is outside its training range more often than expected |
| `drift detected` | The listed features have moved; investigate and consider retraining |
| `insufficient data: ...` | Fewer than 30 distinct dates served so far |
| `unavailable: ...` | The bundle has no drift reference |

The served-row history is kept in memory, so it starts empty after every restart.

**Retraining:** rerun the preprocessing, experiment and comparison notebooks in the experiments
repository, update the artefacts (above) and redeploy.

## Troubleshooting

| Symptom | Cause and fix |
|:--|:--|
| Predictions return 503 | Check the start-up logs. Usually a folder in `models/` is missing: run `make artefacts`. |
| `feature columns do not match` in the logs | The pipeline and a bundle come from different runs. Rerun the notebooks, then `make artefacts`. |
| `ModuleNotFoundError` / `AttributeError` while unpickling | The installed `nucleus-ml` differs from the version that fitted the artefacts. Pin the same version in `pyproject.toml` and run `uv lock`. |
| 400 for a recent date | The archive lags about five days; use an earlier date. |
| 502 | Open-Meteo is down or rate-limiting; retry later. |
| Docker starts but predictions return 503 | `.env` sets `MODEL_DIR` to a local path, overriding the container's `/models`. Comment it out. |
| First request on Render is slow | Free instances sleep when idle; the first request wakes them. |

## Use of generative AI

Generative AI (Claude, Anthropic) was used to support coding, debugging and documentation. All
generated content was reviewed, tested and validated by the student, who is responsible for the
submitted work.
