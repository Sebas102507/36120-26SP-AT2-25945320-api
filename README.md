# Weather Intelligence API

FastAPI service that serves two machine learning models for **Sydney, NSW**, built from the
Open-Meteo Historical Weather API:

| Target | Type | Horizon | Endpoint |
|:--|:--|:--|:--|
| Climate Comfort Index (CCI, 0–100) | Regression | +1, +2, +3 days | `/predict/index/comfort_climate` |
| Weather Hazard Category (WHC) | Multiclass classification | exactly +7 days | `/predict/category/weather_hazard` |

| | |
|:--|:--|
| **Student name** | Juan Vargas T. |
| **Student ID** | 25945320 |
| **Subject** | 36120 Advanced Machine Learning Application — Assessment Task 2 |
| **API repository** | https://github.com/Sebas102507/36120-26SP-AT2-25945320-api |
| **Experiments repository** | https://github.com/Sebas102507/36120-26SP-AT2-25945320-experiments |
| **Custom package** | [`nucleus-ml`](https://test.pypi.org/project/nucleus-ml/) on TestPyPI |

---

## How a prediction is made

Feature engineering is **not re-implemented here**. The API loads the preprocessing pipeline
that was fitted in the experiments repository and runs it unchanged, so a served feature row is
built by exactly the code the models were trained on (`nucleus_ml.weather`):

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

- **Same code:** the pipeline object is pickled from `nucleus_ml`, so the API needs the same
  `nucleus-ml` version that fitted it.
- **Same columns:** at start-up each model's recorded feature list (models may use a subset of
  the features, chosen during refinement) must be produced by the pipeline, or loading fails.
- **No partial data:** if Open-Meteo cannot provide 30 complete days of history, the endpoint
  returns 502 rather than predicting from an incomplete row.

## Repository layout

```text
app/
├── main.py                  start-up: loads the pipeline and both model bundles
├── api/routes/weather.py    the five endpoints, date validation, error codes
├── core/config.py           settings (env vars) and artefact paths
├── models/weather.py        response schemas
└── services/
    ├── features.py          FeatureService (pipeline + Open-Meteo) and ModelBundle loader
    ├── comfort_climate.py   CCI predictor (+1, +2, +3 days)
    └── weather_hazard.py    WHC predictor (+7 days)
ml/model/                    production artefacts (see below)
tests/                       endpoint and service tests (no network or artefacts needed)
```

## Production artefacts

The API expects this layout under `ml/model/`:

```text
ml/model/
├── preprocessing/feature_pipeline.joblib    fitted nucleus_ml preprocessing pipeline
├── comfort_climate/                         promoted CCI bundle
│   ├── <experiment>_h1.joblib, _h2.joblib, _h3.joblib
│   └── metadata.json, manifest.json, drift_reference.json, promotion.json
└── weather_hazard/                          promoted WHC bundle
    ├── <experiment>_h7.joblib
    └── metadata.json, manifest.json, drift_reference.json, promotion.json
```

`manifest.json` lists each model's own feature columns; `drift_reference.json` holds the training
distribution of every feature used, for monitoring.

### How the champion is chosen (experiments repository)

| Stage | What happens | Where |
|:-:|:--|:--|
| 1–2 | Baselines, then every algorithm screened with default settings on the shared features | Experiment notebooks |
| 3 | Shortlist the best three | Comparison notebook |
| 4 | Refinement: hyperparameters and feature families tuned together with walk-forward CV | Comparison notebook |
| 5 | Champion: best mean CV score | Comparison notebook |
| 6 | Final fit on train + validation | Comparison notebook |
| 7 | One test evaluation | Comparison notebook |
| 8 | Promotion to `production/` with a drift reference; served and monitored by this API | Comparison notebook + API |

After running the preprocessing, experiment and comparison notebooks in the experiments
repository, copy the artefacts with:

```sh
make artefacts                                     # assumes ../36120-26SP-AT2-25945320-experiments
make artefacts EXPERIMENTS=/path/to/experiments    # or point to it explicitly
```

## Run locally

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
cp .env.example .env
uv sync --extra dev          # installs nucleus-ml from TestPyPI, everything else from PyPI
PYTHONPATH=app uv run uvicorn main:app --reload --port 8080
```

Open <http://localhost:8080/docs> for the interactive Swagger documentation.

With Docker instead: `make deploy` (builds and starts the container on port 8080), `make down`
to stop it.

## Endpoints

| Method | Path | Description |
|:--|:--|:--|
| GET | `/` | Project overview, endpoints, targets and repository link |
| GET | `/health` | Liveness check (works even if the models failed to load) |
| GET | `/predict/index/comfort_climate?date=YYYY-MM-DD` | CCI for the three days after `date` |
| GET | `/predict/category/weather_hazard?date=YYYY-MM-DD` | WHC for the day seven days after `date` |
| GET | `/model-metadata` | Algorithm, horizon, features, importances and validation metrics |
| GET | `/monitoring/drift` | Whether recently served data still looks like the training data |

### Examples

```sh
curl "http://localhost:8080/predict/index/comfort_climate?date=2025-01-01"
```

```json
{
  "input_date": "2025-01-01",
  "predictions": {"comfort_climate": {"2025-01-02": 72, "2025-01-03": 76, "2025-01-04": 81}}
}
```

```sh
curl "http://localhost:8080/predict/category/weather_hazard?date=2025-01-01"
```

```json
{
  "input_date": "2025-01-01",
  "predictions": {"weather_hazard": {"2025-01-08": "Moderate Risk"}}
}
```

### Errors

| Status | When | Example `detail` |
|:-:|:--|:--|
| 400 | `date` is not a valid `YYYY-MM-DD` date | `Invalid date. Use the YYYY-MM-DD format.` |
| 400 | `date` is outside the supported range, or within ~5 days of today (the archive lags real time) | `Date … is outside the supported prediction range …` |
| 422 | `date` is missing | FastAPI validation error |
| 502 | Open-Meteo is unreachable or cannot provide complete history | `Open-Meteo observations could not be retrieved.` |
| 503 | A model failed to load at start-up | `Prediction models are unavailable. …` |
| 500 | Any unexpected error (logged on the server) | `Internal server error. …` |

## Monitoring and retraining

`GET /monitoring/drift` compares the feature rows the API has served since start-up (one per
distinct date) with each model's `drift_reference.json`. A feature is flagged when more than 10%
of its recent values fall outside the 1st–99th percentile range seen in training, meaning the model
is being asked about conditions it was not trained on.

| `status` | Meaning |
|:--|:--|
| `ok` | No feature is outside its training range more often than expected |
| `drift detected` | Listed features have moved; investigate and consider retraining |
| `insufficient data: ...` | Fewer than 30 distinct dates served so far |
| `unavailable: ...` | The bundle has no drift reference (promoted by an older version) |

The served-row history is kept in memory, so it restarts empty whenever the service restarts.

**Retraining:** rerun the preprocessing, experiment and comparison notebooks in the experiments
repository, run `make artefacts`, and redeploy.

## Configuration

Environment variables (or `.env`, see `.env.example`):

| Variable | Default | Meaning |
|:--|:--|:--|
| `MODEL_DIR` | `ml/model` | Folder containing the artefacts |
| `PREDICTION_MIN_DATE` / `PREDICTION_MAX_DATE` | `2000-01-01` / `2035-12-31` | Accepted input dates |
| `ARCHIVE_LAG_DAYS` | `5` | Most recent days rejected because the archive is not yet complete |
| `GITHUB_URL` | this repository | Shown by `/` |

## Tests

```sh
make test          # or: uv run pytest
```

Predictors and Open-Meteo are replaced by fakes, so the tests need neither network access nor
real artefacts.

## Deploy to Render

1. Push this repository to GitHub, **including `ml/model/`**.
2. In Render choose **New → Blueprint** and select the repository; `render.yaml` defines a
   Docker web service with `/health` as its health check.
3. Render builds the `Dockerfile` (which installs `nucleus-ml` from TestPyPI) and sets `PORT`
   automatically.
4. Open `https://<your-service>.onrender.com/docs`.

## Troubleshooting

| Symptom | Cause and fix |
|:--|:--|
| Predictions return 503 | Check the logs at start-up. Usually `ml/model/` is missing a folder — run `make artefacts`. |
| `feature columns do not match` in the logs | The pipeline and a model bundle come from different runs. Rerun the notebooks, then `make artefacts`. |
| `ModuleNotFoundError` / `AttributeError` while unpickling | The installed `nucleus-ml` differs from the version that fitted the artefacts. Pin the same version in `pyproject.toml`. |
| 502 for recent dates | The archive lags about five days; use an earlier date. |
| Free Render instance slow on the first request | It sleeps when idle; the first request after a pause wakes it. |
