# Weather Intelligence API

A FastAPI service that forecasts the Comfort Climate Index (CCI) and provides a
placeholder Weather Hazard Category (WHC) forecast.

## Run locally

```sh
uv sync --extra dev
PYTHONPATH=app uv run uvicorn main:app --reload --port 8080
```

Open <http://localhost:8080/docs> for Swagger UI, or
<http://localhost:8080/> for the project overview.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Project overview and self-documentation |
| GET | `/health` | Cheap liveness check |
| GET | `/predict/index/comfort_climate?date=YYYY-MM-DD` | CCI forecast for D+1 through D+3 |
| GET | `/predict/category/weather_hazard?date=YYYY-MM-DD` | WHC label for D+7 |
| GET | `/model-metadata` | Model targets, features, and metrics |

## Configuration

The CCI endpoint loads its model according to `ml/model/cci_model_manifest.json`
when the application starts. It supports TorchScript neural models (LSTM/GRU)
and joblib tabular models (such as Random Forest and XGBoost), each with their
own declared preprocessing and scaler artefacts. For a requested base date it
retrieves the required Sydney hourly history from Open-Meteo and returns
forecasts for D+1 through D+3.

```sh
MODEL_DIR=ml/model
```

If the model artefacts cannot be loaded, prediction endpoints return
`503 Service Unavailable`; `/health` continues to return immediately. If
Open-Meteo cannot provide complete history for a CCI request, the endpoint
returns `502 Bad Gateway`.

Configure accepted forecast input dates when needed:

```sh
PREDICTION_MIN_DATE=2000-01-01
PREDICTION_MAX_DATE=2035-12-31
```

## Tests

```sh
uv run pytest
```

## Deploy to Render

1. Push this project to a GitHub repository.
2. In Render, select **New → Blueprint** and select that repository.
3. Render reads `render.yaml`, builds the Docker image, and uses `/health` for
   health checks.
4. Once deployment completes, open `https://<your-service>.onrender.com/docs`.

The container reads Render's `PORT` environment variable automatically.
