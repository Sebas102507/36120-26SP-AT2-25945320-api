# Weather Intelligence API

A FastAPI service that provides dummy forecasts for the Comfort Climate Index
(CCI) and Weather Hazard Category (WHC).

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

The API returns deterministic dummy predictions by default:

```sh
USE_DUMMY_PREDICTIONS=true
```

Set it to `false` only after model loading is implemented. While it is false,
prediction endpoints intentionally return `503 Service Unavailable`; `/health`
continues to return immediately.

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
