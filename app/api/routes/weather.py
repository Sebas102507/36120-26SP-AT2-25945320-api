"""HTTP endpoints of the Weather Intelligence API.

| Method | Path                               | Purpose                               |
|--------|------------------------------------|---------------------------------------|
| GET    | /                                  | Project overview (self-documentation) |
| GET    | /health                            | Liveness check                        |
| GET    | /predict/index/comfort_climate     | CCI for the next three days           |
| GET    | /predict/category/weather_hazard   | WHC exactly seven days ahead          |
| GET    | /model-metadata                    | Metadata of both deployed models      |
| GET    | /monitoring/drift                  | Served data vs training distribution  |

Error responses: 400 for a malformed or unsupported date, 422 when ``date`` is missing,
502 when Open-Meteo cannot supply the required history, 503 when a model is not loaded.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from core.config import (
    ARCHIVE_LAG_DAYS,
    GITHUB_URL,
    PREDICTION_MAX_DATE,
    PREDICTION_MIN_DATE,
    PROJECT_NAME,
    STUDENT_ID,
    STUDENT_NAME,
    VERSION,
)
from fastapi import APIRouter, HTTPException, Query, Request
from models.weather import (
    DriftReport,
    ForecastResponse,
    HealthCheckResponse,
    ModelMetadata,
    ProjectOverview,
)
from services.features import WeatherDataError

router = APIRouter()

ENDPOINTS = [
    "/",
    "/health",
    "/predict/index/comfort_climate",
    "/predict/category/weather_hazard",
    "/model-metadata",
    "/monitoring/drift",
]
DATE_QUERY = Query(..., description="Base date in YYYY-MM-DD format.", examples=["2025-01-01"])


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

def parse_date(value: str) -> date:
    """Parse ``YYYY-MM-DD`` and check it lies inside the supported range (400 otherwise)."""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise HTTPException(status_code=400,
                            detail="Invalid date. Use the YYYY-MM-DD format.") from error

    earliest = date.fromisoformat(PREDICTION_MIN_DATE)
    latest = min(date.fromisoformat(PREDICTION_MAX_DATE),
                 date.today() - timedelta(days=ARCHIVE_LAG_DAYS))
    if not earliest <= parsed <= latest:
        raise HTTPException(
            status_code=400,
            detail=(f"Date {parsed.isoformat()} is outside the supported prediction range "
                    f"({earliest.isoformat()} to {latest.isoformat()}). The Open-Meteo archive "
                    f"lags real time by about {ARCHIVE_LAG_DAYS} days."),
        )
    return parsed


def get_predictor(request: Request, name: str):
    """Return a loaded predictor from application state, or fail with 503."""
    predictor = getattr(request.app.state, name, None)
    if predictor is None:
        raise HTTPException(status_code=503,
                            detail="Prediction models are unavailable. Check the server logs.")
    return predictor


def forecast(request: Request, predictor_name: str, target: str, value: str) -> ForecastResponse:
    """Shared body of both prediction endpoints."""
    day = parse_date(value)
    predictor = get_predictor(request, predictor_name)
    try:
        predictions = predictor.predict(day)
    except WeatherDataError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return ForecastResponse(input_date=day, predictions={target: predictions})


# ----------------------------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------------------------

@router.get("/", response_model=ProjectOverview, tags=["overview"])
def project_overview() -> ProjectOverview:
    """Project title, description, endpoints, prediction targets and repository link."""
    return ProjectOverview(
        project=PROJECT_NAME,
        description=(
            "Machine learning services for Sydney weather forecasting: a regression model for "
            "the Climate Comfort Index over the next three days and a classifier for the "
            "Weather Hazard Category seven days ahead, both built from Open-Meteo "
            "historical observations. Interactive documentation is available at /docs."
        ),
        version=VERSION,
        student=f"{STUDENT_NAME} ({STUDENT_ID})",
        github=GITHUB_URL,
        endpoints=ENDPOINTS,
        prediction_targets=[
            {"target": "Climate Comfort Index (0-100)",
             "endpoint": "/predict/index/comfort_climate?date=YYYY-MM-DD",
             "prediction_type": "regression", "forecast_horizon": "+1, +2 and +3 days"},
            {"target": "Weather Hazard Category",
             "endpoint": "/predict/category/weather_hazard?date=YYYY-MM-DD",
             "prediction_type": "multiclass classification", "forecast_horizon": "+7 days"},
        ],
    )


@router.get("/health", response_model=HealthCheckResponse, tags=["overview"])
def health() -> HealthCheckResponse:
    """Liveness check. Does not depend on the models, so it answers even if they fail to load."""
    return HealthCheckResponse(status="healthy", message=f"{PROJECT_NAME} is running")


@router.get("/predict/index/comfort_climate", response_model=ForecastResponse,
            tags=["predictions"])
def predict_comfort_climate(request: Request, date: str = DATE_QUERY) -> ForecastResponse:
    """Climate Comfort Index (0-100) for the three days after ``date``."""
    return forecast(request, "cci_predictor", "comfort_climate", date)


@router.get("/predict/category/weather_hazard", response_model=ForecastResponse,
            tags=["predictions"])
def predict_weather_hazard(request: Request, date: str = DATE_QUERY) -> ForecastResponse:
    """Weather Hazard Category exactly seven days after ``date``."""
    return forecast(request, "whc_predictor", "weather_hazard", date)


@router.get("/model-metadata", response_model=list[ModelMetadata], tags=["models"])
def model_metadata(request: Request) -> list[dict]:
    """Target, algorithm, horizon, features, importances and validation metrics per model."""
    return [get_predictor(request, name).describe()
            for name in ("cci_predictor", "whc_predictor")]


@router.get("/monitoring/drift", response_model=list[DriftReport], tags=["models"])
def drift(request: Request) -> list[dict]:
    """Compare the feature rows served since start-up with each model's training data.

    A feature counts as drifted when more than 10% of its recent values fall outside the
    1st-99th percentile range seen in training: the model is then being asked about conditions
    it was not trained on, which is the signal to retrain.
    """
    return [get_predictor(request, name).drift()
            for name in ("cci_predictor", "whc_predictor")]
