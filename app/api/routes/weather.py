from datetime import date, datetime, timedelta
from typing import List

from core.config import (
    GITHUB_URL,
    PREDICTION_MAX_DATE,
    PREDICTION_MIN_DATE,
    VERSION,
)
from fastapi import APIRouter, HTTPException, Query, Request, status
from models.weather import (
    ForecastResponse,
    HealthCheckResponse,
    ModelMetadata,
    ProjectOverview,
)

router = APIRouter()

DATE_FORMAT = "%Y-%m-%d"
FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",
    "pressure",
]


def parse_forecast_date(value: str) -> date:
    """Parse and range-check a forecast date supplied as a query parameter."""
    try:
        parsed_date = datetime.strptime(value, DATE_FORMAT).date()
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date. Use the YYYY-MM-DD format.",
        ) from error

    min_date = datetime.strptime(PREDICTION_MIN_DATE, DATE_FORMAT).date()
    max_date = datetime.strptime(PREDICTION_MAX_DATE, DATE_FORMAT).date()
    if not min_date <= parsed_date <= max_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Date is outside the supported prediction range: "
                f"{min_date.isoformat()} to {max_date.isoformat()}."
            ),
        )
    return parsed_date


def require_models_ready(request: Request) -> None:
    """Prevent prediction requests when non-dummy model loading failed."""
    if not getattr(request.app.state, "prediction_models_ready", True):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction models are unavailable. Please try again later.",
        )


@router.get("/", response_model=ProjectOverview, tags=["project"])
async def project_overview() -> ProjectOverview:
    return ProjectOverview(
        title="Weather Intelligence API",
        description="Forecasts comfort climate and weather hazard conditions.",
        version=VERSION,
        github_url=GITHUB_URL,
        endpoints=[
            "GET /",
            "GET /health",
            "GET /predict/index/comfort_climate?date=YYYY-MM-DD",
            "GET /predict/category/weather_hazard?date=YYYY-MM-DD",
            "GET /model-metadata",
        ],
        prediction_targets=[
            {
                "target": "comfort_climate",
                "description": "Comfort Climate Index (CCI), D+1 through D+3.",
            },
            {
                "target": "weather_hazard",
                "description": "Weather Hazard Category (WHC), D+7.",
            },
        ],
    )


@router.get("/health", response_model=HealthCheckResponse, tags=["health"])
async def health_check() -> HealthCheckResponse:
    """Return a cheap liveness response without model or database access."""
    return HealthCheckResponse(
        status="healthy", message="Weather Intelligence API is running"
    )


@router.get(
    "/predict/index/comfort_climate",
    response_model=ForecastResponse,
    tags=["predictions"],
)
async def predict_comfort_climate(
    request: Request,
    date: str = Query(..., description="Forecast base date (YYYY-MM-DD)"),
) -> ForecastResponse:
    input_date = parse_forecast_date(date)
    require_models_ready(request)
    dummy_values = (72, 76, 81)
    return ForecastResponse(
        input_date=input_date,
        predictions={
            "comfort_climate": {
                (input_date + timedelta(days=offset)).isoformat(): value
                for offset, value in enumerate(dummy_values, start=1)
            }
        },
    )


@router.get(
    "/predict/category/weather_hazard",
    response_model=ForecastResponse,
    tags=["predictions"],
)
async def predict_weather_hazard(
    request: Request,
    date: str = Query(..., description="Forecast base date (YYYY-MM-DD)"),
) -> ForecastResponse:
    input_date = parse_forecast_date(date)
    require_models_ready(request)
    return ForecastResponse(
        input_date=input_date,
        predictions={
            "weather_hazard": {
                (input_date + timedelta(days=7)).isoformat(): "Moderate Risk"
            }
        },
    )


@router.get("/model-metadata", response_model=List[ModelMetadata], tags=["models"])
async def model_metadata() -> List[ModelMetadata]:
    return [
        ModelMetadata(
            target="comfort_climate",
            prediction_type="regression",
            algorithm="Random Forest Regressor",
            forecast_horizon="D+1 to D+3",
            feature_list=FEATURES,
            feature_importances={
                "temperature": 0.34,
                "humidity": 0.25,
                "wind_speed": 0.16,
                "precipitation": 0.14,
                "pressure": 0.11,
            },
            performance_metrics={"rmse": 5.8, "mae": 4.3},
        ),
        ModelMetadata(
            target="weather_hazard",
            prediction_type="multiclass classification",
            algorithm="Random Forest Classifier",
            forecast_horizon="D+7",
            feature_list=FEATURES,
            feature_importances={
                "precipitation": 0.31,
                "wind_speed": 0.27,
                "temperature": 0.18,
                "humidity": 0.15,
                "pressure": 0.09,
            },
            performance_metrics={"accuracy": 0.84, "macro_f1": 0.79},
        ),
    ]
