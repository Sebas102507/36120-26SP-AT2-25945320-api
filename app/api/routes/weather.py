from datetime import date, datetime, timedelta
import logging
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
from services.comfort_climate import WeatherDataError

router = APIRouter()
logger = logging.getLogger(__name__)

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


def require_weather_hazard_model(request: Request) -> None:
    if not getattr(request.app.state, "weather_hazard_model_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weather hazard prediction model is unavailable. Please try again later.",
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
    logger.info("Received comfort_climate prediction request for %s", input_date)
    require_models_ready(request)
    predictor = request.app.state.cci_predictor
    try:
        predictions = predictor.predict(input_date)
    except WeatherDataError as error:
        logger.warning("CCI prediction failed for %s: %s", input_date, error)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error
    return ForecastResponse(
        input_date=input_date,
        predictions={"comfort_climate": predictions},
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
    require_weather_hazard_model(request)
    predictor = request.app.state.whc_predictor
    try:
        predictions = predictor.predict(input_date)
    except WeatherDataError as error:
        logger.warning("WHC prediction failed for %s: %s", input_date, error)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error
    return ForecastResponse(
        input_date=input_date,
        predictions={"weather_hazard": predictions},
    )


@router.get("/model-metadata", response_model=List[ModelMetadata], tags=["models"])
async def model_metadata(request: Request) -> List[ModelMetadata]:
    manifest = getattr(request.app.state, "cci_manifest", None)
    metadata = getattr(request.app.state, "cci_metadata", None)
    cci_model = _cci_model_metadata(manifest, metadata)
    return [
        cci_model,
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


def _cci_model_metadata(
    manifest: dict | None, metadata: dict | None
) -> ModelMetadata:
    """Translate the selected experiment bundle into the API metadata schema."""
    if not manifest or not metadata:
        return ModelMetadata(
            target="comfort_climate",
            prediction_type="regression",
            algorithm="Unavailable",
            forecast_horizon="D+1 to D+3",
            feature_list=[],
            feature_importances={},
            performance_metrics={},
        )
    horizons = manifest.get("horizons_days", [1, 2, 3])
    feature_list = metadata.get("sequence_features") or metadata.get("feature_columns", [])
    validation = metadata.get("validation_metrics", {})
    if not validation:
        validation = {
            model["target_column"]: model["validation_metrics"]
            for model in metadata.get("models", [])
            if "target_column" in model and "validation_metrics" in model
        }
    metric_names = ("mae", "rmse")
    metrics = {
        name: sum(values[name] for values in validation.values()) / len(validation)
        for name in metric_names
        if validation and all(name in values for values in validation.values())
    }
    return ModelMetadata(
        target=str(manifest["target"]),
        prediction_type=str(metadata.get("prediction_type", "regression")),
        algorithm=str(metadata.get("architecture") or metadata.get("algorithm", "Unknown")),
        forecast_horizon=f"D+{horizons[0]} to D+{horizons[-1]}",
        feature_list=feature_list,
        feature_importances={},
        performance_metrics=metrics,
    )
