"""Weather Intelligence API: FastAPI entry point.

At start-up the service loads three production artefacts produced by the experiments repository:

1. the fitted ``nucleus_ml`` preprocessing pipeline (hourly observations -> feature row),
2. the promoted Climate Comfort Index bundle (one model per horizon: +1, +2, +3 days),
3. the promoted Weather Hazard Category bundle (one model for +7 days).

If an artefact cannot be loaded, the error is logged and the matching prediction endpoints
answer 503, while ``/`` and ``/health`` keep working.
"""

import logging
from contextlib import asynccontextmanager

from api.routes.weather import router as weather_router
from core.config import CCI_MODEL_DIR, DEBUG, PIPELINE_PATH, PROJECT_NAME, VERSION, WHC_MODEL_DIR
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from services.comfort_climate import ComfortClimatePredictor
from services.features import FeatureService
from services.weather_hazard import WeatherHazardPredictor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def load_predictors(application: FastAPI) -> None:
    """Load the pipeline and both model bundles into ``application.state``."""
    application.state.cci_predictor = None
    application.state.whc_predictor = None
    try:
        features = FeatureService.from_file(PIPELINE_PATH)
    except Exception:  # noqa: BLE001 - any failure must leave /health alive
        logger.exception("Preprocessing pipeline could not be loaded from %s", PIPELINE_PATH)
        return

    for name, predictor_class, directory in (
        ("cci_predictor", ComfortClimatePredictor, CCI_MODEL_DIR),
        ("whc_predictor", WeatherHazardPredictor, WHC_MODEL_DIR),
    ):
        try:
            setattr(application.state, name, predictor_class.from_directory(directory, features))
            logger.info("Loaded %s from %s", name, directory)
        except Exception:  # noqa: BLE001
            logger.exception("%s could not be loaded from %s", name, directory)


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_predictors(application)
    yield


def get_application() -> FastAPI:
    application = FastAPI(
        title=PROJECT_NAME,
        description="Climate Comfort Index and Weather Hazard Category forecasts for Sydney.",
        version=VERSION,
        debug=DEBUG,
        lifespan=lifespan,
        redoc_url=None,
    )
    application.include_router(weather_router)

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        """Return a JSON 500 instead of a bare traceback for anything unforeseen."""
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "Internal server error. Please try again later."})

    return application


app = get_application()
