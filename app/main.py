from contextlib import asynccontextmanager
import logging

from api.routes.weather import router as weather_router
from core.config import DEBUG, MODEL_DIR, PROJECT_NAME, VERSION
from fastapi import FastAPI
from services.comfort_climate import (
    ComfortClimatePredictor,
    load_experiment_metadata,
    load_model_manifest,
)
from services.weather_hazard import WeatherHazardPredictor

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.cci_predictor = None
    application.state.cci_manifest = None
    application.state.cci_metadata = None
    application.state.whc_predictor = None
    try:
        manifest = load_model_manifest(MODEL_DIR)
        application.state.cci_predictor = ComfortClimatePredictor.from_model_directory(MODEL_DIR)
        application.state.cci_manifest = manifest
        application.state.cci_metadata = load_experiment_metadata(MODEL_DIR, manifest)
        application.state.prediction_models_ready = True
        logger.info("Loaded CCI production artefacts from %s", MODEL_DIR)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        application.state.prediction_models_ready = False
        logger.exception("CCI production model could not be loaded: %s", error)
    try:
        application.state.whc_predictor = WeatherHazardPredictor.from_model_directory(MODEL_DIR)
        application.state.weather_hazard_model_ready = True
        logger.info("Loaded WHC production artefacts from %s", MODEL_DIR)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        application.state.weather_hazard_model_ready = False
        logger.exception("WHC production model could not be loaded: %s", error)
    yield


def get_application() -> FastAPI:
    application = FastAPI(
        title=PROJECT_NAME,
        debug=DEBUG,
        version=VERSION,
        lifespan=lifespan,
        redoc_url=None,
    )
    application.state.prediction_models_ready = True
    application.include_router(weather_router)
    return application


app = get_application()
