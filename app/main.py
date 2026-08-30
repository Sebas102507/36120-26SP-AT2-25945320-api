from contextlib import asynccontextmanager

from api.routes.weather import router as weather_router
from core.config import DEBUG, PROJECT_NAME, USE_DUMMY_PREDICTIONS, VERSION
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.prediction_models_ready = USE_DUMMY_PREDICTIONS
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
