"""Application settings, read from environment variables (or a local ``.env`` file)."""

from pathlib import Path

from starlette.config import Config

config = Config(".env")

VERSION = "1.0.0"
DEBUG: bool = config("DEBUG", cast=bool, default=False)
PROJECT_NAME: str = config("PROJECT_NAME", default="Weather Intelligence API")
STUDENT_NAME = "Juan Vargas T."
STUDENT_ID = "25945320"
GITHUB_URL: str = config(
    "GITHUB_URL",
    default="https://github.com/Sebas102507/36120-26SP-AT2-25945320-api",
)

# Accepted input dates. The Open-Meteo archive lags real time by about five days, so dates
# closer to today than ARCHIVE_LAG_DAYS are rejected with a clear message instead of failing
# later with incomplete observations.
PREDICTION_MIN_DATE: str = config("PREDICTION_MIN_DATE", default="2000-01-01")
PREDICTION_MAX_DATE: str = config("PREDICTION_MAX_DATE", default="2035-12-31")
ARCHIVE_LAG_DAYS: int = config("ARCHIVE_LAG_DAYS", cast=int, default=5)

# Production artefacts, copied from the experiments repository with `make artefacts` (see README):
#   models/preprocessing/feature_pipeline.joblib   the fitted nucleus_ml preprocessing pipeline
#   models/comfort_climate/                         the promoted CCI bundle
#   models/weather_hazard/                          the promoted WHC bundle
# In the Docker image they live in /models (MODEL_DIR), apart from the application code.
MODEL_DIR = Path(
    config("MODEL_DIR", default=str(Path(__file__).resolve().parents[2] / "models"))
)
PIPELINE_PATH = MODEL_DIR / "preprocessing" / "feature_pipeline.joblib"
CCI_MODEL_DIR = MODEL_DIR / "comfort_climate"
WHC_MODEL_DIR = MODEL_DIR / "weather_hazard"
