from pathlib import Path

from starlette.config import Config

config = Config(".env")

VERSION = "0.0.1"
DEBUG: bool = config("DEBUG", cast=bool, default=False)
PROJECT_NAME: str = config("PROJECT_NAME", default="Weather Intelligence API")
GITHUB_URL: str = config(
    "GITHUB_URL",
    default="https://github.com/juanvargas/36120-26SP-AT2-25945320-api",
)
USE_DUMMY_PREDICTIONS: bool = config(
    "USE_DUMMY_PREDICTIONS", cast=bool, default=False
)
PREDICTION_MIN_DATE: str = config("PREDICTION_MIN_DATE", default="2000-01-01")
PREDICTION_MAX_DATE: str = config("PREDICTION_MAX_DATE", default="2035-12-31")
MODEL_DIR = Path(
    config("MODEL_DIR", default=str(Path(__file__).resolve().parents[2] / "ml" / "model"))
)
