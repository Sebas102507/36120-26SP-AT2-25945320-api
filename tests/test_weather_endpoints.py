import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from main import get_application
from services.comfort_climate import WeatherDataError


def test_legacy_route_is_disabled_and_swagger_is_available():
    client = TestClient(get_application())

    assert client.get("/api/v1/predict").status_code == 404
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200

def test_project_overview_and_health_are_available():
    client = TestClient(get_application())

    overview = client.get("/")
    health = client.get("/health")

    assert overview.status_code == 200
    assert overview.json()["title"] == "Weather Intelligence API"
    assert "GET /model-metadata" in overview.json()["endpoints"]
    assert health.status_code == 200
    assert health.json() == {
        "status": "healthy",
        "message": "Weather Intelligence API is running",
    }


def test_comfort_climate_forecast_returns_d1_through_d3():
    app = get_application()
    with TestClient(app) as client:
        app.state.cci_predictor.predict = lambda _: {
            "2025-01-02": 72,
            "2025-01-03": 76,
            "2025-01-04": 81,
        }

        response = client.get("/predict/index/comfort_climate?date=2025-01-01")

        assert response.status_code == 200
        assert response.json() == {
            "input_date": "2025-01-01",
            "predictions": {
                "comfort_climate": {
                    "2025-01-02": 72,
                    "2025-01-03": 76,
                    "2025-01-04": 81,
                }
            },
        }


def test_weather_hazard_forecast_returns_d7_label():
    app = get_application()
    with TestClient(app) as client:
        app.state.whc_predictor.predict = lambda _: {"2025-01-08": "Moderate Risk"}

        response = client.get("/predict/category/weather_hazard?date=2025-01-01")

        assert response.status_code == 200
        assert response.json()["predictions"]["weather_hazard"] == {
            "2025-01-08": "Moderate Risk"
        }


def test_prediction_date_errors_are_clear():
    client = TestClient(get_application())

    malformed = client.get("/predict/index/comfort_climate?date=01-01-2025")
    unsupported = client.get("/predict/category/weather_hazard?date=2099-01-01")
    missing = client.get("/predict/index/comfort_climate")

    assert malformed.status_code == 400
    assert malformed.json()["detail"] == "Invalid date. Use the YYYY-MM-DD format."
    assert unsupported.status_code == 400
    assert "outside the supported prediction range" in unsupported.json()["detail"]
    assert missing.status_code == 422


def test_prediction_returns_503_when_models_are_unavailable():
    app = get_application()
    with TestClient(app) as client:
        app.state.prediction_models_ready = False

        response = client.get("/predict/index/comfort_climate?date=2025-01-01")

        assert response.status_code == 503
        assert "Prediction models are unavailable" in response.json()["detail"]


def test_comfort_climate_returns_502_when_weather_data_is_unavailable():
    app = get_application()
    with TestClient(app) as client:
        def unavailable(_):
            raise WeatherDataError("Open-Meteo observations could not be retrieved.")

        app.state.cci_predictor.predict = unavailable
        response = client.get("/predict/index/comfort_climate?date=2025-01-01")

    assert response.status_code == 502
    assert response.json()["detail"] == "Open-Meteo observations could not be retrieved."


def test_model_metadata_describes_both_targets():
    app = get_application()
    with TestClient(app) as client:
        response = client.get("/model-metadata")

        assert response.status_code == 200
        models = response.json()
        assert {model["target"] for model in models} == {
            "comfort_climate",
            "weather_hazard",
        }
        assert models[0]["algorithm"] == "Polynomial Ridge Regression"
        assert {"rmse", "mae"} <= models[0]["performance_metrics"].keys()
        assert {"accuracy", "macro_f1"} <= models[1]["performance_metrics"].keys()
