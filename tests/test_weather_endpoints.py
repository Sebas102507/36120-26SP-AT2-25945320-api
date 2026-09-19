"""Endpoint tests. Predictors are replaced by fakes, so no artefacts or network are needed."""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from main import get_application
from services.features import WeatherDataError


class FakePredictor:
    """Stands in for ComfortClimatePredictor / WeatherHazardPredictor."""

    def __init__(self, result=None, error=None, description=None, drift_status=None):
        self.result, self.error, self.description = result, error, description
        self.drift_status = drift_status

    def predict(self, day):
        if self.error:
            raise self.error
        return self.result

    def describe(self):
        return self.description

    def drift(self):
        return self.drift_status


def metadata(target: str, algorithm: str, metrics: dict) -> dict:
    return {"target": target, "experiment": f"{target}_exp", "prediction_type": "regression",
            "algorithm": algorithm, "forecast_horizon": "1 Day", "feature_list": ["cci"],
            "feature_importances": {"cci": 1.0}, "performance_metrics": metrics}


def drift(target: str, status: str, drifted: list) -> dict:
    return {"target": target, "experiment": f"{target}_exp", "status": status,
            "rows_observed": 40, "features_checked": 90, "drifted_features": drifted}


@pytest.fixture
def client():
    """A client whose start-up found no artefacts; tests inject the predictors they need."""
    app = get_application()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        app.state.cci_predictor = FakePredictor(
            result={"2025-01-02": 72, "2025-01-03": 76, "2025-01-04": 81},
            description=metadata("comfort_climate", "XGBoost Regressor", {"mae": 5.1}),
            drift_status=drift("comfort_climate", "ok", []))
        app.state.whc_predictor = FakePredictor(
            result={"2025-01-08": "Moderate Risk"},
            description=metadata("weather_hazard", "Random Forest Classifier",
                                 {"macro_f1": 0.41}),
            drift_status=drift("weather_hazard", "drift detected", ["wind_gusts_10m"]))
        yield test_client, app


def test_overview_lists_every_endpoint_and_the_repository(client):
    test_client, _ = client
    body = test_client.get("/").json()

    assert body["project"] == "Weather Intelligence API"
    assert body["github"].startswith("https://github.com/")
    assert body["endpoints"] == ["/", "/health", "/predict/index/comfort_climate",
                                 "/predict/category/weather_hazard", "/model-metadata",
                                 "/monitoring/drift"]
    assert "25945320" in body["student"]


def test_health(client):
    test_client, _ = client
    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy",
                               "message": "Weather Intelligence API is running"}


def test_comfort_climate_forecast(client):
    test_client, _ = client
    response = test_client.get("/predict/index/comfort_climate?date=2025-01-01")

    assert response.status_code == 200
    assert response.json() == {
        "input_date": "2025-01-01",
        "predictions": {"comfort_climate": {"2025-01-02": 72, "2025-01-03": 76,
                                            "2025-01-04": 81}},
    }


def test_weather_hazard_forecast(client):
    test_client, _ = client
    response = test_client.get("/predict/category/weather_hazard?date=2025-01-01")

    assert response.status_code == 200
    assert response.json()["predictions"] == {"weather_hazard": {"2025-01-08": "Moderate Risk"}}


@pytest.mark.parametrize("path", ["/predict/index/comfort_climate",
                                  "/predict/category/weather_hazard"])
def test_date_validation(client, path):
    test_client, _ = client
    too_recent = (date.today() - timedelta(days=1)).isoformat()

    malformed = test_client.get(f"{path}?date=01-01-2025")
    impossible = test_client.get(f"{path}?date=2025-02-30")
    too_early = test_client.get(f"{path}?date=1990-01-01")
    recent = test_client.get(f"{path}?date={too_recent}")
    missing = test_client.get(path)

    assert malformed.status_code == 400
    assert malformed.json()["detail"] == "Invalid date. Use the YYYY-MM-DD format."
    assert impossible.status_code == 400
    assert too_early.status_code == 400
    assert "outside the supported prediction range" in too_early.json()["detail"]
    assert recent.status_code == 400
    assert missing.status_code == 422


def test_missing_models_return_503(client):
    test_client, app = client
    app.state.cci_predictor = None

    response = test_client.get("/predict/index/comfort_climate?date=2025-01-01")

    assert response.status_code == 503
    assert "Prediction models are unavailable" in response.json()["detail"]
    assert test_client.get("/model-metadata").status_code == 503
    assert test_client.get("/health").status_code == 200


def test_unavailable_weather_data_returns_502(client):
    test_client, app = client
    app.state.whc_predictor = FakePredictor(error=WeatherDataError("Open-Meteo is down."))

    response = test_client.get("/predict/category/weather_hazard?date=2025-01-01")

    assert response.status_code == 502
    assert response.json()["detail"] == "Open-Meteo is down."


def test_unexpected_errors_return_json_500(client):
    test_client, app = client
    app.state.cci_predictor = FakePredictor(error=RuntimeError("boom"))

    response = test_client.get("/predict/index/comfort_climate?date=2025-01-01")

    assert response.status_code == 500
    assert "Internal server error" in response.json()["detail"]


def test_unknown_route_returns_404(client):
    test_client, _ = client
    assert test_client.get("/predict/unknown").status_code == 404


def test_drift_monitoring_reports_each_model(client):
    test_client, _ = client
    reports = test_client.get("/monitoring/drift").json()

    assert [r["status"] for r in reports] == ["ok", "drift detected"]
    assert reports[1]["drifted_features"] == ["wind_gusts_10m"]


def test_drift_monitoring_needs_loaded_models(client):
    test_client, app = client
    app.state.whc_predictor = None

    assert test_client.get("/monitoring/drift").status_code == 503


def test_model_metadata_describes_both_targets(client):
    test_client, _ = client
    models = test_client.get("/model-metadata").json()

    assert [m["target"] for m in models] == ["comfort_climate", "weather_hazard"]
    assert models[0]["performance_metrics"] == {"mae": 5.1}
    assert models[1]["algorithm"] == "Random Forest Classifier"
