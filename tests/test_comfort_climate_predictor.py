from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from app.services.comfort_climate import ComfortClimatePredictor, WeatherDataError


MODEL_DIR = Path(__file__).resolve().parents[1] / "ml" / "model"


class FakeResponse:
    def __init__(self, hourly: dict[str, list[object]]):
        self.hourly = hourly

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, dict[str, list[object]]]:
        return {"hourly": self.hourly}


def complete_hourly_payload(start: str, periods: int = 14 * 24) -> dict[str, list[object]]:
    timestamps = pd.date_range(start, periods=periods, freq="h")
    return {
        "time": timestamps.strftime("%Y-%m-%dT%H:%M").tolist(),
        "temperature_2m": [22.0] * periods,
        "relative_humidity_2m": [50.0] * periods,
        "wind_speed_10m": [10.0] * periods,
        "cloud_cover": [20.0] * periods,
        "precipitation": [0.0] * periods,
        "wind_gusts_10m": [20.0] * periods,
    }


@pytest.fixture
def predictor() -> ComfortClimatePredictor:
    return ComfortClimatePredictor.from_model_directory(MODEL_DIR)


def test_predictor_fetches_14_days_and_returns_d1_through_d3(monkeypatch, predictor):
    requested_params = {}

    def fake_get(url, params, timeout):
        requested_params.update(params)
        return FakeResponse(complete_hourly_payload("2025-01-01"))

    monkeypatch.setattr("app.services.comfort_climate.requests.get", fake_get)

    result = predictor.predict(date(2025, 1, 14))

    assert requested_params["start_date"] == "2025-01-01"
    assert requested_params["end_date"] == "2025-01-14"
    assert result.keys() == {"2025-01-15", "2025-01-16", "2025-01-17"}
    assert all(0 <= value <= 100 for value in result.values())


def test_predictor_rejects_incomplete_daily_observations(monkeypatch, predictor):
    payload = complete_hourly_payload("2025-01-01", periods=14 * 24 - 1)
    monkeypatch.setattr(
        "app.services.comfort_climate.requests.get",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    with pytest.raises(WeatherDataError, match="Complete hourly observations"):
        predictor.predict(date(2025, 1, 14))


def test_joblib_manifest_uses_three_tabular_horizon_models(monkeypatch, tmp_path):
    model_files = []
    for horizon, value in enumerate((60, 61, 62), start=1):
        model = DummyRegressor(strategy="constant", constant=value).fit([[0]], [value])
        filename = f"cci_h{horizon}.joblib"
        joblib.dump(model, tmp_path / filename)
        model_files.append(filename)

    (tmp_path / "cci_model_manifest.json").write_text(
        """{
          "version": 1,
          "target": "comfort_climate",
          "model_family": "random_forest",
          "adapter": "joblib",
          "input_kind": "tabular",
          "model_files": ["cci_h1.joblib", "cci_h2.joblib", "cci_h3.joblib"],
          "feature_columns": ["cci"],
          "lookback_days": 30,
          "target_inversion": {"type": "none"}
        }"""
    )
    monkeypatch.setattr(
        "app.services.comfort_climate.requests.get",
        lambda *args, **kwargs: FakeResponse(complete_hourly_payload("2024-12-16", 30 * 24)),
    )

    predictor = ComfortClimatePredictor.from_model_directory(tmp_path)

    assert predictor.predict(date(2025, 1, 14)) == {
        "2025-01-15": 60,
        "2025-01-16": 61,
        "2025-01-17": 62,
    }
