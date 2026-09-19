"""Service tests: bundle loading, both predictors and the feature service.

Tiny dummy models are written to a temporary folder in the same layout the experiments repository
exports, and Open-Meteo is replaced by fakes, so no network or real artefact is needed.
"""

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from nucleus_ml import weather as wx
from services.comfort_climate import ComfortClimatePredictor
from services.features import FeatureService, ModelBundle, WeatherDataError
from services.weather_hazard import WeatherHazardPredictor
from sklearn.dummy import DummyClassifier, DummyRegressor

COLUMNS = ["cci", "whi", "doy_sin"]
ROW = pd.DataFrame([[70.0, 20.0, 0.5]], columns=COLUMNS)


class FakeFeatures:
    """Stands in for FeatureService: returns one fixed feature row."""

    feature_columns = COLUMNS

    def features_for(self, day):
        return ROW


def write_bundle(directory, experiment, models, prediction_type="regression", manifest=None,
                 **extra):
    """Write a bundle as the experiments repo does: <experiment>_h<h>.joblib + metadata."""
    directory.mkdir()
    if manifest:
        wx.save_file(manifest, directory / "manifest.json")
    for horizon, model in models.items():
        joblib.dump(model, directory / f"{experiment}_h{horizon}.joblib")
    wx.save_file({
        "experiment": experiment,
        "algorithm": "Dummy",
        "prediction_type": prediction_type,
        "feature_columns": COLUMNS,
        "models": [{"target_column": f"t_t{h}", "validation_metrics": {"mae": float(h)},
                    "feature_importance": {"cci": 0.7, "whi": 0.3}} for h in models],
        **extra,
    }, directory / "metadata.json")
    return directory


def constant_regressor(value):
    return DummyRegressor(strategy="constant", constant=value).fit(ROW, [value])


# ----------------------------------------------------------------------------------------------
# Comfort Climate Index
# ----------------------------------------------------------------------------------------------

def test_cci_predicts_the_three_following_days_clipped_to_0_100(tmp_path):
    directory = write_bundle(tmp_path / "cci", "cci_test", {
        1: constant_regressor(150.0), 2: constant_regressor(61.4), 3: constant_regressor(-5.0)})
    predictor = ComfortClimatePredictor.from_directory(directory, FakeFeatures())

    assert predictor.predict(date(2025, 1, 1)) == {
        "2025-01-02": 100, "2025-01-03": 61, "2025-01-04": 0}


def test_cci_rejects_a_bundle_without_all_three_horizons(tmp_path):
    directory = write_bundle(tmp_path / "cci", "cci_test", {1: constant_regressor(50.0)})

    with pytest.raises(ValueError, match="horizons"):
        ComfortClimatePredictor.from_directory(directory, FakeFeatures())


def test_cci_metadata_reports_mean_and_per_horizon_metrics(tmp_path):
    directory = write_bundle(tmp_path / "cci", "cci_test",
                             {h: constant_regressor(50.0) for h in (1, 2, 3)})
    described = ComfortClimatePredictor.from_directory(directory, FakeFeatures()).describe()

    assert described["target"] == "comfort_climate"
    assert described["features"] == COLUMNS
    assert described["performance_metrics"]["mae"] == pytest.approx(2.0)
    assert described["performance_metrics"]["mae_h3"] == pytest.approx(3.0)
    assert list(described["feature_importance"]) == ["cci", "whi"]


# ----------------------------------------------------------------------------------------------
# Weather Hazard Category
# ----------------------------------------------------------------------------------------------

def test_whc_predicts_the_category_seven_days_ahead(tmp_path):
    classifier = DummyClassifier(strategy="constant", constant=2).fit(
        pd.concat([ROW] * 3), [0, 1, 2])
    directory = write_bundle(
        tmp_path / "whc", "whc_test", {7: classifier}, prediction_type="classification",
        class_labels={"0": "Low Risk", "1": "Moderate Risk", "2": "High Risk",
                      "3": "Extreme Risk"})
    predictor = WeatherHazardPredictor.from_directory(directory, FakeFeatures())

    assert predictor.predict(date(2025, 1, 1)) == {"2025-01-08": "High Risk"}


# ----------------------------------------------------------------------------------------------
# Bundle loading
# ----------------------------------------------------------------------------------------------

def test_bundle_rejects_models_trained_on_different_features(tmp_path):
    directory = write_bundle(tmp_path / "cci", "cci_test", {1: constant_regressor(50.0)})

    with pytest.raises(ValueError, match="do not match"):
        ModelBundle.load(directory, ["another", "feature", "set"])


def test_bundle_without_models_fails_clearly(tmp_path):
    (tmp_path / "empty").mkdir()
    wx.save_file({"feature_columns": COLUMNS}, tmp_path / "empty" / "metadata.json")

    with pytest.raises(FileNotFoundError, match="No '<name>_h<horizon>.joblib'"):
        ModelBundle.load(tmp_path / "empty", COLUMNS)


# ----------------------------------------------------------------------------------------------
# Feature service (Open-Meteo replaced by fakes)
# ----------------------------------------------------------------------------------------------

def test_bundle_reads_each_models_own_feature_list(tmp_path):
    directory = write_bundle(tmp_path / "cci", "cci_test", {1: constant_regressor(50.0)},
                             manifest={"model_features": {"1": ["cci", "whi"]}})

    bundle = ModelBundle.load(directory, COLUMNS)

    assert bundle.model_features == {1: ["cci", "whi"]}


def test_bundle_rejects_a_model_needing_features_the_pipeline_lacks(tmp_path):
    directory = write_bundle(tmp_path / "cci", "cci_test", {1: constant_regressor(50.0)},
                             manifest={"model_features": {"1": ["cci", "unknown"]}})

    with pytest.raises(ValueError, match="does not produce"):
        ModelBundle.load(directory, COLUMNS)


class ColumnRecorder:
    """A fake model that remembers which columns it was given."""

    def predict(self, X):
        self.seen = list(X.columns)
        return np.array([42.0])


def test_each_model_receives_only_its_own_columns():
    recorder = ColumnRecorder()
    bundle = ModelBundle(directory=Path("."), models={1: recorder},
                         model_features={1: ["whi"]}, metadata={})

    assert bundle.predict(1, ROW)[0] == 42.0
    assert recorder.seen == ["whi"]


def test_drift_status_waits_for_enough_rows_then_flags_drift():
    rng = np.random.default_rng(0)
    training = pd.DataFrame({"cci": rng.normal(60, 10, 3000)})
    bundle = ModelBundle(directory=Path("."), models={1: object()},
                         model_features={1: ["cci"]}, metadata={"experiment": "cci_test"},
                         drift_reference=wx.feature_reference(training))

    few = bundle.drift_status("comfort_climate", pd.DataFrame({"cci": [60.0] * 5}))
    normal = bundle.drift_status("comfort_climate",
                                 pd.DataFrame({"cci": rng.normal(60, 10, 60)}))
    shifted = bundle.drift_status("comfort_climate",
                                  pd.DataFrame({"cci": rng.normal(120, 5, 60)}))

    assert few["status"].startswith("insufficient data")
    assert normal["status"] == "ok"
    assert shifted["status"] == "drift detected"
    assert shifted["drifted_features"] == ["cci"]


def test_feature_service_maps_download_failures_to_weather_data_error(monkeypatch):
    def unreachable(pipeline, day):
        raise wx.IngestionError("offline")

    monkeypatch.setattr(wx, "fetch_for_serving", unreachable)

    with pytest.raises(WeatherDataError, match="could not be retrieved"):
        FeatureService(pipeline=object()).features_for(date(2025, 1, 1))


def test_feature_service_maps_incomplete_history_to_weather_data_error(monkeypatch):
    def incomplete(pipeline, hourly, day):
        raise ValueError("Features are incomplete.")

    monkeypatch.setattr(wx, "fetch_for_serving", lambda pipeline, day: pd.DataFrame())
    monkeypatch.setattr(wx, "features_for_date", incomplete)

    with pytest.raises(WeatherDataError, match="Complete Open-Meteo observations"):
        FeatureService(pipeline=object()).features_for(date(2025, 1, 1))


def test_feature_service_downloads_each_date_only_once(monkeypatch):
    calls = []
    monkeypatch.setattr(wx, "fetch_for_serving",
                        lambda pipeline, day: calls.append(day) or pd.DataFrame())
    monkeypatch.setattr(wx, "features_for_date", lambda pipeline, hourly, day: ROW)
    service = FeatureService(pipeline=object())

    service.features_for(date(2025, 1, 1))
    service.features_for(date(2025, 1, 1))
    service.features_for(date(2025, 1, 2))

    assert calls == [date(2025, 1, 1), date(2025, 1, 2)]
