"""Weather Hazard Category inference from the promoted joblib model."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np

from services.comfort_climate import ComfortClimatePredictor


class WeatherHazardPredictor:
    """Loads the promoted WHC classifier and shares daily feature engineering."""

    def __init__(self, model: object, feature_builder: ComfortClimatePredictor, labels: dict[str, str]):
        self.model = model
        self.feature_builder = feature_builder
        self.labels = labels

    @classmethod
    def from_model_directory(cls, model_dir: Path) -> "WeatherHazardPredictor":
        manifest = json.loads((model_dir / "whc_model_manifest.json").read_text())
        model_file = manifest["model_files"]["7"]
        metadata = json.loads((model_dir / "whc_model_metadata.json").read_text())
        feature_builder = ComfortClimatePredictor(
            model=None,
            adapter="joblib",
            input_kind="tabular",
            input_scaler=None,
            target_scaler=None,
            feature_columns=tuple(manifest["feature_columns"]),
            history_days=int(manifest["history_days"]),
            target_inversion={"type": "none"},
        )
        return cls(joblib.load(model_dir / model_file), feature_builder, metadata["class_labels"])

    def predict(self, feature_date: date) -> dict[str, str]:
        daily = self.feature_builder._fetch_daily_observations(feature_date)
        features = self.feature_builder._build_model_input(daily)
        prediction = int(np.asarray(self.model.predict(features)).reshape(-1)[0])
        return {
            (feature_date + timedelta(days=7)).isoformat(): self.labels[str(prediction)]
        }
