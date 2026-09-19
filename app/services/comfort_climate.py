"""Climate Comfort Index (CCI) forecasts for the next three days."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from services.features import FeatureService, ModelBundle, WeatherDataError

__all__ = ["ComfortClimatePredictor", "WeatherDataError"]


class ComfortClimatePredictor:
    """Predicts CCI for day + 1, + 2 and + 3 with one model per horizon."""

    target = "comfort_climate"
    horizons = [1, 2, 3]

    def __init__(self, features: FeatureService, bundle: ModelBundle) -> None:
        if bundle.horizons != self.horizons:
            raise ValueError(f"CCI bundle must contain horizons {self.horizons}, "
                             f"found {bundle.horizons}.")
        self.features = features
        self.bundle = bundle

    @classmethod
    def from_directory(cls, directory: Path, features: FeatureService) -> ComfortClimatePredictor:
        return cls(features, ModelBundle.load(directory, features.feature_columns))

    def predict(self, day: date) -> dict[str, int]:
        """Return ``{"YYYY-MM-DD": CCI}`` for the three days after ``day``."""
        row = self.features.features_for(day)
        return {
            (day + timedelta(days=horizon)).isoformat():
                int(round(float(np.clip(self.bundle.predict(horizon, row)[0], 0, 100))))
            for horizon in self.horizons
        }

    def describe(self) -> dict:
        return self.bundle.describe(self.target)

    def drift(self) -> dict:
        return self.bundle.drift_status(self.target, self.features.recent_rows())
