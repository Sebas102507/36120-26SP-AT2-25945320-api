"""Weather Hazard Category (WHC) forecasts exactly seven days ahead."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from services.features import FeatureService, ModelBundle


class WeatherHazardPredictor:
    """Predicts the hazard category for day + 7."""

    target = "weather_hazard"
    horizon = 7

    def __init__(self, features: FeatureService, bundle: ModelBundle) -> None:
        if bundle.horizons != [self.horizon]:
            raise ValueError(f"WHC bundle must contain horizon {self.horizon}, "
                             f"found {bundle.horizons}.")
        self.features = features
        self.bundle = bundle
        self.labels = {int(code): name for code, name in bundle.metadata["class_labels"].items()}

    @classmethod
    def from_directory(cls, directory: Path, features: FeatureService) -> WeatherHazardPredictor:
        return cls(features, ModelBundle.load(directory, features.feature_columns))

    def predict(self, day: date) -> dict[str, str]:
        """Return ``{"YYYY-MM-DD": "<category>"}`` for the day seven days after ``day``."""
        row = self.features.features_for(day)
        category = int(self.bundle.predict(self.horizon, row).ravel()[0])
        return {(day + timedelta(days=self.horizon)).isoformat(): self.labels[category]}

    def describe(self) -> dict:
        return self.bundle.describe(self.target)

    def drift(self) -> dict:
        return self.bundle.drift_status(self.target, self.features.recent_rows())
