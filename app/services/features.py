"""Shared serving components: the fitted preprocessing pipeline and the model bundles.

Feature engineering is **not** implemented in this repository. The API loads the preprocessing
pipeline that was fitted in the experiments repository (``nucleus_ml.weather``) and calls it
unchanged, so every served feature row is built by exactly the code the models were trained on:

    hourly  = wx.fetch_for_serving(pipeline, date)         # Open-Meteo, UTC -> Sydney time
    row     = wx.features_for_date(pipeline, hourly, date)  # daily aggregation + all features
    predict = model.predict(row[model_features])            # each model uses its own columns

Each promoted model may use a subset of the features (refinement can switch feature families
off), so every bundle records the columns of each model and the API selects them per model.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from nucleus_ml import weather as wx

logger = logging.getLogger(__name__)

# Drift is only judged once enough distinct dates have been served.
MIN_ROWS_FOR_DRIFT = 30


class WeatherDataError(RuntimeError):
    """Raised when Open-Meteo cannot supply the history a prediction needs."""


class FeatureService:
    """Builds the single feature row for a date with the fitted ``nucleus_ml`` pipeline.

    It also remembers the most recently built rows (one per distinct date), which the drift
    monitor compares with the training distribution.
    """

    def __init__(self, pipeline: wx.WeatherPreprocessor, cache_size: int = 256,
                 history_size: int = 365) -> None:
        self.pipeline = pipeline
        self.served: deque[pd.DataFrame] = deque(maxlen=history_size)
        # Historical observations never change, so a row built once can be reused: both
        # endpoints asking about the same date then cost a single Open-Meteo request.
        # Failed lookups raise, and exceptions are never cached.
        self.features_for = lru_cache(maxsize=cache_size)(self._build_features)

    @classmethod
    def from_file(cls, path: Path) -> FeatureService:
        pipeline = wx.load_file(path)
        if not isinstance(pipeline, wx.WeatherPreprocessor):
            raise TypeError(f"{path} does not contain a nucleus_ml WeatherPreprocessor.")
        logger.info("Loaded preprocessing pipeline: %d features, %d days of history",
                    len(pipeline.feature_names_out_), pipeline.warmup_days)
        return cls(pipeline)

    @property
    def feature_columns(self) -> list[str]:
        return self.pipeline.feature_names_out_

    def recent_rows(self) -> pd.DataFrame:
        """The feature rows served recently (one per distinct date), for drift monitoring."""
        return pd.concat(list(self.served)) if self.served else pd.DataFrame()

    def _build_features(self, day: date) -> pd.DataFrame:
        """Fetch the required history for ``day`` and return its feature row."""
        try:
            hourly = wx.fetch_for_serving(self.pipeline, day)
        except (wx.IngestionError, ValueError) as error:
            raise WeatherDataError("Open-Meteo observations could not be retrieved.") from error
        try:
            row = wx.features_for_date(self.pipeline, hourly, day)
        except (KeyError, ValueError) as error:
            raise WeatherDataError(
                f"Complete Open-Meteo observations are not available for {day.isoformat()} "
                "and the 30 days before it. The archive lags real time by about five days."
            ) from error
        self.served.append(row)
        return row


@dataclass
class ModelBundle:
    """One promoted model folder, as written by ``wx.promote_champion``.

    Holds ``<experiment>_h<horizon>.joblib`` models, the feature columns each model uses, the
    ``/model-metadata`` document and, when present, the training drift reference.
    """

    directory: Path
    models: dict[int, object]
    model_features: dict[int, list[str]]
    metadata: dict
    drift_reference: dict | None = field(default=None)

    @classmethod
    def load(cls, directory: Path, feature_columns: list[str]) -> ModelBundle:
        """Load a bundle and check every model's features are produced by the pipeline."""
        directory = Path(directory)
        metadata = wx.load_file(directory / "metadata.json")
        manifest_path = directory / "manifest.json"
        manifest = wx.load_file(manifest_path) if manifest_path.is_file() else {}
        models = {wx.horizon_from_name(path.stem): wx.load_file(path)
                  for path in sorted(directory.glob("*.joblib"))}
        models.pop(None, None)
        if not models:
            raise FileNotFoundError(f"No '<name>_h<horizon>.joblib' model found in {directory}.")

        # Per-model feature lists (refinement may drop families); older bundles use one list.
        per_model = {int(h): list(c) for h, c in manifest.get("model_features", {}).items()}
        model_features = {h: per_model.get(h, list(metadata["feature_columns"])) for h in models}
        available = set(feature_columns)
        for horizon, columns in model_features.items():
            missing = [c for c in columns if c not in available]
            if missing:
                raise ValueError(f"{directory.name}: the h{horizon} model needs features the "
                                 f"preprocessing pipeline does not produce: {missing[:5]}. Copy "
                                 "the pipeline and the models from the same experiments run.")

        reference_path = directory / "drift_reference.json"
        reference = wx.load_file(reference_path) if reference_path.is_file() else None
        logger.info("Loaded %s (%s) for horizons %s", directory.name,
                    metadata.get("algorithm"), sorted(models))
        return cls(directory, models, model_features, metadata, reference)

    @property
    def horizons(self) -> list[int]:
        return sorted(self.models)

    def predict(self, horizon: int, row: pd.DataFrame) -> np.ndarray:
        """Predict with the model for ``horizon``, giving it exactly its own feature columns."""
        return np.asarray(self.models[horizon].predict(row[self.model_features[horizon]]))

    def drift_status(self, target: str, recent: pd.DataFrame) -> dict:
        """Compare recently served feature rows with the training distribution (stage 8)."""
        status = {"target": target,
                  "experiment": self.metadata.get("experiment", self.directory.name),
                  "rows_observed": int(len(recent)), "features_checked": 0,
                  "drifted_features": []}
        if self.drift_reference is None:
            return {**status, "status": "unavailable: the bundle has no drift reference"}
        if len(recent) < MIN_ROWS_FOR_DRIFT:
            return {**status, "status": f"insufficient data: at least {MIN_ROWS_FOR_DRIFT} "
                                        "distinct dates must be served first"}
        report = wx.drift_report(self.drift_reference, recent)
        drifted = report.index[report["drifted"].astype(bool)].tolist()
        return {**status, "features_checked": int(len(report)), "drifted_features": drifted,
                "status": "drift detected" if drifted else "ok"}

    def describe(self, target: str) -> dict:
        """Summarise the bundle in the shape of the ``/model-metadata`` response."""
        entries = self.metadata.get("models", [])
        metrics: dict[str, list[float]] = {}
        importance: dict[str, list[float]] = {}
        for entry in entries:
            for name, value in entry.get("validation_metrics", {}).items():
                if isinstance(value, (int, float)):
                    metrics.setdefault(name, []).append(float(value))
            for name, value in entry.get("feature_importance", {}).items():
                importance.setdefault(name, []).append(float(value))

        # Multi-horizon targets report the mean over horizons plus each horizon separately.
        performance = {name: float(np.mean(values)) for name, values in metrics.items()}
        if len(entries) > 1:
            for entry in entries:
                horizon = wx.horizon_from_name(entry.get("target_column", ""))
                for name, value in entry.get("validation_metrics", {}).items():
                    if isinstance(value, (int, float)):
                        performance[f"{name}_h{horizon}"] = float(value)

        return {
            "target": target,
            "experiment": self.metadata.get("experiment", self.directory.name),
            "prediction_type": self.metadata.get("prediction_type", "unknown"),
            "algorithm": self.metadata.get("algorithm", "unknown"),
            "forecast_horizon": f"{', '.join(map(str, self.horizons))} "
                                f"Day{'s' if self.horizons != [1] else ''}",
            "feature_list": list(self.metadata["feature_columns"]),
            "feature_importances": dict(sorted(
                ((name, float(np.mean(v))) for name, v in importance.items()),
                key=lambda item: item[1], reverse=True)),
            "performance_metrics": performance,
        }
