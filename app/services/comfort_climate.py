"""Manifest-driven Climate Comfort Index production inference."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import torch

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
SYDNEY = {"latitude": -33.8688, "longitude": 151.2093, "timezone": "Australia/Sydney"}
HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "wind_gusts_10m",
)


class CciInferenceError(RuntimeError):
    """Base exception for failures while making a CCI forecast."""


class WeatherDataError(CciInferenceError):
    """Raised when Open-Meteo cannot provide complete source observations."""


@dataclass
class ComfortClimatePredictor:
    """Model adapter and preprocessing contract described by a manifest."""

    model: object
    adapter: str
    input_kind: str
    input_scaler: object | None
    target_scaler: object | None
    feature_columns: tuple[str, ...]
    history_days: int
    target_inversion: dict[str, str]
    timeout: int = 30

    @classmethod
    def from_model_directory(cls, model_dir: Path) -> "ComfortClimatePredictor":
        manifest = load_model_manifest(model_dir)
        logger.info(
            "Loading CCI model: experiment=%s adapter=%s input_kind=%s",
            manifest.get("experiment", "unknown"),
            manifest.get("adapter", "unknown"),
            manifest.get("input_kind", "unknown"),
        )
        required_keys = {"adapter", "input_kind", "lookback_days"}
        missing_keys = required_keys.difference(manifest)
        if missing_keys:
            raise ValueError(f"CCI manifest is missing keys: {sorted(missing_keys)}")

        model_files = manifest.get("model_files") or [manifest.get("model_file")]
        if not all(isinstance(name, str) for name in model_files):
            raise ValueError("CCI manifest must contain model_file or model_files.")
        model_paths = tuple(model_dir / name for name in model_files)
        optional_files = (
            manifest.get("input_scaler_file"),
            manifest.get("target_scaler_file"),
        )
        required = (*model_paths, *(model_dir / name for name in optional_files if name))
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing CCI production artefacts: {', '.join(missing)}")

        adapter = str(manifest["adapter"])
        if adapter == "torchscript":
            if len(model_paths) != 1:
                raise ValueError("A TorchScript CCI deployment requires exactly one model file.")
            model = torch.jit.load(model_paths[0], map_location="cpu")
            model.eval()
        elif adapter == "joblib":
            model = tuple(joblib.load(path) for path in model_paths)
        else:
            raise ValueError(f"Unsupported CCI model adapter: {adapter}")

        input_kind = str(manifest["input_kind"])
        feature_key = "sequence_features" if input_kind == "sequence" else "feature_columns"
        feature_columns = manifest.get(feature_key)
        if not isinstance(feature_columns, list) or not feature_columns:
            raise ValueError(f"CCI manifest must contain non-empty {feature_key}.")
        return cls(
            model=model,
            adapter=adapter,
            input_kind=input_kind,
            input_scaler=(
                joblib.load(model_dir / manifest["input_scaler_file"])
                if manifest.get("input_scaler_file")
                else None
            ),
            target_scaler=(
                joblib.load(model_dir / manifest["target_scaler_file"])
                if manifest.get("target_scaler_file")
                else None
            ),
            feature_columns=tuple(feature_columns),
            history_days=int(manifest.get("history_days", manifest["lookback_days"])),
            target_inversion=dict(manifest.get("target_inversion", {"type": "none"})),
        )

    def predict(self, feature_date: date) -> dict[str, int]:
        logger.info("Starting CCI prediction for base date %s", feature_date)
        daily = self._fetch_daily_observations(feature_date)
        model_input = self._build_model_input(daily)
        predictions = self._run_model(model_input)
        predictions = self._invert_predictions(predictions, daily)
        if len(predictions) != 3:
            raise CciInferenceError(
                f"CCI model returned {len(predictions)} forecasts; exactly three are required."
            )

        result = {
            (feature_date + timedelta(days=horizon)).isoformat(): int(round(value))
            for horizon, value in enumerate(predictions, start=1)
        }
        logger.info("Completed CCI prediction for %s: %s", feature_date, result)
        return result

    def _fetch_daily_observations(self, feature_date: date) -> pd.DataFrame:
        start_date = feature_date - timedelta(days=self.history_days - 1)
        logger.info(
            "Requesting Open-Meteo hourly data from %s through %s (%d daily observations)",
            start_date,
            feature_date,
            self.history_days,
        )
        params = {
            **SYDNEY,
            "start_date": start_date.isoformat(),
            "end_date": feature_date.isoformat(),
            "hourly": ",".join(HOURLY_VARIABLES),
        }
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            hourly = pd.DataFrame(payload["hourly"])
        except (KeyError, ValueError, requests.RequestException) as error:
            raise WeatherDataError("Open-Meteo observations could not be retrieved.") from error

        required = {"time", *HOURLY_VARIABLES}
        missing = required.difference(hourly.columns)
        if missing:
            raise WeatherDataError(f"Open-Meteo response is missing: {sorted(missing)}")

        hourly["time"] = pd.to_datetime(hourly["time"])
        hourly = hourly.set_index("time").sort_index()
        logger.info("Received %d Open-Meteo hourly rows", len(hourly))
        aggregation = {
            "temperature_2m": "mean",
            "relative_humidity_2m": "mean",
            "wind_speed_10m": "mean",
            "cloud_cover": "mean",
            "precipitation": "sum",
            "wind_gusts_10m": "max",
        }
        daily = hourly.resample("D").agg(aggregation)
        hourly_count = hourly.resample("D").size()
        expected_index = pd.date_range(start_date, feature_date, freq="D")
        daily = daily.reindex(expected_index)
        complete_days = hourly_count.reindex(expected_index).eq(24)
        if not complete_days.all() or daily.isna().any(axis=None):
            raise WeatherDataError(
                f"Complete hourly observations are required for each of the preceding "
                f"{self.history_days} days."
            )
        return self._add_cci(daily)

    def _build_model_input(self, daily: pd.DataFrame) -> np.ndarray:
        if len(daily) != self.history_days:
            raise WeatherDataError(
                f"Expected {self.history_days} daily observations, received {len(daily)}."
            )
        day_of_year = daily.index.dayofyear.to_numpy()
        angle = 2 * np.pi * day_of_year / 365.25
        enriched = daily.copy()
        enriched["doy_sin"] = np.sin(angle)
        enriched["doy_cos"] = np.cos(angle)
        if self.input_kind == "sequence":
            model_input = enriched.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
            logger.info("Built CCI sequence input with shape %s", model_input.shape)
            return model_input
        if self.input_kind == "tabular":
            model_input = self._build_tabular_features(enriched).loc[:, self.feature_columns].to_numpy(
                dtype=np.float32
            )
            logger.info("Built CCI tabular input with shape %s", model_input.shape)
            return model_input
        raise ValueError(f"Unsupported CCI input kind: {self.input_kind}")

    def _run_model(self, model_input: np.ndarray) -> np.ndarray:
        if self.input_scaler is not None:
            if self.input_kind == "sequence":
                model_input = self.input_scaler.transform(model_input).reshape(
                    1, self.history_days, -1
                )
            else:
                model_input = self.input_scaler.transform(model_input)
        elif self.input_kind == "sequence":
            model_input = model_input.reshape(1, self.history_days, -1)

        if self.adapter == "torchscript":
            with torch.inference_mode():
                return self.model(torch.tensor(model_input, dtype=torch.float32)).numpy().reshape(-1)
        if isinstance(self.model, tuple):
            if len(self.model) == 1:
                return np.asarray(self.model[0].predict(model_input)).reshape(-1)
            return np.asarray([model.predict(model_input)[0] for model in self.model])
        return np.asarray(self.model.predict(model_input)).reshape(-1)

    def _invert_predictions(self, predictions: np.ndarray, daily: pd.DataFrame) -> np.ndarray:
        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions.reshape(1, -1))[0]
        if self.target_inversion.get("type") == "residual_plus_feature":
            feature = self.target_inversion["feature"]
            predictions = predictions + float(daily.iloc[-1][feature])
        elif self.target_inversion.get("type") != "none":
            raise ValueError("Unsupported CCI target inversion.")
        return np.clip(predictions, 0, 100)

    @staticmethod
    def _build_tabular_features(daily: pd.DataFrame) -> pd.DataFrame:
        """Reproduce the 86 CCI features used by tabular sklearn/XGBoost models."""
        result = daily.copy()
        lag_columns = (
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "precipitation",
            "wind_gusts_10m",
            "cci",
        )
        for column in lag_columns:
            for lag in (1, 2, 3, 7):
                result[f"{column}_lag{lag}"] = result[column].shift(lag)

        for column in ("temperature_2m", "relative_humidity_2m", "cloud_cover", "cci"):
            for window in (3, 7, 14, 30):
                rolled = result[column].rolling(window=window, min_periods=window)
                result[f"{column}_roll{window}_mean"] = rolled.mean()
                result[f"{column}_roll{window}_std"] = rolled.std()

        for window in (3, 7, 14, 30):
            result[f"precipitation_roll{window}_sum"] = (
                result["precipitation"].rolling(window=window, min_periods=window).sum()
            )
            result[f"wind_gusts_10m_roll{window}_max"] = (
                result["wind_gusts_10m"].rolling(window=window, min_periods=window).max()
            )

        result["temperature_2m_delta1"] = result["temperature_2m"].diff(1)
        result["temperature_2m_delta3"] = result["temperature_2m"].diff(3)
        result["cci_delta1"] = result["cci"].diff(1)
        result["cci_delta3"] = result["cci"].diff(3)
        if result.iloc[-1].isna().any():
            raise WeatherDataError("Insufficient observations for the configured CCI feature window.")
        return result.iloc[[-1]]

    @staticmethod
    def _add_cci(daily: pd.DataFrame) -> pd.DataFrame:
        result = daily.copy()
        result["temp_score"] = (1 - (result["temperature_2m"] - 22).abs() / 20).clip(lower=0)
        result["humidity_score"] = (
            1 - (result["relative_humidity_2m"] - 50).abs() / 50
        ).clip(lower=0)
        result["wind_score"] = (1 - (result["wind_speed_10m"] - 10).abs() / 40).clip(lower=0)
        result["cloud_score"] = (1 - result["cloud_cover"] / 100).clip(lower=0)
        result["rain_score"] = (1 - result["precipitation"] / 20).clip(lower=0)
        result["cci"] = 100 * (
            0.35 * result["temp_score"]
            + 0.20 * result["humidity_score"]
            + 0.15 * result["wind_score"]
            + 0.15 * result["cloud_score"]
            + 0.15 * result["rain_score"]
        )
        return result


def load_model_manifest(model_dir: Path) -> dict[str, object]:
    """Load the model-independent CCI deployment contract."""
    metadata_path = model_dir / "cci_model_manifest.json"
    with metadata_path.open() as file:
        return json.load(file)


def load_experiment_metadata(model_dir: Path, manifest: dict[str, object]) -> dict[str, object]:
    """Load the experiment report named by the deployment manifest."""
    metadata_file = manifest.get("metadata_file")
    if not isinstance(metadata_file, str):
        raise ValueError("CCI manifest must declare metadata_file.")
    metadata_path = model_dir / metadata_file
    with metadata_path.open() as file:
        return json.load(file)
