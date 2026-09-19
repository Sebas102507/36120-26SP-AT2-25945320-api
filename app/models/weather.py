"""Response schemas. Field names follow the example responses in the assessment brief."""

from datetime import date
from typing import Dict, List, Literal, Union

from pydantic import BaseModel, Field


class PredictionTarget(BaseModel):
    target: str
    endpoint: str
    prediction_type: str
    forecast_horizon: str


class ProjectOverview(BaseModel):
    project: str
    description: str
    version: str
    student: str
    github: str
    endpoints: List[str]
    prediction_targets: List[PredictionTarget]


class HealthCheckResponse(BaseModel):
    status: Literal["healthy"]
    message: str


class ForecastResponse(BaseModel):
    input_date: date
    predictions: Dict[str, Dict[str, Union[int, str]]]


class ModelMetadata(BaseModel):
    target: str
    experiment: str
    prediction_type: str
    algorithm: str
    forecast_horizon: str
    features: List[str] = Field(description="Feature columns the deployed model(s) receive.")
    feature_importance: Dict[str, float] = Field(
        description="Mean feature importance across horizons (empty for dummy baselines)."
    )
    performance_metrics: Dict[str, float] = Field(
        description="Validation metrics; multi-horizon models also report each horizon."
    )


class DriftReport(BaseModel):
    target: str
    experiment: str
    status: str = Field(description="ok, drift detected, insufficient data or unavailable.")
    rows_observed: int = Field(description="Distinct dates served since the API started.")
    features_checked: int
    drifted_features: List[str] = Field(
        description="Features whose recent values often fall outside the training range."
    )
