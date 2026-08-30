from datetime import date
from typing import Dict, List, Literal, Union

from pydantic import BaseModel, Field


class ProjectOverview(BaseModel):
    title: str
    description: str
    version: str
    github_url: str
    endpoints: List[str]
    prediction_targets: List[Dict[str, str]]


class HealthCheckResponse(BaseModel):
    status: Literal["healthy"]
    message: str


class ForecastResponse(BaseModel):
    input_date: date
    predictions: Dict[str, Dict[str, Union[int, str]]]


class ModelMetadata(BaseModel):
    target: str
    prediction_type: str
    algorithm: str
    forecast_horizon: str
    feature_list: List[str]
    feature_importances: Dict[str, float] = Field(
        description="Relative feature importances for the model."
    )
    performance_metrics: Dict[str, float]
