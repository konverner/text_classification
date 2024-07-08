from typing import List

from pydantic import BaseModel


class ClassificationResult(BaseModel):
    """Classification result schema."""

    text: str
    label: str
    score: float


class PredictionRequest(BaseModel):
    """Prediction request schema."""

    user_id: str
    texts: List[str]


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    outputs: List[ClassificationResult]
