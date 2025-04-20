from enum import Enum
from datetime import datetime
from .base import Model, PyObjectId
from pydantic import Field
from typing import Dict

class DetectionType(str, Enum):
    FAKE_NEWS = "fake_news"
    DEEPFAKE = "deepfake"
    IMAGE_MANIPULATION = "image_manipulation"

class Result(Model):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    content_id: str
    detection_type: DetectionType
    is_fake: bool
    confidence: float
    explanation: str
    model_used: str
    model_version: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)