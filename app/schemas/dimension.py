# 3. app/schemas/dimension.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DimensionBase(BaseModel):
    width: float
    height: float
    depth: Optional[float]
    unit: str = "mm"

class ObjectMeasurement(BaseModel):
    object_type: str
    dimensions: DimensionBase
    confidence_score: float
    capture_angle: str
    lighting_condition: str
    reference_object: str

class MeasurementResponse(ObjectMeasurement):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True