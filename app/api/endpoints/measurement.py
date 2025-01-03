#app/api/endpoints/measurement.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import shutil
from pathlib import Path
from app.services.image_processor import ImageProcessor
from app.schemas.dimension import ObjectMeasurement, MeasurementResponse
from app.core.config import settings
import os
from datetime import datetime

router = APIRouter()
image_processor = ImageProcessor()

@router.post("/measure/", response_model=MeasurementResponse)
async def measure_object(
    file: UploadFile = File(...),
    object_type: str = "unknown"
):
    # Create upload directory if it doesn't exist
    upload_dir = Path(settings.UPLOAD_FOLDER)
    upload_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    file_path = upload_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process image - only pass the file_path as per ImageProcessor implementation
        result = await image_processor.process_image(file_path)
        
        # Create response
        measurement = ObjectMeasurement(
            object_type=object_type,
            dimensions=result["measurements"][0]["dimensions"],  # Get first detected object's dimensions
            confidence_score=result["measurements"][0]["confidence_score"],
            capture_angle="front",
            lighting_condition="normal",
            reference_object="none"  # Since we're not using reference objects in current implementation
        )
        
        # Create response with additional fields
        response = MeasurementResponse(
            id=1,  # You'd get this from database
            created_at=datetime.now(),
            **measurement.dict()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
        
    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)