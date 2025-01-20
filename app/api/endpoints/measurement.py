# app/api/endpoints/measurement.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
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
        # Process image and get results including visualization
        result = await image_processor.process_image(file_path)
        
        # Create measurement objects for each detected object
        measurements = []
        for detection in result["measurements"]:
            measurement = ObjectMeasurement(
                object_type=detection["object_type"],
                dimensions=detection["dimensions"],
                confidence_score=detection["confidence_score"],
                capture_angle="front",
                lighting_condition="normal",
                reference_object="none"
            )
            measurements.append(measurement)
        
        # Create response with visualization
        response = MeasurementResponse(
            id=1,  # You'd get this from database
            created_at=datetime.now(),
            measurements=measurements,
            visualization_image=result["visualization"],
            image_dimensions=result["image_dimensions"]
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