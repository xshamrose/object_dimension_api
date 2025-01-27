# app/api/endpoints/arcore.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.arcore_processor import ARCoreProcessor
from app.schemas.dimension import ObjectMeasurement, MeasurementResponse
from datetime import datetime
import os
from pathlib import Path
from app.core.config import settings

router = APIRouter()
arcore_processor = ARCoreProcessor()

@router.post("/measure/", response_model=MeasurementResponse)
async def measure_objects_with_arcore(
    file: UploadFile = File(...)
):
    # Create upload directory if it doesn't exist
    upload_dir = Path(settings.UPLOAD_FOLDER)
    upload_dir.mkdir(exist_ok=True)

    # Save uploaded file
    file_path = upload_dir / file.filename
    with file_path.open("wb") as buffer:
        buffer.write(await file.read())

    try:
        # Process image and get results including visualization
        result = await arcore_processor.process_image(file_path)

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