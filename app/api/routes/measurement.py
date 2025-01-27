from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..dependencies import get_settings

from ...services.enhanced_processor import EnhancedImageProcessor

router = APIRouter(prefix="/api/v1/measurements", tags=["measurements"])
processor = EnhancedImageProcessor()

@router.post("/")
async def calculate_dimensions(
    image: UploadFile = File(...),
    object_type: str = "unknown"
):
    """Calculate dimensions from uploaded image
    
    Args:
        image: Image file to process
        object_type: Type of object in image (e.g. 'phone', 'credit_card', 'a4_paper')
                    This helps improve accuracy when object dimensions are known
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    settings = get_settings()
    
    # Save uploaded file
    temp_path = Path(settings.UPLOAD_DIR) / image.filename
    try:
        with open(temp_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
            
        # Process image with enhanced processor
        results = await processor.process_image(temp_path)
        
        return {
            "success": True,
            "data": results,
            "message": "Dimensions calculated successfully"
        }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()