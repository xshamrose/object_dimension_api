import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List, Any

from .depth_calculator import DepthCalculator

class EnhancedImageProcessor:
    """Enhanced image processor with improved dimension calculation"""
    
    # Base depth constant (in mm) - can be calibrated based on typical usage
    BASE_DEPTH = 1000  # 1 meter default base depth
    
    def __init__(self):
        """Initialize the processor with default settings"""
        self.depth_calculator = DepthCalculator()
    
    async def process_image(self, image_path: Path) -> Dict[str, Any]:
        """Process an image and return measurements
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing measurements and detected objects
        """
        # Read and preprocess the image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        preprocessed = self._preprocess_image(image)
        
        # Get camera matrix (this would normally come from camera calibration)
        # For now using a default matrix based on typical smartphone cameras
        height, width = image.shape[:2]
        focal_length = max(height, width)  # Approximation for typical smartphone
        camera_matrix = np.array([
            [focal_length, 0, width/2],
            [0, focal_length, height/2],
            [0, 0, 1]
        ])
        
        # Detect objects in the image
        objects = await self._detect_objects(preprocessed)
        
        # Process each detected object
        results = []
        for obj in objects:
            bbox = obj["bbox"]
            object_type = obj.get("type", "unknown")
            
            # Calculate dimensions using known object types when available
            dimensions = self._calculate_real_dimensions(
                preprocessed,
                bbox,
                camera_matrix,
                object_type
            )
            
            results.append({
                **obj,
                "dimensions": dimensions
            })
            
            # Draw measurements on the image
            self._draw_measurements(image, bbox, dimensions)
        
        # Save annotated image
        output_path = image_path.parent / f"annotated_{image_path.name}"
        cv2.imwrite(str(output_path), image)
        
        return {
            "objects": results,
            "annotated_image": str(output_path)
        }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better object detection"""
        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add any additional preprocessing steps here
        
        return normalized
    
    async def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in the image
        
        This would typically use a deep learning model or computer vision API
        For now returning a mock detection
        """
        height, width = image.shape[:2]
        return [{
            "type": "phone",
            "confidence": 0.95,
            "bbox": (width//4, height//4, 3*width//4, 3*height//4)
        }]
    
    

    def _calculate_real_dimensions(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        camera_matrix: np.ndarray,
        object_type: str = "unknown"
    ) -> Dict[str, float]:
        """Calculate real-world dimensions of detected object
        
        Uses DepthCalculator for more accurate measurements based on known object
        dimensions when available.
        """
        # Try to get depth using known object dimensions
        depth = self.depth_calculator.estimate_depth_from_known_object(
            bbox,
            image.shape,
            camera_matrix,
            object_type
        )
        
        if depth is None:
            # Fallback to size-based estimation for unknown objects
            x1, y1, x2, y2 = bbox
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            
            # Use relative size for depth estimation
            image_diagonal = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
            object_diagonal = np.sqrt(pixel_width**2 + pixel_height**2)
            relative_size = object_diagonal / image_diagonal
            depth = self.BASE_DEPTH * (1 / relative_size)
        
        # Calculate final dimensions using depth calculator
        return self.depth_calculator.calculate_real_dimensions(
            depth,
            bbox,
            camera_matrix
        )
    
    def _draw_measurements(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        dimensions: Dict[str, float]
    ) -> None:
        """Draw measurements on the image"""
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw measurements
        text = f"W: {dimensions['width']}mm"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        text = f"H: {dimensions['height']}mm"
        cv2.putText(image, text, (x2+5, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        text = f"D: {dimensions['depth']}mm"
        cv2.putText(image, text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)