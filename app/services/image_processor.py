#app/services/image_processor.py
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from pathlib import Path
from typing import Tuple, Dict, List

class ImageProcessor:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Camera calibration matrix (you should calibrate this for your specific camera)
        self.camera_matrix = np.array([
            [1000, 0, 512],
            [0, 1000, 384],
            [0, 0, 1]
        ], dtype=np.float32)
        
    async def process_image(self, image_path: Path) -> Dict[str, any]:
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Failed to load image")

            # Get original dimensions
            original_height, original_width = image.shape[:2]

            # Process image
            processed = self._preprocess_image(image)
            
            # Detect objects using YOLO
            results = self.model(processed)
            
            # Create visualization image
            visualization = processed.copy()
            
            measurements = []
            for r in results[0].boxes:
                box = r.xyxy[0].cpu().numpy()  # get box coordinates in (x1, y1, x2, y2) format
                x1, y1, x2, y2 = map(int, box)
                conf = float(r.conf[0])
                cls = int(r.cls[0])
                class_name = self.model.names[cls]
                
                # Calculate real-world dimensions
                dimensions = self._calculate_real_dimensions(
                    processed, (x1, y1, x2, y2), self.camera_matrix
                )
                
                # Draw bounding box
                cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw dimensions
                self._draw_measurements(
                    visualization,
                    (x1, y1, x2, y2),
                    dimensions,
                    class_name
                )
                
                measurements.append({
                    "object_type": class_name,
                    "dimensions": dimensions,
                    "confidence_score": float(conf),
                    "bbox": [x1, y1, x2, y2]
                })
            
            # Encode the visualization image
            _, buffer = cv2.imencode('.jpg', visualization)
            visualization_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "measurements": measurements,
                "visualization": visualization_base64,
                "image_dimensions": {
                    "width": original_width,
                    "height": original_height
                }
            }
            
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Resize if needed
        max_size = 1024
        height, width = image.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        return image

    def _calculate_real_dimensions(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        camera_matrix: np.ndarray
    ) -> Dict[str, float]:
        x1, y1, x2, y2 = bbox
        
        # Calculate pixel dimensions
        pixel_width = x2 - x1
        pixel_height = y2 - y1
        
        # Estimate depth using size and position in image
        image_center = image.shape[1] / 2
        object_center_x = (x1 + x2) / 2
        
        # Calculate depth based on focal length and assumed real-world size
        focal_length = camera_matrix[0, 0]
        assumed_width = 500  # mm (assumed average object width)
        depth = (focal_length * assumed_width) / pixel_width
        
        # Calculate real dimensions using similar triangles
        real_width = (pixel_width * depth) / focal_length
        real_height = (pixel_height * depth) / focal_length
        
        return {
            "width": round(real_width, 1),
            "height": round(real_height, 1),
            "depth": round(depth, 1),
            "unit": "mm"
        }

    def _draw_measurements(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        dimensions: Dict[str, float],
        object_type: str
    ) -> None:
        x1, y1, x2, y2 = bbox
        
        # Draw width measurement
        cv2.line(image, (x1, y2 + 20), (x2, y2 + 20), (0, 0, 255), 2)
        cv2.line(image, (x1, y2 + 15), (x1, y2 + 25), (0, 0, 255), 2)
        cv2.line(image, (x2, y2 + 15), (x2, y2 + 25), (0, 0, 255), 2)
        cv2.putText(
            image,
            f"Width: {dimensions['width']:.1f}mm",
            (x1, y2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        
        # Draw height measurement
        cv2.line(image, (x2 + 20, y1), (x2 + 20, y2), (0, 0, 255), 2)
        cv2.line(image, (x2 + 15, y1), (x2 + 25, y1), (0, 0, 255), 2)
        cv2.line(image, (x2 + 15, y2), (x2 + 25, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            f"Height: {dimensions['height']:.1f}mm",
            (x2 + 30, (y1 + y2) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        
        # Draw object type and confidence
        cv2.putText(
            image,
            f"{object_type}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )