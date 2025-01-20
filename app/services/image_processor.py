import cv2
import numpy as np
from ultralytics import YOLO
import base64
from pathlib import Path
from typing import Tuple, Dict, List

class ImageProcessor:
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 300
    BASE_DEPTH = 1000  # baseline depth in mm
    MAX_SIZE = 640  # YOLO11 default size

    def __init__(self):
        # Initialize YOLO11 model (using medium variant for balanced performance)
        self.model = YOLO('yolo11m-seg.pt')  # Change to yolo11n/s/l/x as needed
        
        # Camera calibration matrix (you should calibrate this for your specific camera)
        self.camera_matrix = np.array([
            [1000, 0, 512],
            [0, 1000, 384],
            [0, 0, 1]
        ], dtype=np.float32)
        
    async def process_image(self, image_path: Path) -> Dict[str, any]:
        try:
            # Validate image path
            if not image_path.exists() or image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                raise ValueError("Invalid image path or unsupported file format")
        
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Failed to load image")

            # Get original dimensions
            original_height, original_width = image.shape[:2]

            # Process image
            processed = self._preprocess_image(image)
            
            # Detect objects using YOLO11
            results = self.model.predict(
                source=processed,
                conf=self.CONF_THRESHOLD,
                iou=self.IOU_THRESHOLD,
                max_det=self.MAX_DETECTIONS,
                verbose=False
            )
            
            # Create visualization image
            visualization = processed.copy()
            
            measurements = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    # Get box coordinates in (x1, y1, x2, y2) format
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    # Calculate real-world dimensions
                    dimensions = self._calculate_real_dimensions(
                        processed, (x1, y1, x2, y2), self.camera_matrix
                    )
                    
                    # Draw bounding box and measurements
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
        # Resize if needed while maintaining aspect ratio
        height, width = image.shape[:2]
        if height > self.MAX_SIZE or width > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(height, width)
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
        
        # Get focal length from camera matrix
        focal_length = camera_matrix[0, 0]
        
        # Calculate object's position in image
        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2
        object_center_x = (x1 + x2) / 2
        object_center_y = (y1 + y2) / 2
        
        # Calculate distance from center (for perspective correction)
        distance_from_center = np.sqrt(
            (object_center_x - image_center_x) ** 2 + 
            (object_center_y - image_center_y) ** 2
        )
        max_distance = np.sqrt(image_center_x ** 2 + image_center_y ** 2)
        distance_factor = 1 + (distance_from_center / max_distance) * 0.5
        
        # Estimate depth using image size and object size relationships
        image_diagonal = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
        object_diagonal = np.sqrt(pixel_width**2 + pixel_height**2)
        relative_size = object_diagonal / image_diagonal
        
        # Base depth calculation (objects taking up less of frame are typically further away)
        depth = self.BASE_DEPTH * (1 / relative_size) * distance_factor
        
        # Calculate real dimensions using the depth and focal length
        real_width = (pixel_width * depth) / focal_length
        real_height = (pixel_height * depth) / focal_length
        
        # Apply aspect ratio correction
        aspect_ratio = pixel_height / pixel_width
        if aspect_ratio > 2 or aspect_ratio < 0.5:
            # For objects with extreme aspect ratios, adjust calculations
            avg_dimension = (real_width + real_height) / 2
            real_width = avg_dimension * np.sqrt(1 / aspect_ratio)
            real_height = avg_dimension * np.sqrt(aspect_ratio)
        
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
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
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
        
        # Draw object type
        cv2.putText(
            image,
            f"{object_type}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )