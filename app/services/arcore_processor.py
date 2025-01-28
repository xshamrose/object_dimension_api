# app/services/arcore_processor.py
from google.cloud import vision
from google.cloud.vision_v1 import types
import numpy as np
import cv2
import base64
from pathlib import Path
from google.cloud import vision
import google.auth.credentials
# Create credentials from the API key
from google.oauth2 import credentials
TOKEN_KEY = "YOUR_API_KEY_ENV_VARIABLE_NAME"
creds = credentials.Credentials(token=(TOKEN_KEY))
class ARCoreProcessor:
    CONF_THRESHOLD = 0.5
    MAX_DETECTIONS = 100
    BASE_DEPTH = 1000  # baseline depth in mm
    MAX_SIZE = 640
    def __init__(self):
        self.vision_client = vision.ImageAnnotatorClient(credentials=creds)

    async def process_image(self, image_path: Path) -> dict:
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

            # Preprocess image
            processed = self._preprocess_image(image)

            # Detect objects using Google Vision API
            objects = await self._detect_objects(processed)

            # Measure object dimensions
            measurements = []
            visualization = processed.copy()
            for obj in objects:
                x1, y1, x2, y2 = obj.bounding_box.vertices[0].x, obj.bounding_box.vertices[0].y, \
                                obj.bounding_box.vertices[2].x, obj.bounding_box.vertices[2].y
                dimensions = self._calculate_real_dimensions(
                    processed, (x1, y1, x2, y2), self.vision_client.intrinsic_params
                )
                self._draw_measurements(visualization, (x1, y1, x2, y2), dimensions, obj.name)
                measurements.append({
                    "object_type": obj.name,
                    "dimensions": dimensions,
                    "confidence_score": obj.score,
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

    async def _detect_objects(self, image: np.ndarray) -> list:
        # Convert the image to a format compatible with the Vision API
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        image = types.Image(content=image_bytes)

        # Detect objects using the Vision API
        objects = self.vision_client.object_localization(image=image).localized_object_annotations

        # Filter out low-confidence detections
        return [obj for obj in objects if obj.score >= self.CONF_THRESHOLD]

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
        bbox: tuple,
        camera_matrix: np.ndarray
    ) -> dict:
        # Implement depth estimation and real-world dimensions calculation
        # using the Google Vision API's camera matrix and object bounding box
        x1, y1, x2, y2 = bbox
        pixel_width = x2 - x1
        pixel_height = y2 - y1

        # Estimate depth using the Vision API's depth estimation
        depth = self._estimate_depth(camera_matrix, pixel_width, pixel_height)

        # Calculate real dimensions using the depth and camera matrix
        real_width = (pixel_width * depth) / camera_matrix[0, 0]
        real_height = (pixel_height * depth) / camera_matrix[1, 1]

        return {
            "width": round(real_width, 1),
            "height": round(real_height, 1),
            "depth": round(depth, 1),
            "unit": "mm"
        }

    def _estimate_depth(
        self,
        camera_matrix: np.ndarray,
        pixel_width: int,
        pixel_height: int
    ) -> float:
        # Implement depth estimation using the Google Vision API's camera matrix
        # and object bounding box dimensions
        focal_length = camera_matrix[0, 0]
        return self.BASE_DEPTH * (focal_length / max(pixel_width, pixel_height))

    def _draw_measurements(
        self,
        image: np.ndarray,
        bbox: tuple,
        dimensions: dict,
        object_type: str
    ) -> None:
        # Implement drawing bounding boxes and measurement annotations on the image
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{object_type} - {dimensions['width']:.1f}x{dimensions['height']:.1f}x{dimensions['depth']:.1f}mm",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )