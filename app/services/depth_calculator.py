import numpy as np
from typing import Tuple, Dict

class DepthCalculator:
    # Constants for typical smartphone camera parameters
    SENSOR_HEIGHT_MM = 4.8  # Typical smartphone camera sensor height
    SENSOR_WIDTH_MM = 6.4   # Typical smartphone camera sensor width
    
    def calculate_depth_from_camera(
        self,
        focal_length_pixels: float,
        real_object_height_mm: float,
        pixel_height: float
    ) -> float:
        """
        Calculate depth using the basic formula:
        distance = (focal_length * real_object_height) / pixel_height
        
        Args:
            focal_length_pixels: Camera's focal length in pixels
            real_object_height_mm: Known/reference object height in mm
            pixel_height: Object height in pixels
        
        Returns:
            Estimated depth in millimeters
        """
        return (focal_length_pixels * real_object_height_mm) / pixel_height
    
    def estimate_depth_from_known_object(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
        camera_matrix: np.ndarray,
        object_type: str = "phone"
    ) -> float:
        """
        Estimate depth using known object dimensions as reference.
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            image_shape: Image dimensions (height, width)
            camera_matrix: Camera intrinsic matrix
            object_type: Type of object for reference dimensions
        
        Returns:
            Estimated depth in millimeters
        """
        # Reference dimensions for common objects (in mm)
        REFERENCE_OBJECTS = {
            "phone": {"height": 150, "width": 75},  # Average smartphone size
            "credit_card": {"height": 53.98, "width": 85.60},
            "a4_paper": {"height": 297, "width": 210},
        }
        
        x1, y1, x2, y2 = bbox
        pixel_height = y2 - y1
        pixel_width = x2 - x1
        
        focal_length = camera_matrix[0, 0]  # Focal length in pixels
        
        if object_type in REFERENCE_OBJECTS:
            ref_height = REFERENCE_OBJECTS[object_type]["height"]
            ref_width = REFERENCE_OBJECTS[object_type]["width"]
            
            # Calculate depth using both height and width
            depth_from_height = self.calculate_depth_from_camera(
                focal_length, ref_height, pixel_height
            )
            depth_from_width = self.calculate_depth_from_camera(
                focal_length, ref_width, pixel_width
            )
            
            # Use the average of both measurements
            depth = (depth_from_height + depth_from_width) / 2
            
            # Apply distance-based correction
            image_center_x = image_shape[1] / 2
            image_center_y = image_shape[0] / 2
            object_center_x = (x1 + x2) / 2
            object_center_y = (y1 + y2) / 2
            
            # Calculate distance from center for perspective correction
            distance_from_center = np.sqrt(
                (object_center_x - image_center_x) ** 2 + 
                (object_center_y - image_center_y) ** 2
            )
            max_distance = np.sqrt(image_center_x ** 2 + image_center_y ** 2)
            correction_factor = 1 + (distance_from_center / max_distance) * 0.2
            
            return depth * correction_factor
        
        return None

    def calculate_real_dimensions(
        self,
        depth: float,
        bbox: Tuple[int, int, int, int],
        camera_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate real-world dimensions using the estimated depth.
        
        Args:
            depth: Estimated depth in millimeters
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            camera_matrix: Camera intrinsic matrix
        
        Returns:
            Dictionary containing width, height in millimeters
        """
        x1, y1, x2, y2 = bbox
        pixel_width = x2 - x1
        pixel_height = y2 - y1
        
        focal_length = camera_matrix[0, 0]
        
        # Calculate real dimensions using similar triangles
        real_width = (pixel_width * depth) / focal_length
        real_height = (pixel_height * depth) / focal_length
        
        # Apply aspect ratio correction for extreme cases
        aspect_ratio = pixel_height / pixel_width
        if aspect_ratio > 2 or aspect_ratio < 0.5:
            avg_dimension = (real_width + real_height) / 2
            real_width = avg_dimension * np.sqrt(1 / aspect_ratio)
            real_height = avg_dimension * np.sqrt(aspect_ratio)
        
        return {
            "width": round(real_width, 1),
            "height": round(real_height, 1),
            "depth": round(depth, 1),
            "unit": "mm"
        }