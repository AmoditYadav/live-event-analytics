import cv2
import numpy as np
import supervision as sv

# Predefined HSV ranges for an exhaustive list of colors
# Format: { color_name: [(lower1, upper1), (lower2, upper2), ...] }
# Note: HSV in OpenCV uses H: 0-179, S: 0-255, V: 0-255
# Expanded to capture multiple shades per color while isolating accurately.
COLOR_RANGES = {
    "red": [
        (np.array([0, 80, 50]), np.array([10, 255, 255])),
        (np.array([170, 80, 50]), np.array([180, 255, 255]))
    ],
    "orange": [
        (np.array([11, 80, 50]), np.array([25, 255, 255]))
    ],
    "yellow": [
        (np.array([26, 80, 50]), np.array([35, 255, 255]))
    ],
    "green": [
        (np.array([36, 60, 40]), np.array([85, 255, 255]))
    ],
    "cyan": [
        (np.array([86, 80, 50]), np.array([100, 255, 255]))
    ],
    "blue": [
        (np.array([101, 80, 50]), np.array([130, 255, 255]))
    ],
    "purple": [
        (np.array([131, 80, 50]), np.array([160, 255, 255]))
    ],
    "pink": [
        (np.array([161, 80, 50]), np.array([169, 255, 255]))
    ],
    "black": [
        (np.array([0, 0, 0]), np.array([180, 255, 50]))
    ],
    "white": [
        (np.array([0, 0, 200]), np.array([180, 50, 255]))
    ],
    "grey": [
        (np.array([0, 0, 50]), np.array([180, 50, 200]))
    ],
    "brown": [
        (np.array([10, 50, 20]), np.array([20, 255, 200]))
    ]
}

def filter_by_clothing_color(frame: np.ndarray, detections: sv.Detections, target_color: str = "red") -> sv.Detections:
    """
    Filters detections based on the presence of a specific color within the bounding box.
    """
    if len(detections) == 0:
        return detections
        
    target_color = target_color.lower()
    if target_color not in COLOR_RANGES:
        return detections
        
    ranges = COLOR_RANGES[target_color]
    keep_indices = []
    
    for i, bbox in enumerate(detections.xyxy):
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # Check if the bounding box is valid
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop = frame[y1:y2, x1:x2]
        
        # Convert crop to HSV
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Normalize brightness using CLAHE on the V (Value) channel to handle outdoor sunlight/shadows
        h, s, v = cv2.split(hsv_crop)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_eq = clahe.apply(v)
        hsv_crop = cv2.merge((h, s, v_eq))
        
        # Create an empty mask
        mask = np.zeros(hsv_crop.shape[:2], dtype=np.uint8)
        
        # Apply all HSV ranges for the target color
        for (lower, upper) in ranges:
            color_mask = cv2.inRange(hsv_crop, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)
            
        # Compute the ratio of pixels that match the color
        total_pixels = crop.shape[0] * crop.shape[1]
        matching_pixels = cv2.countNonZero(mask)
        
        ratio = matching_pixels / total_pixels
        
        # Reduced threshold to 15% to accept different shades correctly without being overly sensitive
        if ratio > 0.15:
            keep_indices.append(i)
            
    # Return a filtered sv.Detections object using the keep_indices
    return detections[keep_indices]
