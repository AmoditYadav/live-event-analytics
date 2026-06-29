import numpy as np
from ultralytics import YOLO
import supervision as sv


class YOLODetector:
    def __init__(self, engine_path="yolov8m.pt"):
        self.model = YOLO(engine_path, task="detect")
        self.model.to("cuda")

    def detect(self, frames):
        if not frames:
            return []
        results = self.model(frames, batch=len(frames), verbose=False)
        batch_detections = []
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.class_id == 0]
            batch_detections.append(detections)
        return batch_detections
