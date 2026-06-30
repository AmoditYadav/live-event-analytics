import numpy as np
from ultralytics import YOLO
import supervision as sv

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model_path = model_path
        self.model = YOLO(model_path, task="detect")
        if model_path.endswith(".pt"):
            self.model.to("cuda")

    def detect(self, frames):
        if not frames:
            return []
        
        if not hasattr(self.model, "to") or self.model_path.endswith(".engine"):
            results = []
            for frame in frames:
                results.append(self.model(frame, verbose=False)[0])
        else:
            results = self.model(frames, batch=len(frames), verbose=False)
            
        batch_detections = []
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.class_id == 0]
            batch_detections.append(detections)
        return batch_detections
