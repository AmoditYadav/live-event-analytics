import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|allowed_extensions;ALL|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5"
import asyncio
import cv2
import json
import yaml
import uvicorn
import numpy as np
import time
import csv
import requests
import threading
from contextlib import asynccontextmanager
import supervision as sv
from fastapi import FastAPI, WebSocket
import warnings

from core.video_stream import MultiCameraReader
from core.detector import YOLODetector
from core.tracker import EventTracker
from core.color_filter import filter_by_clothing_color
from app.server import app, manager

if not hasattr(np, 'long'):
    np.long = np.int64

warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

global_dashboard_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

def create_dashboard_collage(frames, target_size=(640, 360)):
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    if not frames:
        return canvas
    resized = [cv2.resize(f, target_size) for f in frames]
    n = len(resized)
    if n == 1:
        canvas[0:360, 0:640] = resized[0]
    elif n == 2:
        canvas[0:360, 0:640] = resized[0]
        canvas[0:360, 640:1280] = resized[1]
        cv2.line(canvas, (640, 0), (640, 360), (220, 220, 220), 4)
    elif n == 3:
        canvas[0:360, 0:640] = resized[0]
        canvas[0:360, 640:1280] = resized[1]
        canvas[360:720, 0:640] = resized[2]
        cv2.line(canvas, (640, 0), (640, 360), (220, 220, 220), 4)
        cv2.line(canvas, (0, 360), (1280, 360), (220, 220, 220), 4)
    else:
        canvas[0:360, 0:640] = resized[0]
        canvas[0:360, 640:1280] = resized[1]
        canvas[360:720, 0:640] = resized[2]
        canvas[360:720, 640:1280] = resized[3]
        cv2.line(canvas, (640, 0), (640, 720), (220, 220, 220), 4)
        cv2.line(canvas, (0, 360), (1280, 360), (220, 220, 220), 4)
    return canvas


def orchestrator_loop(loop):
    global global_dashboard_canvas
    csv_filename = f"analytics_export_{int(time.time())}.csv"
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Total Occupancy", "Unique Chokepoint Footfall", "Total People Identified"])
    
    loop_times = []
    last_log_time = time.time()
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    camera_configs = config.get("cameras", [])
    

    
    model_path = "weights/yolov8n.engine" if os.path.exists("weights/yolov8n.engine") else ("weights/yolov8n.onnx" if os.path.exists("weights/yolov8n.onnx") else "yolov8n.pt")
    print(f"[pipeline] Loading YOLO model: {model_path} ...")
    detector = YOLODetector(model_path)
    print("[pipeline] YOLO ready.")
    
    print("[pipeline] Loading Re-ID model ...")
    tracker = EventTracker(camera_configs)
    print("[pipeline] Re-ID ready.")
    
    camera_reader = MultiCameraReader(camera_configs)
    camera_reader.start()
    box_annotator = sv.BoxAnnotator(thickness=2)
    last_known_frames = {}
    

    
    global_stats = {"total_occupancy": 0, "fps": 0.0, "chokepoint_footfall": 0, "total_identified": 0}
    
    while True:
        loop_start = time.time()
        
        frames_dict = camera_reader.read()
        for cam_id, frame in frames_dict.items():
            if frame is not None:
                last_known_frames[cam_id] = frame
        
        if last_known_frames:
            active_cams = list(last_known_frames.keys())
            batch_frames = [last_known_frames[cam_id] for cam_id in active_cams]
            batch_detections = detector.detect(batch_frames)
            
            global_stats["total_occupancy"] = 0
            target_color = manager.target_color
            target_color_count = 0
            annotated_frames = []
            
            for idx, cam_id in enumerate(active_cams):
                frame = batch_frames[idx]
                detections = batch_detections[idx]
                
                tracked_objects = tracker.update(cam_id, frame, detections)
                global_stats["total_occupancy"] += len(tracked_objects)
                
                filtered_detections = tracked_objects
                if target_color and target_color != "all":
                    filtered_detections = filter_by_clothing_color(frame, tracked_objects, target_color)
                
                target_color_count += len(filtered_detections)
                
                annotated = box_annotator.annotate(scene=frame.copy(), detections=filtered_detections)
                
                if filtered_detections.tracker_id is not None:
                    for i, box in enumerate(filtered_detections.xyxy):
                        tid = filtered_detections.tracker_id[i]
                        cv2.putText(annotated, f"#{tid}", (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                annotated_frames.append(annotated)
            
            if annotated_frames:
                global_dashboard_canvas = create_dashboard_collage(annotated_frames)
            
            loop_times.append(time.time() - loop_start)
            if len(loop_times) > 30:
                loop_times.pop(0)
            
            global_stats["fps"] = round(1.0 / (sum(loop_times) / len(loop_times)), 1) if loop_times else 0.0
            
            if time.time() - last_log_time >= 5.0:
                print(f"[pipeline] FPS: {global_stats['fps']} | Occupancy: {global_stats['total_occupancy']}")
                last_log_time = time.time()
                with open(csv_filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        global_stats["total_occupancy"],
                        tracker.get_total_count(),
                        tracker.get_total_unique_ids()
                    ])
            
            payload = {
                "telemetry": {
                    "total_occupancy": global_stats["total_occupancy"],
                    "unique_footfall": tracker.get_total_count(),
                    "total_unique_people": tracker.get_total_unique_ids(),
                    "target_color_count": target_color_count if target_color != "all" else global_stats["total_occupancy"],
                    "system_fps": global_stats["fps"],
                }
            }
            asyncio.run_coroutine_threadsafe(manager.broadcast(json.dumps(payload)), loop)
        
        elapsed = time.time() - loop_start
        sleep_time = max(0.001, (1.0 / 30.0) - elapsed)
        time.sleep(sleep_time)




@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    while True:
        await asyncio.sleep(0.016)
        if global_dashboard_canvas is not None:
            _, buffer = cv2.imencode('.jpg', global_dashboard_canvas, [cv2.IMWRITE_JPEG_QUALITY, 50])
            await websocket.send_bytes(buffer.tobytes())

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    task_thread = threading.Thread(target=orchestrator_loop, args=(loop,), daemon=True)
    task_thread.start()
    yield

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)