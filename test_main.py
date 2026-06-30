import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|allowed_extensions;ALL|reconnect;1|reconnect_streamed;1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import threading
import time
import queue
import multiprocessing as mp
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from ultralytics import YOLO
import yt_dlp
import yaml

app = FastAPI()
encoder_pool = ThreadPoolExecutor(max_workers=4)

# Multi-processing Queues for high-speed frame transfer
capture_queue = mp.Queue(maxsize=5)
detection_queue = queue.Queue(maxsize=5)
rendering_queue = queue.Queue(maxsize=5)
shutdown_event = mp.Event()

global_telemetry = {
    "unique_count": 0,
    "fps": 30.0,
    "total_occupancy": 0
}

class SimpleTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_count = 0
    
    def update(self, detections):
        for track in self.tracks:
            track['age'] += 1
            track['hits_since_update'] += 1
        
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(detections), len(self.tracks)))
            for d, det in enumerate(detections):
                for t, track in enumerate(self.tracks):
                    iou_matrix[d, t] = self._calculate_iou(det[:4], track['bbox'])
            
            indices = np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape)
            used_detections = set()
            used_tracks = set()
            
            for det_idx, track_idx in zip(indices[0], indices[1]):
                if det_idx not in used_detections and track_idx not in used_tracks and iou_matrix[det_idx, track_idx] > self.iou_threshold:
                    used_detections.add(det_idx)
                    used_tracks.add(track_idx)
                    if det_idx in unmatched_detections:
                        unmatched_detections.remove(det_idx)
                    
                    track = self.tracks[track_idx]
                    det = detections[det_idx]
                    track['bbox'] = det[:4]
                    track['confidence'] = det[4]
                    track['class_id'] = det[5]
                    track['class_name'] = det[6]
                    track['hits_since_update'] = 0
                    track['hit_streak'] += 1
                    matched_tracks.append(track)
        
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_track = {
                'id': self.track_id_count,
                'bbox': det[:4],
                'confidence': det[4],
                'class_id': det[5],
                'class_name': det[6],
                'age': 0,
                'hit_streak': 1,
                'hits_since_update': 0
            }
            self.tracks.append(new_track)
            self.track_id_count += 1
            
        self.tracks = [t for t in self.tracks if t['hits_since_update'] < self.max_age]
        return [t for t in self.tracks if t['hit_streak'] >= self.min_hits or t['age'] < self.min_hits]

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

def create_dashboard_collage(annotated_frames):
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    items = list(annotated_frames.values())
    valid_frames = [f for f in items if f is not None]
    
    if len(valid_frames) == 0:
        cv2.putText(canvas, "CONNECTING TO LIVE TRACKING FEEDS...", (380, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return canvas
        
    if len(valid_frames) == 1:
        canvas = cv2.resize(valid_frames[0], (1280, 720))
    elif len(valid_frames) == 2:
        f1 = cv2.resize(valid_frames[0], (640, 720))
        f2 = cv2.resize(valid_frames[1], (640, 720))
        canvas[:, :640] = f1
        canvas[:, 640:] = f2
        cv2.line(canvas, (640, 0), (640, 720), (220, 220, 220), 4)
    elif len(valid_frames) >= 3:
        f1 = cv2.resize(valid_frames[0], (640, 360))
        f2 = cv2.resize(valid_frames[1], (640, 360))
        f3 = cv2.resize(valid_frames[2], (640, 360))
        canvas[0:360, 0:640] = f1
        canvas[0:360, 640:1280] = f2
        canvas[360:720, 0:640] = f3
        if len(valid_frames) == 4:
            f4 = cv2.resize(valid_frames[3], (640, 360))
            canvas[360:720, 640:1280] = f4
        cv2.line(canvas, (640, 0), (640, 720), (220, 220, 220), 4)
        cv2.line(canvas, (0, 360), (1280, 360), (220, 220, 220), 4)
        
    return canvas

def producer_worker(cam_id, url, q_capture, evt_shutdown):
    ydl_opts = {
        'format': 'best[height<=720]', 
        'quiet': True,
        'extractor_args': {'youtube': ['player_client=ios']}
    }
    cap = None
    while not evt_shutdown.is_set():
        if cap is None:
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    resolved_url = info.get('url')
                cap = cv2.VideoCapture(resolved_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                time.sleep(2)
                continue
                
        ret, frame = cap.read()
        if ret:
            try:
                q_capture.put((cam_id, frame), timeout=0.1)
            except queue.Full:
                pass
        else:
            cap = None
            time.sleep(1)

def processor_worker(camera_configs):
    model_path = "weights/yolov8n.engine" if os.path.exists("weights/yolov8n.engine") else ("weights/yolov8n.onnx" if os.path.exists("weights/yolov8n.onnx") else "yolov8n.pt")
    model = YOLO(model_path, task="detect")
    if model_path.endswith(".pt"):
        model.to("cuda")
        
    trackers = {cfg['id']: SimpleTracker() for cfg in camera_configs}
    class_names = model.names
    
    while not shutdown_event.is_set():
        try:
            cam_id, frame = capture_queue.get(timeout=1)
            results = model(frame, conf=0.4, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = class_names[class_id]
                        detections.append([x1, y1, x2, y2, confidence, class_id, class_name])
                        
            confirmed_tracks = trackers[cam_id].update(detections)
            detection_queue.put((cam_id, frame, confirmed_tracks))
        except queue.Empty:
            continue

def renderer_worker():
    global global_telemetry
    last_frames = {}
    unique_ids = set()
    
    target_fps = 30
    frame_duration = 1.0 / target_fps
    
    while not shutdown_event.is_set():
        loop_start = time.time()
        
        try:
            cam_id, frame, tracks = detection_queue.get_nowait()
            annotated_frame = frame.copy()
            
            for track in tracks:
                tx1, ty1, tx2, ty2 = map(int, track['bbox'])
                tid = track['id']
                tname = track['class_name']
                tconf = track['confidence']
                
                unique_ids.add(tid)
                
                np.random.seed(tid)
                color = tuple(map(int, np.random.randint(0, 255, 3)))
                cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), color, 2)
                label = f"{tname} #{tid} ({tconf:.2f})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (tx1, ty1 - label_size[1] - 10), (tx1 + label_size[0], ty1), color, -1)
                cv2.putText(annotated_frame, label, (tx1, ty1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            last_frames[cam_id] = annotated_frame
            global_telemetry["total_occupancy"] = len(tracks)
        except queue.Empty:
            pass
            
        if last_frames:
            canvas = create_dashboard_collage(last_frames)
            try:
                rendering_queue.put_nowait((time.time(), canvas))
            except queue.Full:
                pass
        
        global_telemetry["unique_count"] = len(unique_ids)
        elapsed = time.time() - loop_start
        actual_fps = 1.0 / elapsed if elapsed > 0 else target_fps
        global_telemetry["fps"] = round(actual_fps, 1)
        
        time.sleep(max(0, frame_duration - elapsed))

def _sync_compress(canvas_frame):
    status, encoded_img = cv2.imencode('.jpg', canvas_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return encoded_img.tobytes() if status else None

@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    
    while not shutdown_event.is_set():
        try:
            _, canvas = rendering_queue.get(timeout=1.0)
            jpeg_bytes = await loop.run_in_executor(encoder_pool, _sync_compress, canvas)
            if jpeg_bytes:
                await websocket.send_bytes(jpeg_bytes)
        except queue.Empty:
            continue

@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    while not shutdown_event.is_set():
        try:
            await websocket.send_json({
                "telemetry": {
                    "total_occupancy": global_telemetry["total_occupancy"],
                    "unique_footfall": global_telemetry["unique_count"],
                    "total_unique_people": global_telemetry["unique_count"],
                    "target_color_count": global_telemetry["total_occupancy"],
                    "system_fps": global_telemetry["fps"],
                }
            })
            await asyncio.sleep(1)
        except WebSocketDisconnect:
            break

@app.get("/")
def get_dashboard():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    camera_configs = config.get("cameras", [])
    
    shutdown_event.clear()
    
    producers = []
    for cfg in camera_configs:
        p = mp.Process(target=producer_worker, args=(cfg['id'], cfg['rtsp_url'], capture_queue, shutdown_event), daemon=True)
        p.start()
        producers.append(p)
    
    threading.Thread(target=processor_worker, args=(camera_configs,), daemon=True).start()
    threading.Thread(target=renderer_worker, daemon=True).start()
    
    yield
    shutdown_event.set()
    for p in producers:
        p.terminate()

app.router.lifespan_context = lifespan

if __name__ == '__main__':
    mp.freeze_support()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
