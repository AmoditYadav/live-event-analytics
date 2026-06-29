import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import asyncio
import cv2
import json
import yaml
import uvicorn
import numpy as np
import time
import csv
import subprocess
import shutil
import requests
import threading
from contextlib import asynccontextmanager
import supervision as sv
from fastapi import FastAPI
import warnings
import imageio_ffmpeg
from core.video_stream import MultiCameraReader
from core.detector import YOLODetector
from core.tracker import EventTracker
from core.color_filter import filter_by_clothing_color
from app.server import app, manager

if not hasattr(np, 'long'):
    np.long = np.int64

warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

global_dashboard_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
ffmpeg_process_lock = threading.Lock()
current_ffmpeg_process = None


def make_status_canvas(lines, size=(1280, 720)):
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(canvas, "LIVE EVENT ANALYTICS", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (56, 189, 248), 3)
    y = 145
    for line in lines:
        cv2.putText(canvas, str(line), (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 240, 255), 2)
        y += 45
    cv2.putText(canvas, time.strftime("%Y-%m-%d %H:%M:%S"), (40, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 155, 190), 2)
    return canvas

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
    elif n == 3:
        canvas[0:360, 0:640] = resized[0]
        canvas[0:360, 640:1280] = resized[1]
        canvas[360:720, 0:640] = resized[2]
    else:
        canvas[0:360, 0:640] = resized[0]
        canvas[0:360, 640:1280] = resized[1]
        canvas[360:720, 0:640] = resized[2]
        canvas[360:720, 640:1280] = resized[3]
    return canvas

def resolve_ffmpeg(config_path: str) -> str:
    if config_path and config_path != "ffmpeg" and os.path.isfile(config_path):
        return config_path
    bundled = imageio_ffmpeg.get_ffmpeg_exe()
    if bundled and os.path.isfile(bundled):
        return bundled
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return "ffmpeg"

def probe_nvenc(ffmpeg_exe: str) -> bool:
    black = bytes(320 * 240 * 3)
    test = subprocess.run(
        [
            ffmpeg_exe, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "320x240", "-r", "30", "-i", "-",
            "-c:v", "h264_nvenc", "-preset", "p1", "-frames:v", "1",
            "-f", "null", "-",
        ],
        input=black,
        capture_output=True,
        timeout=10,
    )
    return test.returncode == 0

def make_ffmpeg_process(ffmpeg_exe: str, use_nvenc: bool) -> subprocess.Popen:
    video_codec_args = (
        ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull", "-profile:v", "baseline", "-pix_fmt", "yuv420p", "-b:v", "5M"]
        if use_nvenc
        else ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-profile:v", "baseline", "-pix_fmt", "yuv420p", "-b:v", "5M"]
    )
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "1280x720",
        "-r", "30",
        "-i", "-",
        *video_codec_args,
        "-f", "rtsp",
        "rtsp://localhost:8554/dashboard",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

def set_ffmpeg_process(proc):
    global current_ffmpeg_process
    with ffmpeg_process_lock:
        current_ffmpeg_process = proc


def get_ffmpeg_process():
    with ffmpeg_process_lock:
        return current_ffmpeg_process


def ffmpeg_stderr_logger(proc):
    try:
        for raw_line in iter(proc.stderr.readline, b""):
            line = raw_line.decode(errors="replace").strip()
            if line:
                print(f"[ffmpeg] {line}")
    except Exception as exc:
        print(f"[ffmpeg] stderr logger stopped: {exc}")


def stream_pusher():
    global global_dashboard_canvas
    while True:
        ffmpeg_proc = get_ffmpeg_process()
        if ffmpeg_proc is not None and ffmpeg_proc.poll() is None and ffmpeg_proc.stdin:
            try:
                ffmpeg_proc.stdin.write(global_dashboard_canvas.tobytes())
                ffmpeg_proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                print(f"[pipeline] FFmpeg stdin unavailable: {exc}")
        time.sleep(1.0 / 30.0)

def orchestrator_loop(loop):
    global global_dashboard_canvas
    global_dashboard_canvas = make_status_canvas([
        "Starting pipeline...",
        "Waiting for camera frames and MediaMTX.",
        "If this remains visible, verify RTSP inputs in config/config.yaml.",
    ])
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
    
    ffmpeg_exe = resolve_ffmpeg(config.get("ffmpeg_path", "ffmpeg"))
    use_nvenc = probe_nvenc(ffmpeg_exe)
    encoder_name = "h264_nvenc" if use_nvenc else "libx264"
    print(f"[pipeline] ffmpeg: {ffmpeg_exe}")
    print(f"[pipeline] video encoder: {encoder_name}")
    
    model_path = "yolov8m.engine" if os.path.exists("yolov8m.engine") else "yolov8m.pt"
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
    
    ffmpeg_process = make_ffmpeg_process(ffmpeg_exe, use_nvenc)
    set_ffmpeg_process(ffmpeg_process)
    print(f"[pipeline] FFmpeg PID {ffmpeg_process.pid} -> rtsp://localhost:8554/dashboard")
    threading.Thread(target=ffmpeg_stderr_logger, args=(ffmpeg_process,), daemon=True).start()
    threading.Thread(target=stream_pusher, daemon=True).start()
    
    print("[pipeline] Waiting for MediaMTX WHEP stream to become available...")
    while True:
        try:
            resp = requests.get("http://localhost:8889/dashboard", timeout=2)
            if resp.status_code == 200:
                break
        except requests.RequestException:
            pass
        if ffmpeg_process.poll() is not None:
            print("[pipeline] FFmpeg exited before MediaMTX became ready - restarting...")
            ffmpeg_process = make_ffmpeg_process(ffmpeg_exe, use_nvenc)
            set_ffmpeg_process(ffmpeg_process)
            threading.Thread(target=ffmpeg_stderr_logger, args=(ffmpeg_process,), daemon=True).start()
        time.sleep(1)
    
    print("[pipeline] RTSP stream primed - dashboard path is live on MediaMTX")
    
    global_stats = {"total_occupancy": 0, "fps": 0.0, "chokepoint_footfall": 0, "total_identified": 0}
    
    while True:
        loop_start = time.time()
        
        if ffmpeg_process.poll() is not None:
            print("[pipeline] FFmpeg died - restarting...")
            ffmpeg_process = make_ffmpeg_process(ffmpeg_exe, use_nvenc)
            set_ffmpeg_process(ffmpeg_process)
            threading.Thread(target=ffmpeg_stderr_logger, args=(ffmpeg_process,), daemon=True).start()
        
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
        else:
            global_dashboard_canvas = make_status_canvas([
                "No camera frames received yet.",
                "Telemetry WebSocket may connect even when video inputs are unavailable.",
                "Check MediaMTX paths: rtsp://localhost:8554/cam1 and /cam2.",
            ])
            
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    task_thread = threading.Thread(target=orchestrator_loop, args=(loop,), daemon=True)
    task_thread.start()
    yield

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)