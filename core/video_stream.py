import cv2
import threading
import time
from collections import deque
import logging
import streamlink

class MultiCameraReader:
    def __init__(self, camera_configs, queue_size=1, reconnect_delay=5.0):
        self.camera_configs = camera_configs
        self.reconnect_delay = reconnect_delay
        
        # Thread-safe ring buffer per camera to drop stale frames
        # maxlen=1 guarantees zero latency by only keeping the newest frame
        self.queues = {cam['id']: deque(maxlen=queue_size) for cam in camera_configs}
        
        self.running = False
        self.threads = []

    def start(self):
        self.running = True
        for cam in self.camera_configs:
            t = threading.Thread(target=self._read_stream, args=(cam,), daemon=True)
            self.threads.append(t)
            t.start()

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join()

    def _read_stream(self, cam):
        cam_id = cam['id']
        rtsp_url = cam['rtsp_url']
        
        while self.running:
            try:
                sl_streams = streamlink.streams(rtsp_url)
                resolved_url = rtsp_url
                if sl_streams:
                    for quality in ("720p", "480p", "best"):
                        if quality in sl_streams:
                            resolved_url = sl_streams[quality].url
                            break
                cap = cv2.VideoCapture(resolved_url)
            except Exception:
                cap = cv2.VideoCapture(rtsp_url)
            
            # Reduce latency by setting buffer size if supported
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                logging.warning(f"Camera {cam_id} failed to open. Retrying in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                continue

            logging.info(f"Camera {cam_id} connected.")
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Camera {cam_id} lost connection. Reconnecting...")
                    break
                
                # Append to deque (automatically drops oldest if full due to maxlen)
                self.queues[cam_id].append(frame)
                
            cap.release()
            
            if self.running:
                time.sleep(self.reconnect_delay)

    def read(self):
        """
        Retrieves the latest available frame from each camera.
        Returns a dictionary {cam_id: frame}.
        """
        current_frames = {}
        for cam_id in self.queues:
            try:
                # Pop the most recent frame, ignoring older ones (zero latency)
                current_frames[cam_id] = self.queues[cam_id].pop()
            except IndexError:
                # Deque is empty
                current_frames[cam_id] = None
                
        return current_frames
