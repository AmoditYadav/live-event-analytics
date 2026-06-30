import cv2
import threading
import time
import logging
import yt_dlp

class MultiCameraReader:
    def __init__(self, camera_configs, queue_size=1, reconnect_delay=5.0):
        self.camera_configs = camera_configs
        self.reconnect_delay = reconnect_delay
        self.latest_frames = {cam['id']: None for cam in camera_configs}
        self._lock = threading.Lock()
        self.running = False
        self.threads = []

    def start(self):
        self.running = True
        for cam in self.camera_configs:
            t = threading.Thread(target=self._capture_loop, args=(cam,), daemon=True)
            self.threads.append(t)
            t.start()

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join()

    def _resolve_url(self, url):
        ydl_opts = {'format': 'best[height<=720]', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            return info_dict.get('url')

    def _capture_loop(self, cam):
        cam_id = cam['id']
        url = cam['rtsp_url']

        while self.running:
            direct_url = self._resolve_url(url)
            cap = cv2.VideoCapture(direct_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                logging.warning(f"Camera {cam_id} failed to open. Retrying in {self.reconnect_delay}s...")
                cap.release()
                time.sleep(self.reconnect_delay)
                continue

            logging.info(f"Camera {cam_id} connected.")

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Camera {cam_id} lost connection. Reconnecting...")
                    break
                with self._lock:
                    self.latest_frames[cam_id] = frame

            cap.release()

            if self.running:
                time.sleep(self.reconnect_delay)

    def read(self):
        with self._lock:
            return dict(self.latest_frames)
