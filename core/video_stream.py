import os

import cv2
import threading
import logging
import yt_dlp

class MultiCameraReader:
    def __init__(self, camera_configs, queue_size=1, reconnect_delay=5.0):
        self.camera_configs = camera_configs
        self.latest_frames = {cam['id']: None for cam in camera_configs}
        self._lock = threading.Lock()
        self.threads = []

    def start(self):
        for cam in self.camera_configs:
            t = threading.Thread(
                target=self._capture_loop,
                args=(cam['id'], cam['rtsp_url']),
                daemon=True,
            )
            self.threads.append(t)
            t.start()

    def stop(self):
        for t in self.threads:
            t.join()

    def _resolve_url(self, url):
        ydl_opts = {'format': 'best[height<=720]', 'quiet': True, 'extractor_args': {'youtube': ['player_client=android']}}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            return info_dict.get('url')

    def _capture_loop(self, cam_id, url):
        direct_url = self._resolve_url(url)
        cap = cv2.VideoCapture(direct_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logging.warning(f"Camera {cam_id} failed to open.")
            cap.release()
            return

        logging.info(f"Camera {cam_id} connected.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Camera {cam_id} read failed.")
                break
            with self._lock:
                self.latest_frames[cam_id] = frame

        cap.release()

    def read(self):
        with self._lock:
            return dict(self.latest_frames)
