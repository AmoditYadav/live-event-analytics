import cv2
import threading
import time
from collections import deque
import logging
import yt_dlp

class MultiCameraReader:
    def __init__(self, camera_configs, queue_size=1, reconnect_delay=5.0):
        self.camera_configs = camera_configs
        self.reconnect_delay = reconnect_delay
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
        url = cam['rtsp_url']

        while self.running:
            try:
                ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=False)
                    direct_url = info_dict.get('url')
                cap = cv2.VideoCapture(direct_url)
            except Exception:
                cap = cv2.VideoCapture(url)

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
                current_frames[cam_id] = self.queues[cam_id].pop()
            except IndexError:
                current_frames[cam_id] = None
        return current_frames
