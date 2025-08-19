import cv2
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
video_path = HERE.parent / 'raw' / 's7.mp4'

class VideoCapture:
    def __init__(self, width=640, height=360):
        self.cap = cv2.VideoCapture(str(video_path))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = 0.0
        self.prev = time.time()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return ret, None

        now = time.time()
        dt = now - self.prev
        if dt > 0:
            self.fps = 1.0 / dt

        self.prev = now
        return True, frame

    def is_open(self):
        return self.cap.isOpened()

    def release(self):
        if self.cap:
            self.cap.release()
