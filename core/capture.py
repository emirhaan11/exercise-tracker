import cv2
import time


class CamCapture:
    def __init__(self, cam_index=0, width=1080, height=720):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.prev = time.time()
        self.fps = 0.0

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
