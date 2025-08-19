import cv2
from core.capture import VideoCapture

if __name__ == '__main__':
    cam = VideoCapture()

    while cam.is_open():
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break
        cv2.putText(frame,
                    f'FPS: {cam.fps:.1f}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
