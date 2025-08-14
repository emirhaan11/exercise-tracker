import cv2
from capture import VideoCapture
from pose import PoseEstimator

if __name__ == '__main__':
    cam = VideoCapture()
    pose = PoseEstimator()

    while cam.is_open():
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        result = pose.process(frame)
        frame = pose.draw(frame, result)
        cv2.putText(frame,
                    f'fps: {cam.fps:.1f}',
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),
                    2)
        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cam.release()
    cv2.destroyAllWindows()