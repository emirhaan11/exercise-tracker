import cv2
from capture import VideoCapture
from pose import PoseEstimator
from features import angle_3pts
from utils import lm_xy

LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 24, 26, 28

if __name__ == '__main__':
    cam = VideoCapture()
    pose = PoseEstimator()


    while cam.is_open():
        ret, frame = cam.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        res = pose.process(frame)

        knee_angle = None

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            hip = lm_xy(lms[LEFT_HIP], w, h)
            knee = lm_xy(lms[LEFT_KNEE], w, h)
            ankle = lm_xy(lms[LEFT_ANKLE], w, h)

            knee_angle = angle_3pts(hip, knee, ankle)

        out = frame.copy()
        out = pose.draw(out, res)

        if knee_angle is not None:
            cv2.putText(out,
                        f'Knee angle: {knee_angle:.1f}',
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0) if knee_angle <= 100 else (0,0,255),
                        2
                        )

        cv2.imshow('Camera', out)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cam.release()
    cv2.destroyAllWindows()