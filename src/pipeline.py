import time
import cv2
from capture import VideoCapture
from pose import PoseEstimator
from features import angle_3pts, ema, lm_xy
from rules import squat_depth_ok
from rep_counter import SquatRepCounter, SquatRepCounterConfig

LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 23, 25, 27
RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 12, 24, 26, 28

if __name__ == '__main__':
    # ADDING OBJECTS
    cam = VideoCapture()
    pose = PoseEstimator()
    counter = SquatRepCounter()
    #############################################################################################################

    # VARIABLES
    smooth_angle = None
    state = None
    reps = 0
    #############################################################################################################

    # OPENING CAM
    while cam.is_open():
        ret, frame = cam.read()
        if not ret:
            break

        #############################################################################################################

        # DETERMINING THE ANGLES
        h, w = frame.shape[:2]
        res = pose.process(frame)
        knee_angle = None

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark

            hip = lm_xy(lms[LEFT_HIP], w, h)
            knee = lm_xy(lms[LEFT_KNEE], w, h)
            ankle = lm_xy(lms[LEFT_ANKLE], w, h)
            knee_angle = angle_3pts(hip, knee, ankle)

        smooth_angle = ema(smooth_angle, knee_angle, alpha=0.3)

        #############################################################################################################

        # COUNTING REP
        state, reps, just_counted = counter.update(angle=smooth_angle, now=time.monotonic())

        #############################################################################################################

        # ADDING INFORMATION TO THE SCREEN AND SHOWING THE SCREEN
        out = frame.copy()
        out = pose.draw(frame_bgr=out, res=res)

        cv2.putText(out,
                    "ESC: exit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        cv2.putText(out,
                    f"FPS: {cam.fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        if knee_angle is not None:
            ok = squat_depth_ok(knee_angle, threshold=100)
            color = (0, 255, 0) if ok else (0, 0, 255)  # If the angle more than 100 it will be red

            cv2.putText(out,
                        f'Knee angle: {knee_angle:.1f}',
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

        cv2.putText(out,
                    f'Rep: {reps}',
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        cv2.putText(out,
                    f'State: {state}',
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        #############################################################################################################

        # CHECKING POSITION

        # KNEE POSITION


        #############################################################################################################

        cv2.imshow('Camera', out)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
