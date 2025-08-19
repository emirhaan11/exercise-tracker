import cv2
import mediapipe as mp


class PoseEstimator:
    def __init__(self, static=False, complexity=1, smooth=True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=static,
                                      model_complexity=complexity,
                                      enable_segmentation=False,
                                      smooth_segmentation=smooth,
                                      smooth_landmarks=True)
        self.drawer = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        return result

    def draw(self, frame_bgr, res):
        if res.pose_landmarks:
            self.drawer.draw_landmarks(frame_bgr, res.pose_landmarks,
                                       self.mp_pose.POSE_CONNECTIONS,
                                       landmark_drawing_spec=self.styles.get_default_pose_landmarks_style())
        return frame_bgr