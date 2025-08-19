import os
import csv
import cv2

from read_video import VideoCapture
from pose import PoseEstimator
from rep_counter import SquatRepCounter, SquatRepCounterConfig
from features import (
    lm_xy, ema, angle_3pts,
    torso_tilt_deg, hip_knee_alignment_deg, heel_lift_norm,
    squat_depth_ok
)


# Landmarks (left side)
LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 23, 25, 27

# Tuning for offline video
MIN_REP_TIME_SEC   = 0.9
MIN_TOP_FRAMES     = 3
MIN_BOTTOM_FRAMES  = 3

# Output CSV (rep-only)
REP_LOG_PATH = "../../notebooks/logs/bad_rep.csv"
REP_FIELDS = [
    "min_knee_angle",
    "torso_tilt_min",
    "torso_tilt_max",
    "torso_tilt_mean",
    "hip_knee_align_mean_deg",
    "heel_lift_max_norm",
]

def _append_csv_row(path, fields, row_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in fields})

if __name__ == '__main__':
    cam = VideoCapture()
    pose = PoseEstimator()

    counter = SquatRepCounter(SquatRepCounterConfig(
        down_thresh=100,
        up_thresh=160,
        min_bottom_frames=MIN_BOTTOM_FRAMES,
        min_top_frames=MIN_TOP_FRAMES,
        min_rep_time=MIN_REP_TIME_SEC
    ))

    smooth_angle = None
    state = "UP"
    reps = 0

    current_rep_min_angle = float("inf")
    torso_tilts = []
    hip_knee_aligns = []
    heel_lift_max_norm = 0.0

    ankle_y_baseline = None
    ref_leg_len = None

    while cam.is_open():
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        res = pose.process(frame)

        knee_angle = None
        shoulder = hip = knee = ankle = None

        if res and res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            shoulder = lm_xy(lms[LEFT_SHOULDER], w, h)
            hip      = lm_xy(lms[LEFT_HIP],      w, h)
            knee     = lm_xy(lms[LEFT_KNEE],     w, h)
            ankle    = lm_xy(lms[LEFT_ANKLE],    w, h)

            if shoulder and hip and knee and ankle:
                knee_angle = angle_3pts(hip, knee, ankle)

        smooth_angle = ema(smooth_angle, knee_angle, alpha=0.3)
        display_angle = smooth_angle if smooth_angle is not None else knee_angle

        state, reps, just_counted = counter.update(angle=display_angle)

        if display_angle is not None:
            current_rep_min_angle = min(current_rep_min_angle, display_angle)

        if shoulder and hip:
            torso_tilts.append(torso_tilt_deg(shoulder, hip))

        if shoulder and hip and knee:
            hip_knee_aligns.append(hip_knee_alignment_deg(shoulder, hip, knee))

        if hip and ankle:
            if ref_leg_len is None:
                ref_leg_len = ((hip[0]-ankle[0])**2 + (hip[1]-ankle[1])**2) ** 0.5
            if ankle_y_baseline is None and state == "UP" and reps <= 1:
                ankle_y_baseline = ankle[1]
            if ankle_y_baseline is not None and ref_leg_len is not None:
                hlift = heel_lift_norm(ankle[1], ankle_y_baseline, ref_len_pix=ref_leg_len)
                heel_lift_max_norm = max(heel_lift_max_norm, hlift)

        if just_counted:
            torso_tilt_min  = round(min(torso_tilts), 2)  if torso_tilts else -1
            torso_tilt_max  = round(max(torso_tilts), 2)  if torso_tilts else -1
            torso_tilt_mean = round(sum(torso_tilts)/len(torso_tilts), 2) if torso_tilts else -1
            hip_knee_align_mean = round(sum(hip_knee_aligns)/len(hip_knee_aligns), 2) if hip_knee_aligns else -1
            heel_lift_max = round(heel_lift_max_norm, 3)

            _append_csv_row(REP_LOG_PATH, REP_FIELDS, {
                "min_knee_angle":       round(current_rep_min_angle if current_rep_min_angle != float("inf") else -1, 2),
                "torso_tilt_min":       torso_tilt_min,
                "torso_tilt_max":       torso_tilt_max,
                "torso_tilt_mean":      torso_tilt_mean,
                "hip_knee_align_mean_deg": hip_knee_align_mean,
                "heel_lift_max_norm":   heel_lift_max,
            })

            current_rep_min_angle = float("inf")
            torso_tilts.clear()
            hip_knee_aligns.clear()
            heel_lift_max_norm = 0.0

        # ---------- Overlay (same as live pipeline) ----------
        out = frame.copy()
        out = pose.draw(out, res)

        cv2.putText(out, "ESC: exit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if display_angle is not None:
            color = (0,255,0) if squat_depth_ok(display_angle, threshold=100) else (0,0,255)
            cv2.putText(out, f'Angle: {display_angle:.1f}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(out, f'Reps: {reps}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(out, f'State: {state}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow('Video', out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
