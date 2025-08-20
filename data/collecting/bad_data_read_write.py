import os
import csv
import cv2

from core.video_capture import VideoCapture
from core.pose import PoseEstimator
from core.rep_counter import SquatRepCounter, SquatRepCounterConfig
from core.features import (
    lm_xy, ema, angle_3pts,
    torso_tilt_deg, hip_knee_alignment_deg, heel_lift_norm,
    squat_depth_ok
)

# Landmarks (left side)
LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 23, 25, 27

# Debounce for offline video
MIN_REP_TIME_SEC   = 0.9
MIN_TOP_FRAMES     = 3
MIN_BOTTOM_FRAMES  = 3

# -------- NEW: Forced window logging when no full rep --------
FORCE_LOG_FRAMES   = 45   # Write a line every 45 frames (about 1.5 sec @30fps)
MIN_VALID_FRAMES   = 10   # there should be at least as many valid frames in the window

# Output CSV 
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

def _flush_and_reset(window_buf, out_path):
    """
    window_buf = dict with keys:
      'knee_mins', 'torso_tilts', 'hip_knee', 'heel_max' (float), 'valid_frames' (int)
    Writes a row if enough valid frames, then resets buffers.
    """
    if window_buf["valid_frames"] >= MIN_VALID_FRAMES:
        # Aggregate window metrics
        if window_buf["knee_mins"]:
            min_knee_angle = round(min(window_buf["knee_mins"]), 2)
        else:
            min_knee_angle = -1

        ts = window_buf["torso_tilts"]
        torso_tilt_min  = round(min(ts), 2)  if ts else -1
        torso_tilt_max  = round(max(ts), 2)  if ts else -1
        torso_tilt_mean = round(sum(ts)/len(ts), 2) if ts else -1

        hk = window_buf["hip_knee"]
        hip_knee_align_mean = round(sum(hk)/len(hk), 2) if hk else -1

        heel_lift_max = round(window_buf["heel_max"], 3)

        _append_csv_row(out_path, REP_FIELDS, {
            "min_knee_angle":           min_knee_angle,
            "torso_tilt_min":           torso_tilt_min,
            "torso_tilt_max":           torso_tilt_max,
            "torso_tilt_mean":          torso_tilt_mean,
            "hip_knee_align_mean_deg":  hip_knee_align_mean,
            "heel_lift_max_norm":       heel_lift_max,
        })

    # reset
    window_buf["knee_mins"].clear()
    window_buf["torso_tilts"].clear()
    window_buf["hip_knee"].clear()
    window_buf["heel_max"] = 0.0
    window_buf["valid_frames"] = 0
    window_buf["frames_in_window"] = 0

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

    # --- Rep accumulators (for real reps) ---
    current_rep_min_angle = float("inf")
    torso_tilts = []
    hip_knee_aligns = []
    heel_lift_max_norm = 0.0

    # --- Heel-lift baseline ---
    ankle_y_baseline = None
    ref_leg_len = None

    # --- NEW: window buffer for forced logging ---
    window_buf = {
        "knee_mins": [],
        "torso_tilts": [],
        "hip_knee": [],
        "heel_max": 0.0,
        "valid_frames": 0,
        "frames_in_window": 0,
    }

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

        # Smooth & state update
        smooth_angle = ema(smooth_angle, knee_angle, alpha=0.3)
        display_angle = smooth_angle if smooth_angle is not None else knee_angle
        state, reps, just_counted = counter.update(angle=display_angle)

        # ---------- accumulate (rep + window) ----------
        if display_angle is not None:
            # rep
            current_rep_min_angle = min(current_rep_min_angle, display_angle)
            # window
            window_buf["knee_mins"].append(display_angle)

        if shoulder and hip:
            ttilt = torso_tilt_deg(shoulder, hip)
            torso_tilts.append(ttilt)
            window_buf["torso_tilts"].append(ttilt)

        if shoulder and hip and knee:
            hk = hip_knee_alignment_deg(shoulder, hip, knee)
            hip_knee_aligns.append(hk)
            window_buf["hip_knee"].append(hk)

        if hip and ankle:
            if ref_leg_len is None:
                ref_leg_len = ((hip[0]-ankle[0])**2 + (hip[1]-ankle[1])**2) ** 0.5
            if ankle_y_baseline is None and state == "UP" and reps <= 1:
                ankle_y_baseline = ankle[1]
            if ankle_y_baseline is not None and ref_leg_len is not None:
                hlift = heel_lift_norm(ankle[1], ankle_y_baseline, ref_len_pix=ref_leg_len)
                heel_lift_max_norm = max(heel_lift_max_norm, hlift)
                window_buf["heel_max"] = max(window_buf["heel_max"], hlift)

        if res and res.pose_landmarks and shoulder and hip:
            window_buf["valid_frames"] += 1
        window_buf["frames_in_window"] += 1

        # ---------- finalize on real rep ----------
        if just_counted:
            torso_tilt_min  = round(min(torso_tilts), 2)  if torso_tilts else -1
            torso_tilt_max  = round(max(torso_tilts), 2)  if torso_tilts else -1
            torso_tilt_mean = round(sum(torso_tilts)/len(torso_tilts), 2) if torso_tilts else -1
            hip_knee_align_mean = round(sum(hip_knee_aligns)/len(hip_knee_aligns), 2) if hip_knee_aligns else -1
            heel_lift_max = round(heel_lift_max_norm, 3)

            _append_csv_row(REP_LOG_PATH, REP_FIELDS, {
                "min_knee_angle":          round(current_rep_min_angle if current_rep_min_angle != float("inf") else -1, 2),
                "torso_tilt_min":          torso_tilt_min,
                "torso_tilt_max":          torso_tilt_max,
                "torso_tilt_mean":         torso_tilt_mean,
                "hip_knee_align_mean_deg": hip_knee_align_mean,
                "heel_lift_max_norm":      heel_lift_max,
            })

            # reset rep accs
            current_rep_min_angle = float("inf")
            torso_tilts.clear()
            hip_knee_aligns.clear()
            heel_lift_max_norm = 0.0

            _flush_and_reset(window_buf, REP_LOG_PATH)

        # ---------- forced window log (no rep) ----------
        elif window_buf["frames_in_window"] >= FORCE_LOG_FRAMES:
            _flush_and_reset(window_buf, REP_LOG_PATH)

        # ---------- overlay ----------
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

    _flush_and_reset(window_buf, REP_LOG_PATH)

    cam.release()
    cv2.destroyAllWindows()

