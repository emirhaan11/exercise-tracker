import time
import cv2
from pathlib import Path
from capture import CamCapture
from pose import PoseEstimator
from rep_counter import SquatRepCounter, SquatRepCounterConfig
from features import (
    lm_xy, ema, angle_3pts,
    torso_tilt_deg, hip_knee_alignment_deg, heel_lift_norm,
    squat_depth_ok
)
import joblib

# ---------------- Model Loader ----------------
CLF = None
THRESHOLD = 0.5


def _try_load_joblib(p: Path):
    try:
        m = joblib.load(str(p))
        print(f"[MODEL] Loaded: {p}")
        return m
    except Exception as e:
        print(f"[MODEL] Failed: {p} -> {e}")
        return None


def load_model():
    HERE = Path(__file__).resolve().parent
    p = HERE.parent / "models" / "squat_clf.pkl"
    if p.exists():
        m = _try_load_joblib(p)
        if m is not None:
            return m
    print("[MODEL] Not found. Tried:")


_model = load_model()


class _DirectClf:
    FEATURES = [
        "min_knee_angle", "torso_tilt_min", "torso_tilt_max", "torso_tilt_mean",
        "hip_knee_align_mean_deg", "heel_lift_max_norm"
    ]

    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, feat_dict):
        import numpy as np
        x = np.array([[float(feat_dict.get(k, -1.0)) for k in self.FEATURES]], dtype=float)
        if hasattr(self.model, "predict_proba"):
            p = float(self.model.predict_proba(x)[0, 1])
            y = int(p >= self.threshold)
        else:
            y = int(self.model.predict(x)[0])
            p = None
        return y, p


CLF = _DirectClf(_model, threshold=THRESHOLD) if _model is not None else None
print("[MODEL] CLF is", "READY" if CLF is not None else "NONE", "| CWD:", Path.cwd())

# --------------- Landmarks & Counter ---------------
LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 23, 25, 27
MIN_REP_TIME_SEC = 0.5
MIN_TOP_FRAMES = 3
MIN_BOTTOM_FRAMES = 3

if __name__ == '__main__':
    cam = CamCapture()
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
    heel_lift_max_norm_val = 0.0

    ankle_y_baseline = None
    ref_leg_len = None

    last_pred_label = None
    last_pred_proba = None
    show_pred_until = 0.0

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
            hip = lm_xy(lms[LEFT_HIP], w, h)
            knee = lm_xy(lms[LEFT_KNEE], w, h)
            ankle = lm_xy(lms[LEFT_ANKLE], w, h)

            if shoulder and hip and knee and ankle:
                knee_angle = angle_3pts(hip, knee, ankle)

        smooth_angle = ema(smooth_angle, knee_angle, alpha=0.3)
        display_angle = smooth_angle if smooth_angle is not None else knee_angle

        state, reps, just_counted = counter.update(angle=display_angle, now=time.monotonic())

        if display_angle is not None:
            current_rep_min_angle = min(current_rep_min_angle, display_angle)

        if shoulder and hip:
            ttilt = torso_tilt_deg(shoulder, hip)
            torso_tilts.append(ttilt)

        if shoulder and hip and knee:
            hk = hip_knee_alignment_deg(shoulder, hip, knee)
            hip_knee_aligns.append(hk)

        if hip and ankle:
            if ref_leg_len is None:
                ref_leg_len = ((hip[0] - ankle[0]) ** 2 + (hip[1] - ankle[1]) ** 2) ** 0.5
            if ankle_y_baseline is None and state == "UP" and reps <= 1:
                ankle_y_baseline = ankle[1]
            if ankle_y_baseline is not None and ref_leg_len is not None:
                hlift = heel_lift_norm(ankle[1], ankle_y_baseline, ref_len_pix=ref_leg_len)
                heel_lift_max_norm_val = max(heel_lift_max_norm_val, hlift)

        if just_counted and CLF is not None:
            torso_tilt_min = round(min(torso_tilts), 2) if torso_tilts else -1
            torso_tilt_max = round(max(torso_tilts), 2) if torso_tilts else -1
            torso_tilt_mean = round(sum(torso_tilts) / len(torso_tilts), 2) if torso_tilts else -1
            hip_knee_align_mean = round(sum(hip_knee_aligns) / len(hip_knee_aligns), 2) if hip_knee_aligns else -1
            heel_lift_max = round(heel_lift_max_norm_val, 3)

            feat = {
                "min_knee_angle": round(current_rep_min_angle if current_rep_min_angle != float("inf") else -1, 2),
                "torso_tilt_min": torso_tilt_min,
                "torso_tilt_max": torso_tilt_max,
                "torso_tilt_mean": torso_tilt_mean,
                "hip_knee_align_mean_deg": hip_knee_align_mean,
                "heel_lift_max_norm": heel_lift_max,
            }
            try:
                last_pred_label, last_pred_proba = CLF.predict(feat)
                show_pred_until = max(show_pred_until, time.monotonic() + 2.0)
                if last_pred_proba is not None:
                    if last_pred_proba >= 0.7:
                        final_label = 1  # GOOD
                    else:
                        final_label = 0  # BAD
                else:
                    final_label = last_pred_label
            except Exception as e:
                print("Prediction error:", e)
                last_pred_label, last_pred_proba = None, None
                show_pred_until = 0.0

            current_rep_min_angle = float("inf")
            torso_tilts.clear()
            hip_knee_aligns.clear()
            heel_lift_max_norm_val = 0.0

        # ---- Overlay ----
        out = frame.copy()
        out = pose.draw(out, res)

        cv2.putText(out, "ESC: exit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if display_angle is not None:
            color = (0, 255, 0) if squat_depth_ok(display_angle, threshold=100) else (0, 0, 255)
            cv2.putText(out, f'Angle: {display_angle:.1f}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(out, f'Reps: {reps}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, f'State: {state}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if CLF is not None and time.monotonic() < show_pred_until and last_pred_label is not None:
            txt = "GOOD" if final_label == 1 else "BAD"
            print(f'rep: {reps} ({txt})')
            col = (0, 255, 0) if final_label == 1 else (0, 0, 255)
            cv2.putText(out, f"Form: {txt} (p={last_pred_proba:.2f})", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)

        cv2.imshow('Camera', out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
