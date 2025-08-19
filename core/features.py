import numpy as np
from math import degrees, acos

def angle_3pts(a, b, c):
    # a, b, c: (x,y) pixel coordinates
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)

    v1 = a - b
    v2 = c - b

    # Calculate the degree with linear algebra
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    cosang = np.clip(cosang, -1.0, 1.0)

    # Convert the radian to degree and return it
    return np.degrees(np.arccos(cosang))


def lm_xy(landmark, width, height):  # pose result -> (x,y) pixsel
    return int(landmark.x * width), int(landmark.y * height)


def ema(prev_angle, curr_angle, alpha=0.3):  # Exponential Moving Are
    if curr_angle is None:
        return prev_angle
    if prev_angle is None:
        return curr_angle
    return alpha * curr_angle + (1 - alpha) * prev_angle


def squat_depth_ok(knee_angle_deg, threshold = 100) -> bool:
    if knee_angle_deg is None:
        return False
    return knee_angle_deg <= threshold


# FOR FORMS
def torso_tilt_deg(shoulder, hip):
    sx, sy = shoulder
    hx, hy = hip
    vx, vy = hx - sx, hy - sy
    norm = np.hypot(vx, vy) + 1e-8
    dot = (vx * 0) + (vy * -1)  # vertical up
    cosang = np.clip(dot / norm, -1.0, 1.0)
    return degrees(acos(cosang))

def hip_knee_alignment_deg(shoulder, hip, knee):
    v1 = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]], dtype=float)
    v2 = np.array([knee[0] - hip[0],     knee[1] - hip[1]],     dtype=float)
    n1 = np.linalg.norm(v1) + 1e-8
    n2 = np.linalg.norm(v2) + 1e-8
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return degrees(np.arccos(cosang))

def heel_lift_norm(current_ankle_y, baseline_ankle_y, ref_len_pix):
    if ref_len_pix is None or ref_len_pix <= 1e-6:
        return 0.0
    if baseline_ankle_y is None or current_ankle_y is None:
        return 0.0
    delta_up = float(baseline_ankle_y - current_ankle_y)  # smaller y = higher (upwards)
    return delta_up / ref_len_pix




