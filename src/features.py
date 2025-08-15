import numpy as np


def angle_3pts(a, b, c):
    # a, b, c: (x,y) pixel coordinates
    a, b, c = np.array(a), np.array(b), np.array(c)

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


