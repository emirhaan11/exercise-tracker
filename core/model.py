import joblib
import numpy as np
from pathlib import Path

FEATURES = [
    "min_knee_angle","torso_tilt_min","torso_tilt_max","torso_tilt_mean",
    "hip_knee_align_mean_deg","heel_lift_max_norm"
]

main_path = Path(__file__)

class SquatClassifier:
    def __init__(self, model_path=str(main_path.parent / 'models' / 'squat_clf.pkl'), threshold=0.5):
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def predict(self, feat_dict):
        # create vector
        x = []
        for k in FEATURES:
            v = feat_dict.get(k, -1.0)
            if v is None:
                v = -1.0
            x.append(float(v))
        X = np.array([x], dtype=float)

        # predict
        if hasattr(self.model, "predict_proba"):
            p = float(self.model.predict_proba(X)[0,1])
            y = int(p >= self.threshold)
        else:
            y = int(self.model.predict(X)[0])
            p = None
        return y, p
