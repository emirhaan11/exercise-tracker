import time
from dataclasses import dataclass


@dataclass()
class SquatRepCounterConfig:
    down_thresh: float = 90.0  # Maximum knee angle to be considered as down
    up_thresh: float = 150.0  # Minimum knee angle to be considered as up
    min_bottom_frames: int = 3  # Minimum number of frames to stay in the bottom position
    min_top_frames: int = 3  # Minimum number of frames to stay in the top position
    min_rep_time: float = 0.4  # Minimum time (seconds) between two reps to prevent bouncing


class SquatRepCounter:
    """
      - UP (at the top position)
      - DOWN (at the bottom position)
      - MID (in between)
    One rep: UP -> DOWN -> back to UP cycle is completed.
    """

    def __init__(self, cfg: SquatRepCounterConfig | None = None):
        self.cfg = cfg or SquatRepCounterConfig()
        self.state = "UP"  # Initial assumption
        self.count = 0
        self.top_frames = 0
        self.bottom_frames = 0
        self.last_rep_time = 0.0

    def reset(self):
        # Reset counter and state.
        self.state = "UP"
        self.count = 0
        self.top_frames = 0
        self.bottom_frames = 0
        self.last_rep_time = 0.0

    def update(self, angle: float | None, now: float | None = None):

        # Call this method on each frame.
        if angle is None:
            # If angle is missing, keep the current state, do not count reps.
            return self.state, self.count, False

        if now is None:
            now = time.time()

        just_counted = False
        cfg = self.cfg

        # Hysteresis logic
        if angle <= cfg.down_thresh:
            self.bottom_frames += 1
            self.top_frames = 0
            # If stayed in bottom long enough, lock to DOWN
            if self.state != "DOWN" and self.bottom_frames >= cfg.min_bottom_frames:
                self.state = "DOWN"

        elif angle >= cfg.up_thresh:
            self.top_frames += 1
            self.bottom_frames = 0
            # If stayed in top long enough, lock to UP
            if self.state != "UP" and self.top_frames >= cfg.min_top_frames:
                # If previous state was DOWN and enough time passed, count rep
                if self.state == "DOWN" and (now - self.last_rep_time) >= cfg.min_rep_time:
                    self.count += 1
                    self.last_rep_time = now
                    just_counted = True
                self.state = "UP"

        else:
            # Middle range
            self.top_frames = 0
            self.bottom_frames = 0

        return self.state, self.count, just_counted
