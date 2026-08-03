"""
Tennis Player Slide Detection — Heuristic / State-Machine Pipeline (Option A)
==============================================================================

Detects sliding events from:
  - player COM (x, y) court coordinates, per frame
  - pose keypoints (COCO-17 style: ankles, knees, hips), per frame
  - racket position (optional, used only to corroborate / filter)

Pipeline stages:
  1. Load & smooth raw signals (Savitzky-Golay)
  2. Compute kinematic features (COM velocity/accel, ankle-hip relative
     velocity, stance width, knee flexion, hip vertical oscillation)
  3. Detect foot-plant windows (ankle velocity below threshold)
  4. Score each plant window for "slide-ness"
     (COM keeps moving while foot is planted, deceleration is smooth
     rather than stepped)
  5. Run a small hysteresis state machine over per-frame slide scores
     to get clean, debounced slide events (onset/offset frame + confidence)

No labeled data required. This is meant to be a strong, interpretable
baseline you can either ship directly or use to auto-label data for
a downstream sequence model (option B).

Author: ML pipeline sketch — tune all THRESH_* constants to your court
        coordinate units, frame rate, and camera calibration.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.signal import savgol_filter
from typing import Optional


# ----------------------------------------------------------------------------
# 0. Config — tune these against your data (units: meters & seconds, ideally
#    after homography to real-world court coordinates, NOT raw pixels)
# ----------------------------------------------------------------------------

FPS = 30.0

# Smoothing
SG_WINDOW = 9          # must be odd; ~0.3s at 30fps
SG_POLY = 2

# Foot-plant detection
ANKLE_PLANT_VEL_THRESH = 0.35     # m/s — ankle considered "planted" below this
MIN_PLANT_FRAMES = 4              # minimum consecutive frames to count as a plant

# Slide scoring
STRIDE_LENGTH_BASELINE = 0.9      # m — typical single-step COM displacement,
                                   # calibrate per player if possible
SLIDE_DISPLACEMENT_RATIO_THRESH = 0.5   # com_disp / stride_baseline during plant
DECEL_SMOOTHNESS_THRESH = 0.6     # 0-1, higher = smoother/more slide-like

# State machine hysteresis
SCORE_ON_THRESH = 0.55
SCORE_OFF_THRESH = 0.35
MIN_EVENT_FRAMES = int(0.15 * FPS)   # ignore blips shorter than ~150ms


# ----------------------------------------------------------------------------
# 1. Data containers
# ----------------------------------------------------------------------------

@dataclass
class FrameData:
    frame_idx: int
    com_xy: np.ndarray                 # shape (2,) court coords of player center/hip
    left_ankle: np.ndarray             # (2,)
    right_ankle: np.ndarray            # (2,)
    left_knee: np.ndarray              # (2,)
    right_knee: np.ndarray             # (2,)
    left_hip: np.ndarray               # (2,)
    right_hip: np.ndarray              # (2,)
    racket_xy: Optional[np.ndarray] = None
    racket_speed: Optional[float] = None   # optional precomputed racket speed


@dataclass
class SlideEvent:
    start_frame: int
    end_frame: int
    confidence: float
    peak_com_speed: float

    @property
    def duration_s(self) -> float:
        return (self.end_frame - self.start_frame) / FPS


# ----------------------------------------------------------------------------
# 2. Smoothing utilities
# ----------------------------------------------------------------------------

def smooth_series(arr: np.ndarray, window: int = SG_WINDOW, poly: int = SG_POLY) -> np.ndarray:
    """Savitzky-Golay smoothing along the time axis. arr shape: (T,) or (T, D)."""
    n = arr.shape[0]
    w = min(window, n - (1 - n % 2))  # keep window odd and <= n
    if w < poly + 2:
        return arr.copy()
    if w % 2 == 0:
        w -= 1
    if arr.ndim == 1:
        return savgol_filter(arr, w, poly)
    return np.stack([savgol_filter(arr[:, d], w, poly) for d in range(arr.shape[1])], axis=1)


def velocity(series: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Central-difference velocity. series: (T, D) or (T,) -> same shape."""
    v = np.gradient(series, axis=0) * fps
    return v


def speed(series: np.ndarray, fps: float = FPS) -> np.ndarray:
    v = velocity(series, fps)
    if v.ndim == 1:
        return np.abs(v)
    return np.linalg.norm(v, axis=1)


# ----------------------------------------------------------------------------
# 3. Feature extraction
# ----------------------------------------------------------------------------

@dataclass
class FeatureSeries:
    com_xy: np.ndarray
    com_speed: np.ndarray
    com_accel: np.ndarray
    ankle_speed_min: np.ndarray       # min(left_ankle_speed, right_ankle_speed) per frame
    planted_ankle_xy: np.ndarray      # position of whichever ankle is more stationary
    hip_y: np.ndarray                 # vertical hip position (for bounce/oscillation)
    stance_width: np.ndarray
    knee_flex_mean: np.ndarray        # proxy: hip-knee-ankle "openness", lower = more bent


def extract_features(frames: list[FrameData]) -> FeatureSeries:
    T = len(frames)

    com_xy = smooth_series(np.stack([f.com_xy for f in frames]))
    l_ankle = smooth_series(np.stack([f.left_ankle for f in frames]))
    r_ankle = smooth_series(np.stack([f.right_ankle for f in frames]))
    l_knee = smooth_series(np.stack([f.left_knee for f in frames]))
    r_knee = smooth_series(np.stack([f.right_knee for f in frames]))
    l_hip = smooth_series(np.stack([f.left_hip for f in frames]))
    r_hip = smooth_series(np.stack([f.right_hip for f in frames]))

    com_spd = speed(com_xy)
    com_acc = np.gradient(com_spd) * FPS

    l_ankle_spd = speed(l_ankle)
    r_ankle_spd = speed(r_ankle)

    ankle_speed_min = np.minimum(l_ankle_spd, r_ankle_spd)
    # whichever ankle is slower this frame is treated as the "planted" one
    planted_is_left = l_ankle_spd <= r_ankle_spd
    planted_ankle_xy = np.where(planted_is_left[:, None], l_ankle, r_ankle)

    hip_mid = (l_hip + r_hip) / 2.0
    hip_y = hip_mid[:, 1]

    stance_width = np.linalg.norm(l_ankle - r_ankle, axis=1)

    # crude knee-flexion proxy: vertical distance hip->knee vs knee->ankle
    # (smaller ratio ~ more bent knee; replace with real joint angle if you
    # have 3D pose / more keypoints)
    l_thigh = np.linalg.norm(l_hip - l_knee, axis=1)
    l_shank = np.linalg.norm(l_knee - l_ankle, axis=1)
    r_thigh = np.linalg.norm(r_hip - r_knee, axis=1)
    r_shank = np.linalg.norm(r_knee - r_ankle, axis=1)
    eps = 1e-6
    knee_flex_mean = ((l_shank / (l_thigh + eps)) + (r_shank / (r_thigh + eps))) / 2.0

    return FeatureSeries(
        com_xy=com_xy,
        com_speed=com_spd,
        com_accel=com_acc,
        ankle_speed_min=ankle_speed_min,
        planted_ankle_xy=planted_ankle_xy,
        hip_y=hip_y,
        stance_width=stance_width,
        knee_flex_mean=knee_flex_mean,
    )


# ----------------------------------------------------------------------------
# 4. Foot-plant window detection
# ----------------------------------------------------------------------------

def find_plant_windows(ankle_speed_min: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) inclusive frame indices where ankle speed
    stays below ANKLE_PLANT_VEL_THRESH for at least MIN_PLANT_FRAMES."""
    planted = ankle_speed_min < ANKLE_PLANT_VEL_THRESH
    windows = []
    start = None
    for i, p in enumerate(planted):
        if p and start is None:
            start = i
        elif not p and start is not None:
            if i - start >= MIN_PLANT_FRAMES:
                windows.append((start, i - 1))
            start = None
    if start is not None and len(planted) - start >= MIN_PLANT_FRAMES:
        windows.append((start, len(planted) - 1))
    return windows


# ----------------------------------------------------------------------------
# 5. Slide scoring within plant windows
# ----------------------------------------------------------------------------

def deceleration_smoothness(com_speed_window: np.ndarray) -> float:
    """
    Sliding -> smooth, monotonic-ish deceleration curve.
    Stepping -> stair-stepped speed profile (periodic re-acceleration each
    stride) even while a single foot looks "planted" between frames.

    We approximate smoothness as 1 - (variance of second derivative,
    normalized) -- a proxy for how "single smooth decay" vs "jagged/bouncy"
    the speed curve is over the window.
    """
    if len(com_speed_window) < 4:
        return 0.0
    d2 = np.diff(com_speed_window, n=2)
    jaggedness = np.std(d2)
    scale = np.std(com_speed_window) + 1e-6
    smoothness = 1.0 - np.clip(jaggedness / scale, 0, 1)
    return float(smoothness)


def score_plant_window(feats: FeatureSeries, start: int, end: int) -> float:
    """Returns a 0-1 slide score for a single foot-plant window."""
    com_disp = np.linalg.norm(feats.com_xy[end] - feats.com_xy[start])
    disp_ratio = com_disp / STRIDE_LENGTH_BASELINE
    disp_score = np.clip(disp_ratio / SLIDE_DISPLACEMENT_RATIO_THRESH, 0, 1.5)
    disp_score = min(disp_score, 1.0)

    smoothness = deceleration_smoothness(feats.com_speed[start:end + 1])
    smoothness_score = np.clip(
        smoothness / DECEL_SMOOTHNESS_THRESH if DECEL_SMOOTHNESS_THRESH > 0 else 0,
        0, 1,
    )

    # wide stance + bent knee during the plant supports a slide/brace posture
    stance = np.mean(feats.stance_width[start:end + 1])
    knee_flex = np.mean(feats.knee_flex_mean[start:end + 1])
    posture_score = np.clip((stance / 0.6) * (1.2 - knee_flex), 0, 1)

    # weighted combination — displacement-during-plant is the dominant signal
    score = 0.55 * disp_score + 0.30 * smoothness_score + 0.15 * posture_score
    return float(np.clip(score, 0, 1))


def per_frame_slide_score(feats: FeatureSeries, plant_windows: list[tuple[int, int]]) -> np.ndarray:
    T = len(feats.com_speed)
    scores = np.zeros(T)
    for (s, e) in plant_windows:
        w_score = score_plant_window(feats, s, e)
        scores[s:e + 1] = np.maximum(scores[s:e + 1], w_score)
    return scores


# ----------------------------------------------------------------------------
# 6. Hysteresis state machine -> clean events
# ----------------------------------------------------------------------------

def extract_slide_events(scores: np.ndarray, com_speed: np.ndarray) -> list[SlideEvent]:
    events = []
    in_event = False
    start_idx = None
    for i, s in enumerate(scores):
        if not in_event and s >= SCORE_ON_THRESH:
            in_event = True
            start_idx = i
        elif in_event and s < SCORE_OFF_THRESH:
            in_event = False
            end_idx = i - 1
            if end_idx - start_idx + 1 >= MIN_EVENT_FRAMES:
                conf = float(np.mean(scores[start_idx:end_idx + 1]))
                peak = float(np.max(com_speed[start_idx:end_idx + 1]))
                events.append(SlideEvent(start_idx, end_idx, conf, peak))
            start_idx = None
    if in_event:
        end_idx = len(scores) - 1
        if end_idx - start_idx + 1 >= MIN_EVENT_FRAMES:
            conf = float(np.mean(scores[start_idx:end_idx + 1]))
            peak = float(np.max(com_speed[start_idx:end_idx + 1]))
            events.append(SlideEvent(start_idx, end_idx, conf, peak))
    return events


# ----------------------------------------------------------------------------
# 7. (Optional) racket-phase corroboration — filters/boosts confidence
#    for slides that co-occur with a stroke near contact
# ----------------------------------------------------------------------------

def corroborate_with_racket(events: list[SlideEvent], frames: list[FrameData],
                             boost: float = 0.1) -> list[SlideEvent]:
    boosted = []
    for ev in events:
        racket_speeds = [
            f.racket_speed for f in frames[ev.start_frame:ev.end_frame + 1]
            if f.racket_speed is not None
        ]
        conf = ev.confidence
        if racket_speeds and max(racket_speeds) > np.percentile(
            [f.racket_speed for f in frames if f.racket_speed is not None] or [0], 75
        ):
            conf = min(1.0, conf + boost)
        boosted.append(SlideEvent(ev.start_frame, ev.end_frame, conf, ev.peak_com_speed))
    return boosted


# ----------------------------------------------------------------------------
# 8. Top-level pipeline entry point
# ----------------------------------------------------------------------------

def detect_slides(frames: list[FrameData], use_racket: bool = True) -> list[SlideEvent]:
    feats = extract_features(frames)
    plant_windows = find_plant_windows(feats.ankle_speed_min)
    scores = per_frame_slide_score(feats, plant_windows)
    events = extract_slide_events(scores, feats.com_speed)
    if use_racket:
        events = corroborate_with_racket(events, frames)
    return events


# ----------------------------------------------------------------------------
# 9. Demo with synthetic data (replace with your real tracker/pose output)
# ----------------------------------------------------------------------------

def _make_synthetic_frames(n_frames: int = 90) -> list[FrameData]:
    """
    Builds a toy sequence:
      frames  0-39: normal running (feet plant in fixed WORLD position for
                    each ~10-frame stride, then jump forward to the next
                    plant spot -- COM barely moves once a foot is down)
      frames 40-55: slide -- lead foot plants at a fixed WORLD position and
                    STAYS there while COM keeps translating on top of it,
                    decelerating smoothly
      frames 56-89: standing still

    Purely for demonstrating the pipeline runs end-to-end; replace with your
    real tracker/pose output.
    """
    rng = np.random.default_rng(0)
    frames = []
    com = np.array([0.0, 0.0])
    vel = np.array([3.0, 0.0])  # running at 3 m/s

    stride_len = 10  # frames per stride during normal running
    plant_world_xy = com + np.array([0.35, -0.9])  # current planted-foot world pos

    for i in range(n_frames):
        if i < 40:
            # normal running: re-plant foot every `stride_len` frames at the
            # player's current position (foot stays fixed in world coords
            # for the stride, COM moves only a little while that foot is down)
            if i % stride_len == 0:
                plant_world_xy = com + np.array([0.35, -0.9])
            com = com + vel / FPS + rng.normal(0, 0.005, 2)
            stance_scale = 0.5
        elif i < 56:
            # slide phase: foot plants once at i==40 and stays fixed in
            # world coordinates for the whole slide while COM keeps sliding
            # over/past it, decelerating smoothly
            if i == 40:
                plant_world_xy = com + np.array([0.35, -0.9])
            vel = vel * 0.90
            com = com + vel / FPS + rng.normal(0, 0.005, 2)
            stance_scale = 1.0
        else:
            vel = np.array([0.0, 0.0])
            stance_scale = 0.4

        l_ankle = plant_world_xy + rng.normal(0, 0.01, 2)
        r_ankle = com + np.array([-0.30, -0.85]) * stance_scale + rng.normal(0, 0.01, 2)

        l_knee = com + np.array([0.1, -0.5]) * stance_scale
        r_knee = com + np.array([-0.1, -0.5]) * stance_scale
        l_hip = com + np.array([0.1, -0.1])
        r_hip = com + np.array([-0.1, -0.1])

        racket_xy = com + np.array([0.5, 0.2])
        racket_speed = np.linalg.norm(vel) * (1.5 if 45 <= i < 50 else 0.5)

        frames.append(FrameData(
            frame_idx=i, com_xy=com.copy(),
            left_ankle=l_ankle, right_ankle=r_ankle,
            left_knee=l_knee, right_knee=r_knee,
            left_hip=l_hip, right_hip=r_hip,
            racket_xy=racket_xy, racket_speed=racket_speed,
        ))
    return frames


if __name__ == "__main__":
    frames = _make_synthetic_frames()
    events = detect_slides(frames)
    print(f"Detected {len(events)} slide event(s):")
    for ev in events:
        print(
            f"  frames {ev.start_frame}-{ev.end_frame} "
            f"({ev.duration_s:.2f}s), confidence={ev.confidence:.2f}, "
            f"peak_com_speed={ev.peak_com_speed:.2f} m/s"
        )
