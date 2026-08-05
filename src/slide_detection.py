import numpy as np 
import pandas as pd 
from typing import List 
from .homography import transform_player_keypoints 
from scipy.signal import savgol_filter
from dataclasses import dataclass 
from .utils import get_center_of_box 
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


@dataclass
class SlideEvent:
    start_frame: int
    end_frame: int
    confidence: float
    peak_com_speed: float


class SlideDetection: 
    def __init__(self, window_size : int = 15, valid_window_ratio:float = .7,  min_gap_frames :int = 5 , stride : int = 1 ,DECEL_SMOOTHNESS_THRESH :float = .4, 
                 SLIDE_DISPLACEMENT_RATIO_THRESH : float = .5, STRIDE_LENGTH_BASELINE:float = .9, MIN_PLANT_FRAMES :int = 4 
                 , ANKLE_PLANT_VEL_THRESH :float = .8, fps : int = 30 ) : 
        self.min_gap_frames = min_gap_frames
        self.window_size = window_size
        self.stride = stride 
        self.fps  = fps 
        self.valid_window_ratio = valid_window_ratio
        self.DECEL_SMOOTHNESS_THRESH = DECEL_SMOOTHNESS_THRESH
        self.SLIDE_DISPLACEMENT_RATIO_THRESH = SLIDE_DISPLACEMENT_RATIO_THRESH
        self.ANKLE_PLANT_VEL_THRESH = ANKLE_PLANT_VEL_THRESH
        self.MIN_PLANT_FRAMES = MIN_PLANT_FRAMES
        self.STRIDE_LENGTH_BASELINE = STRIDE_LENGTH_BASELINE
        self.SCORE_ON_THRESH = 0.55
        self.SCORE_OFF_THRESH = 0.35
        self.MIN_EVENT_FRAMES = int(0.15 * fps)
    def _get_window(self , keypoints_annotations : List , last_window_center_index:int = None ) : 
        radius = int(self.window_size / 2) 
        if last_window_center_index is None or last_window_center_index < radius : 
            new_center_index =  radius
        else : 
            new_center_index = last_window_center_index + self.stride 
        window = keypoints_annotations[new_center_index - radius : (new_center_index + radius)+1] 

        return window , new_center_index 

    def _check_window(self, window  ):
        previous_frame = None 
        valid_frames = 0 
        valid_num = int(self.valid_window_ratio * len(window))

        for frame in window : 

            if not frame['player_pos'] or frame['player_pos'] is None or not frame['valid_keypoints']:
                continue
            frame_id = frame['frame_id']
            valid_frames += 1 
            if previous_frame is not None : 
                if frame_id - previous_frame > self.min_gap_frames : 
                    return False 

            previous_frame = frame_id 
        return valid_frames > valid_num 
     
    def _turn_window_into_df(self , keypoints_window , annotations ) : 
 
        keypoints_data = { 11 : 'left_hip' , 12 : 'right_hip' , 13 : 'left_knee' , 14 : 'right_knee' , 15 : 'left_ankle' , 16 :'right_ankle' , 17 : 'com'}  
        dataframe_rows = [] 
        annotation_map = {
            ann["frame_id"]: ann["court_points"]
            for ann in annotations
        }
        for frame in keypoints_window : 
            
            player_keypoints = frame['valid_keypoints']
            keypoints_indexes = [point['index'] for point in player_keypoints]
            keypoints_indexes.append(17)
            com_x , com_y = get_center_of_box(frame['player_pos'])
            keypoints_values = [point['xy'] for point in player_keypoints]
            keypoints_values.append([com_x , com_y])
            court_keypoints = annotation_map.get(frame["frame_id"])

            if keypoints_values and court_keypoints:
                projected_player_keypoints = transform_player_keypoints(court_keypoints , keypoints_values)
            flattened_row = []
            for index in keypoints_data.keys() : 
                if index in keypoints_indexes : 
                    point_index = keypoints_indexes.index(index) 

                    x, y = projected_player_keypoints[point_index] 
                    projected_point = [x, y]
                else : 
                    projected_point = [np.nan , np.nan ]
                flattened_row.extend(projected_point) 
            flattened_row.insert(0 , frame['frame_id'])
            dataframe_rows.append(flattened_row)

        columns = [ column  for name in keypoints_data.values() for column in (f"{name}_x", f"{name}_y") ]
        columns.insert(0 , 'frame_id')
        df = pd.DataFrame(dataframe_rows , columns = columns)
        return df 
    def _smooth_trajectory(self ,  df, window = 9, poly = 2) :
            """
            Fill small detection gaps then apply Savitzky-Golay smoothing.
            Why Savitzky-Golay?  It fits a polynomial locally, which preserves
            the sharp peak at a bounce better than a simple moving average.
            """
            # Reindex to dense frame range so gaps become NaN
            full_idx = pd.RangeIndex(df["frame_id"].min(), df["frame_id"].max() + 1)
            df = (
                df.set_index("frame_id")
                .reindex(full_idx)
                .rename_axis("frame_id")
                .reset_index()
            )
    
            # Savitzky-Golay needs at least window+1 non-NaN points
            for col in ['left_hip_x' , 'left_hip_y' , 'right_hip_x' , 'right_hip_y' , 'left_knee_x' , 'left_knee_y' , 'right_knee_x' , 'right_knee_y' , 'left_ankle_x' , 'left_ankle_y' , 'right_ankle_x' , 'right_ankle_y' , 'com_x' , 'com_y']: 
                valid = df[col].notna() 
                if valid.sum() > window:
                    df.loc[valid, f"{col}_smoothed"] = savgol_filter(
                        df.loc[valid, col], window_length=window, polyorder=poly
                    )
                else:
                    df[f"{col}_smoothed"] = df[col]
            return df 
            
    def _feature_engineering(self , df )  : 
        df[['left_hip_x' , 'left_hip_y' , 'right_hip_x' , 'right_hip_y' , 'left_knee_x' , 'left_knee_y' , 'right_knee_x' , 'right_knee_y' , 'left_ankle_x' , 'left_ankle_y' , 'right_ankle_x' , 'right_ankle_y' , 'com_x' , 'com_y']] = df[['left_hip_x' , 'left_hip_y' , 'right_hip_x' , 'right_hip_y' , 'left_knee_x' , 'left_knee_y' , 'right_knee_x' , 'right_knee_y' , 'left_ankle_x' , 'left_ankle_y' , 'right_ankle_x' , 'right_ankle_y', 'com_x' , 'com_y']].interpolate(limit_direction='both')
        df[['left_hip_x' , 'left_hip_y' , 'right_hip_x' , 'right_hip_y' , 'left_knee_x' , 'left_knee_y' , 'right_knee_x' , 'right_knee_y' , 'left_ankle_x' , 'left_ankle_y' , 'right_ankle_x' , 'right_ankle_y' ,  'com_x' , 'com_y']] = df[['left_hip_x' , 'left_hip_y' , 'right_hip_x' , 'right_hip_y' , 'left_knee_x' , 'left_knee_y' , 'right_knee_x' , 'right_knee_y' , 'left_ankle_x' , 'left_ankle_y' , 'right_ankle_x' , 'right_ankle_y' ,  'com_x' , 'com_y']].bfill()
        df = self._smooth_trajectory(df) 
        for col in ['com_x' , 'com_y' , 'left_hip_x' , 'left_hip_y' , 'right_hip_x' , 'right_hip_y' , 'left_knee_x' , 'left_knee_y' , 'right_knee_x' , 'right_knee_y' , 'left_ankle_x' , 'left_ankle_y' , 'right_ankle_x' , 'right_ankle_y']:
            smothed_col = f"{col}_smoothed"
            new_col  = f"{col}_smoothed_v"
            df[new_col] = np.gradient(df[smothed_col].values) * self.fps 
        for col in [ 'com','left_hip' , 'right_hip'  , 'left_knee'  , 'right_knee'  , 'left_ankle' , 'right_ankle']:
            x_v_col = f"{col}_x_smoothed_v"
            y_v_col = f"{col}_y_smoothed_v"
            new_col = f"{col}_smoothed_speed" 
            df[new_col] = np.linalg.norm(
                df[[x_v_col, y_v_col]],
                axis=1 )
        df['com_smoothed_acc'] = np.gradient(df['com_smoothed_speed'].values )* self.fps 
        
        return df
    def _extract_features(self , df ) : 
        l_ankle_spd = df['left_ankle_smoothed_speed'].values
        r_ankle_spd = df['right_ankle_smoothed_speed'].values
        l_ankle = df[['left_ankle_x_smoothed' , 'left_ankle_y_smoothed']].values
        r_ankle = df[['right_ankle_x_smoothed' , 'right_ankle_y_smoothed']].values
        l_knee = df[['left_knee_x_smoothed' , 'left_knee_y_smoothed']].values
        r_knee = df[['right_knee_x_smoothed' , 'right_knee_y_smoothed']].values
        l_hip = df[['left_hip_x_smoothed' , 'left_hip_y_smoothed']].values
        r_hip = df[['right_hip_x_smoothed' , 'right_hip_y_smoothed']].values
        com_xy = df[['com_x_smoothed' , 'com_y_smoothed']].values
        com_speed = df['com_smoothed_speed'].values
        com_acc = df['com_smoothed_acc'].values


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
                com_speed=com_speed,
                com_accel=com_acc,
                ankle_speed_min=ankle_speed_min,
                planted_ankle_xy=planted_ankle_xy,
                hip_y=hip_y,
                stance_width=stance_width,
                knee_flex_mean=knee_flex_mean,
            )
    def find_plant_windows(self , ankle_speed_min: np.ndarray) -> list[tuple[int, int]]:
        """Return list of (start, end) inclusive frame indices where ankle speed
        stays below ANKLE_PLANT_VEL_THRESH for at least MIN_PLANT_FRAMES."""
        planted = ankle_speed_min < self.ANKLE_PLANT_VEL_THRESH
        windows = []
        start = None
        for i, p in enumerate(planted):
            if p and start is None:
                start = i
            elif not p and start is not None:
                if i - start >= self.MIN_PLANT_FRAMES:
                    windows.append((start, i - 1))
                start = None
        if start is not None and len(planted) - start >= self.MIN_PLANT_FRAMES:
            windows.append((start, len(planted) - 1))
        return windows
            
    def deceleration_smoothness(self , com_speed_window: np.ndarray) -> float:
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
    def score_plant_window(self , feats: FeatureSeries, start: int, end: int) -> float:
        """Returns a 0-1 slide score for a single foot-plant window."""
        com_disp = np.linalg.norm(feats.com_xy[end] - feats.com_xy[start])
        disp_ratio = com_disp / self.STRIDE_LENGTH_BASELINE
        disp_score = np.clip(disp_ratio / self.SLIDE_DISPLACEMENT_RATIO_THRESH, 0, 1.5)
        disp_score = min(disp_score, 1.0)

        smoothness = self.deceleration_smoothness(feats.com_speed[start:end + 1])
        smoothness_score = np.clip(
            smoothness / self.DECEL_SMOOTHNESS_THRESH if self.DECEL_SMOOTHNESS_THRESH > 0 else 0,
            0, 1,
        )

        # wide stance + bent knee during the plant supports a slide/brace posture
        stance = np.mean(feats.stance_width[start:end + 1])
        knee_flex = np.mean(feats.knee_flex_mean[start:end + 1])
        posture_score = np.clip((stance / 0.6) * (1.2 - knee_flex), 0, 1)

        # weighted combination — displacement-during-plant is the dominant signal
        score = 0.55 * disp_score + 0.30 * smoothness_score + 0.15 * posture_score
        return float(np.clip(score, 0, 1))
    def per_frame_slide_score(self , feats: FeatureSeries, plant_windows: list[tuple[int, int]]) -> np.ndarray:
        T = len(feats.com_speed)
        scores = np.zeros(T)
        for (s, e) in plant_windows:
            w_score = self.score_plant_window(feats, s, e)
            scores[s:e + 1] = np.maximum(scores[s:e + 1], w_score)
        return scores
    def extract_slide_events(self, scores: np.ndarray, com_speed: np.ndarray , frames_ids : List) -> list[SlideEvent]:
        events = []
        in_event = False
        start_idx = None
        for i, s in enumerate(scores):
            if not in_event and s >= self.SCORE_ON_THRESH:
                in_event = True
                start_idx = i
            elif in_event and s < self.SCORE_OFF_THRESH:
                in_event = False
                end_idx = i - 1
                if end_idx - start_idx + 1 >= self.MIN_EVENT_FRAMES:
                    conf = float(np.mean(scores[start_idx:end_idx + 1]))
                    peak = float(np.max(com_speed[start_idx:end_idx + 1]))
                    start = frames_ids[start_idx]
                    end = frames_ids[end_idx]
                    events.append(SlideEvent(start, end, conf, peak))
                start_idx = None
        if in_event:
            end_idx = len(scores) - 1
            if end_idx - start_idx + 1 >= self.MIN_EVENT_FRAMES:
                conf = float(np.mean(scores[start_idx:end_idx + 1]))
                peak = float(np.max(com_speed[start_idx:end_idx + 1]))
                start = frames_ids[start_idx]
                end = frames_ids[end_idx]
                events.append(SlideEvent(start, end, conf, peak))
        return events

    def infer(self  , keypoints_annotations , annotations ) : 
        window_center = 0 
        video_events = []
        while window_center + int(self.window_size / 2 ) < len(keypoints_annotations): 

            window , new_center = self._get_window(keypoints_annotations , window_center) 
            near_player_window = [w[0] if w[0]['id'] == 'near' else w[1] for w in window]
            far_player_window = [w[0] if w[0]['id'] == 'far' else w[1] for w in window]
            players_windows = [near_player_window , far_player_window] 
            for player_window in players_windows :
                state = self._check_window(player_window)

                if state : 
                    df  = self._turn_window_into_df(player_window , annotations)
                    engineered = self._feature_engineering(df)
                    feats = self._extract_features(engineered)
                    plant_windows = self.find_plant_windows(feats.ankle_speed_min)
                    scores = self.per_frame_slide_score(feats, plant_windows)
                    events = self.extract_slide_events(scores, feats.com_speed , engineered.frame_id.values )
                    if events : 

                        video_events.append(events)
            window_center = new_center

        return video_events  
        
            
        