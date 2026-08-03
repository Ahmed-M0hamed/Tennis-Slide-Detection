import numpy as np 
import pandas as pd 
from typing import List 
from .homography import transform_player_keypoints 
from scipy.signal import savgol_filter
class SlideDetection: 
    def __init__(self, window_size : int = 15, min_gap_frames :int = 5 , stride : int = 1 , fps : int = 25 ) : 
        self.min_gap_frames = min_gap_frames
        self.window_size = window_size
        self.stride = stride 
        self.fps  = fps 
    def _get_window(self , keypoints_annotations : List , last_window_center_index:int = None ) : 
        radius = int(self.window_size / 2) 
        if last_window_center_index is None or last_window_center_index < radius : 
            new_center_index =  radius
        else : 
            new_center_index = last_window_center_index + self.stride 
        window = keypoints_annotations[new_center_index - radius : (new_center_index + radius)+1] 

        return window , new_center_index 

    def _check_window(self, window):
        gap = 1
        index = 0 

        while index < len(window) : 
            if window[index] is None or not window[index]['valid_keypoints']:
                gap += 1
                if gap > self.min_gap_frames:
                    return False
            else:
                gap = 1
            index +=1 
        return True
    def _turn_window_into_df(self , keypoints_window , annotations ) : 
 
        keypoints_data = {  5 : 'left_shoulder' , 6 : 'right_shoulder' , 7 : 'left_elbow' , 8 : 'right_elbow' , 9 : 'left_wrist' , 10 : 'right_wrist' , 11 : 'left_hip' , 12 : 'right_hip' , 13 : 'left_knee' , 14 : 'right_knee' , 15 : 'left_ankle' , 16 :'right_ankle' }  
        dataframe_rows = [] 
        annotation_map = {
            ann["frame_id"]: ann["court_points"]
            for ann in annotations
        }
        for frame in keypoints_window : 
            
            player_keypoints = frame['valid_keypoints']
            keypoints_indexes = [point['index'] for point in player_keypoints] 
            keypoints_values = [point['xy'] for point in player_keypoints]
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
                    projected_point = [None , None ]
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
            for col in ['left_hip_x' , 'left_hip_y' , 'right_hip_x' , 'right_hip_y' , 'left_knee_x' , 'left_knee_y' , 'right_knee_x' , 'right_knee_y' , 'left_ankle_x' , 'left_ankle_y' , 'right_ankle_x' , 'right_ankle_y']: 
                valid = df[col].notna() 
                if valid.sum() > window:
                    df.loc[valid, f"{col}_smoothed"] = savgol_filter(
                        df.loc[valid, col], window_length=window, polyorder=poly
                    )
                else:
                    df[f"{col}_smoothed"] = df[col]
            return df 
            
    def _feature_engineering(self , df )  : 
        df[['left_hip_x' , 'left_hip_y' , 'right_hip_x' , 'right_hip_y' , 'left_knee_x' , 'left_knee_y' , 'right_knee_x' , 'right_knee_y' , 'left_ankle_x' , 'left_ankle_y' , 'right_ankle_x' , 'right_ankle_y']] = df[['left_hip_x' , 'left_hip_y' , 'right_hip_x' , 'right_hip_y' , 'left_knee_x' , 'left_knee_y' , 'right_knee_x' , 'right_knee_y' , 'left_ankle_x' , 'left_ankle_y' , 'right_ankle_x' , 'right_ankle_y']].interpolate()
        df[['left_hip_x' , 'left_hip_y' , 'right_hip_x' , 'right_hip_y' , 'left_knee_x' , 'left_knee_y' , 'right_knee_x' , 'right_knee_y' , 'left_ankle_x' , 'left_ankle_y' , 'right_ankle_x' , 'right_ankle_y']] = df[['left_hip_x' , 'left_hip_y' , 'right_hip_x' , 'right_hip_y' , 'left_knee_x' , 'left_knee_y' , 'right_knee_x' , 'right_knee_y' , 'left_ankle_x' , 'left_ankle_y' , 'right_ankle_x' , 'right_ankle_y']].bfill()
        df = self._smooth_trajectory(df) 
        print(df.head())
        df['com_x_smoothed'] = (df['left_hip_x_smoothed'] + df['right_hip_x_smoothed']) / 2
        df['com_y_smoothed'] = (df['left_hip_y_smoothed'] + df['right_hip_y_smoothed']) / 2
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
    def test(self  , keypoints_annotations , annotations ) : 
        window_center = 0 


        while window_center + int(self.window_size / 2 ) < 20 : 

            window , new_center = self._get_window(keypoints_annotations , window_center) 
            near_player_window = [w[0] if w[0]['id'] == 'near' else w[1] for w in window]
            far_player_window = [w[0] if w[0]['id'] == 'far' else w[1] for w in window]
            players_windows = [near_player_window , far_player_window] 
            for player_window in players_windows :
                state = self._check_window(player_window)

                if state : 
                    df  = self._turn_window_into_df(player_window , annotations)
                    engineered = self._feature_engineering(df) 
            window_center = new_center
        return engineered  
            