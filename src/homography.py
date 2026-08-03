import cv2 
from .court_tempelete import build_court_template 
import numpy as np 

def compute_homography(
    src_pts,
    dst_pts ,
    min_points = 4,
    ransac_thresh = 5.0
) :

    if src_pts is None or len(src_pts) < min_points:
        return None, None

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    return H, mask
def transform_player_keypoints(court_keypoints , player_keypoints ) : 
    court_template, kp_world = build_court_template()
    dtl , dtr , dbl , dbr , stl , sbl , str_ , sbr , svtl , svtr , svbl , svbr , svc_mid_t ,svc_mid_b = keypoints
    kp_frame = np.array([dtl , dtr ,dbr ,  dbl  , stl , str_ ,sbr,  sbl   , svtl ,
                              svtr , svbr, svbl , svc_mid_t ,svc_mid_b] , dtype=np.float32) 
    H, mask = compute_homography(kp_frame, kp_world) 
    players_keypoints = np.array(player_keypoints , dtype=np.float32) 
    if H is not None:
        players_proj = cv2.perspectiveTransform(
            players_keypoints.reshape(-1, 1, 2), H).reshape(-1, 2)
        return players_proj 
    else :
        return [[None , None ] for _ in range(len(player_keypoints))]
def transform_ball_players_court_position(keypoints , ball_positon , player_1_position , player_2_position) :
    #             kp_world = np.array([
    #         dtl, dtr, dbr, dbl,stl, str_,sbr,sbl,svtl ,svtr , svbr , svbl  , svc_mid_t , svc_mid_b 
    # ], dtype=np.float32)
    court_template, kp_world = build_court_template()
    dtl , dtr , dbl , dbr , stl , sbl , str_ , sbr , svtl , svtr , svbl , svbr , svc_mid_t ,svc_mid_b = keypoints
    kp_frame = np.array([dtl , dtr ,dbr ,  dbl  , stl , str_ ,sbr,  sbl   , svtl ,
                          svtr , svbr, svbl , svc_mid_t ,svc_mid_b] , dtype=np.float32) 
    H, mask = compute_homography(kp_frame, kp_world)
    ball = np.array([ball_positon] ,dtype=np.float32 )
    player_1 = np.array([player_1_position ] , dtype=np.float32)
    player_2 = np.array([player_2_position ] , dtype=np.float32)
    court_corners = np.array([dtl , dtr , dbl , dbr] , dtype=np.float32) 
    if H is not None:
        ball_proj = cv2.perspectiveTransform(
            ball.reshape(-1, 1, 2), H).reshape(-1, 2)
        player_1_proj = cv2.perspectiveTransform(
            player_1.reshape(-1, 1, 2), H).reshape(-1, 2)
        player_2_proj = cv2.perspectiveTransform(
                    player_2.reshape(-1, 1, 2), H).reshape(-1, 2)
        corners_proj = cv2.perspectiveTransform(
                    court_corners.reshape(-1, 1, 2), H).reshape(-1, 2)

        
        return ball_proj[0] , player_1_proj[0] , player_2_proj[0] , corners_proj
    else : 
        return [None , None ] , [None , None ],[None , None ],[[None , None ] ,[None , None ],[None , None ],[None , None ]]


