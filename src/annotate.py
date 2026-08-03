import numpy as np 
import matplotlib as plt 
import cv2 
def draw_keypoint(frame ,keypoint_racket_annotation   ) : 
    pairs = [(14 , 16) , (13,15) , (12 ,14) , (11,13) , (6,12) , (5,11) 
         ,(5,6) , (6,8) , (5,7) , (8,10 ) , (7,9) , (11,12)] 
    for detection in keypoint_racket_annotation :
        if detection['racket'] != None :  
            xr1 , yr1 , xr2 , yr2 = detection['racket'] 
            cv2.rectangle(frame , (int(xr1) , int(yr1) ) , (int(xr2), int(yr2)) , (255, 0 , 0 ) , 2) 
        if detection['valid_keypoints'] :     
            for point in detection['valid_keypoints' ] : 
                x, y = point['xy'] 
                cv2.circle(frame, center=(int(x), int(y) ) , radius=2 , color = (0 , 0 ,255) , thickness=1 , lineType= 1) 
            for pair in pairs : 
                points = [point['xy'] for point in detection['valid_keypoints' ] for p in pair if point['index'] == p  ]
                if len(points) == 2 : 
                    p1 , p2 = points 
                    cv2.line(frame , (int(p1[0]) , int(p1[1]) ) , (int(p2[0]) , int(p2[1]) )  , (255,0,0) ,1  )



def visualize(frames , annotations , text_args , keypoints_annotations , shots_type , events ) : 
    for annotation in annotations: 
        frame_id  = annotation['frame_id']
        cv2.putText(frames[frame_id] , f'frame: {frame_id}' , (100 ,100) ,*text_args)  
        if 'court_points' in annotation :
            for  i , point in enumerate(annotation['court_points'] ) : 
                point_x , point_y = point 
                cv2.circle(frames[frame_id] , (point_x , point_y) , 5 , (0,0,255) , 2 , 1 )
                cv2.putText(frames[frame_id] , f'p_{i}' , (point_x , point_y - 5) , *text_args)
        if 'persons' in annotation  and annotation['persons']  : 
            for person in annotation['persons'] : 
                x1 , y1 ,x2 ,y2 = person['xyxy'] 
                cv2.rectangle(frames[frame_id] , (x1 ,y1 ) , (x2, y2) , (255,0,0) , 2)  
        if 'ball_position' in annotation : 
            x , y = annotation['ball_position']
            cv2.circle(frames[frame_id] , (x, y ) , 5 , (0,0,0), 2 , 1) 
        if any(event in annotation for event in events) : 
            detected_events = list(set(events).intersection(annotation.keys())) 
            cv2.putText(frames[frame_id] , f"events :{detected_events}" , (100 ,150 ) , *text_args) 
        if any(shot['frame'] == frame_id and shot['shot'] != None for shot in shots_type ) : 
            shot_type = [shot_type for shot_type in shots_type if shot_type['frame'] == frame_id][0]
            shot_classification = [shot_type["shot"]] 
            if shot_type['chop_detection'] == 'chop' : 
                shot_classification.append(shot_type['chop_detection'])
            cv2.putText(frames[frame_id] , f'Shot: {" & ".join(shot_classification)}' , (100 ,200 ) , *text_args)
        if any(keypoint_annotation[0]['frame_id'] == frame_id for keypoint_annotation in keypoints_annotations) : 
            keypoints_annotation = [keypoints_annotation for keypoints_annotation in keypoints_annotations if keypoints_annotation[0]['frame_id'] == frame_id][0] 
            draw_keypoint(frames[frame_id] ,keypoints_annotation  ) 
            
    return frames 