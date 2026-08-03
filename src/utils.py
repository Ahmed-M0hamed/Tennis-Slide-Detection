import cv2 
import json 
def write_video(  output_path , frames , fps  ) : 
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc,fps , (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames :
            out.write(frame)
        out.release()


def read_frames(path) : 
    frames = [] 
    cap = cv2.VideoCapture(path) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True : 
        ret , frame = cap.read()  
        if not ret : 
            break 
        frames.append(frame) 
    cap.release() 
    return frames  ,fps 

def read_jsonl(path) : 
        with open(path , 'rb') as f : 
                data = [json.loads(line) for line in f if line.strip()]
        return data 