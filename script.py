import os
import cv2
import json
import numpy as np
from darwin.client import Client
from config import V7_KEY

# Set your folder paths
video_folder = "/Users/anagireddygari/.darwin/datasets/honors/eos/images"
annotations_folder = "/Users/anagireddygari/.darwin/datasets/honors/eos/releases/test/annotations"

def load_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
        data = data['annotations'][0]['frames']

    frame_annotations = {}
    for frame_idx, info in data.items():
        bbox_data = info['bounding_box']
        bbox = {
            'bbox': [
                bbox_data['x'],
                bbox_data['y'],
                bbox_data['w'],
                bbox_data['h']
            ],
            'label': 'Object'
        }
        frame_annotations.setdefault(frame_idx, []).append(bbox)

    return frame_annotations

def analyze_roi_with_mask(frame, mask, annotation):
    x, y, w, h = map(int, annotation['bbox'])
    roi_mask = mask[y:y+h, x:x+w]
    edges = cv2.Canny(roi_mask, 100, 200)
    frame[y:y+h, x:x+w] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return frame

def process_one_frame(video_path, annotation_path):
    import os
import cv2
import json
import numpy as np
from darwin.client import Client
from config import V7_KEY

# Set your folder paths
video_folder = "/Users/anagireddygari/.darwin/datasets/honors/eos/images"
annotations_folder = "/Users/anagireddygari/.darwin/datasets/honors/eos/releases/test/annotations"

def load_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
        data = data['annotations'][0]['frames']

    frame_annotations = {}
    for frame_idx, info in data.items():
        bbox_data = info['bounding_box']
        bbox = {
            'bbox': [
                bbox_data['x'],
                bbox_data['y'],
                bbox_data['w'],
                bbox_data['h']
            ],
            'label': 'Object'
        }
        frame_annotations.setdefault(frame_idx, []).append(bbox)

    return frame_annotations

def analyze_roi_with_mask(frame, mask, annotation):
    x, y, w, h = map(int, annotation['bbox'])
    roi_mask = mask[y:y+h, x:x+w]
    edges = cv2.Canny(roi_mask, 100, 200)
    frame[y:y+h, x:x+w] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return frame

def preprocess(video_path, annotation_path):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorKNN()

    frame_idx = 0
    while True:
        if frame_idx < 900:
            frame_idx += 1
            continue
        
        ret, frame = cap.read()
        if frame is None:
            break
            
        
        fgmask = fgbg.apply(frame)
        # cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        # cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
        #     cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        print(frame_idx)
        
        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgmask)


        keyboard = cv2.waitKey(30)
        frame_idx += 1
        if keyboard == 'q' or keyboard == 27:
            break




def main():
    video_files = [f for f in os.listdir(video_folder) if f.endswith('GX010159.MP4')]
  

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_folder, video_file)
        annotation_path = os.path.join(annotations_folder, f"{video_name}.json")

        if os.path.exists(annotation_path):
            print(f"➡️ Processing single frame from {video_file}")
            preprocess(video_path, annotation_path)
        else:
            print(f"⚠️ No matching annotation file for {video_file}, skipping.")

if __name__ == "__main__":
    main()
