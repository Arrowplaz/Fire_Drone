import os
import cv2
import json
import numpy as np
from darwin.client import Client
# from config import V7_KEY

# Set your folder paths
video_folder = "~/Fire_Drone"
annotations_folder = "~/"

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


def preprocess(video_path, output_path='output.mp4'):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorKNN()

    # Get frame width, height, and fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        

        fgmask = fgbg.apply(frame)

        # Apply the mask to the original frame
        fgmask_3ch = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)  # Make it 3 channels
        subtracted = cv2.bitwise_and(frame, fgmask_3ch)

        print(frame_idx)
        
        # Show the background subtracted frame
        cv2.imshow('Background Subtracted', subtracted)

        # Save the subtracted frame
        out.write(subtracted)

        keyboard = cv2.waitKey(30)
        frame_idx += 1
        if keyboard == ord('q') or keyboard == 27:
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()





def main():

    # If data needs to be pulled
    # V7_KEY = V7_KEY = "Wg78nME.WAvo-Sp-C7yriT1csJEg-f6nwhr57A8u"
    # client = Client.from_api_key(V7_KEY)
    # dataset = client.get_remote_dataset('eos')
    # release = dataset.get_release()
    
    # if release:
    #     print("Release found. Pulling annotations...")
    #     dataset.pull(release=release)
    #     print("Annotations pulled successfully.")
    #     return
        


    video_files = [f for f in os.listdir(video_folder)]
  

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
