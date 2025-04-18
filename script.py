import os
import cv2
import json
import numpy as np
from darwin.client import Client
from config import V7_KEY

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
            'x': bbox_data['x'],
            'y': bbox_data['y'],
            'width': bbox_data['w'],
            'height': bbox_data['h'],
            'label': 'Object'  # Since no label is given, we assign "Object"
        }
        frame_annotations.setdefault(frame_idx, []).append(bbox)

    return frame_annotations

def analyze_roi(frame, bbox, margin_ratio=0.75):
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

    # Adding margin to the bounding box
    margin_w = w * margin_ratio
    margin_h = h * margin_ratio
    x = max(int(x - margin_w / 2), 0)
    y = max(int(y - margin_h / 2), 0)
    w = int(w + margin_w)
    h = int(h + margin_h)

    # Crop the ROI from the frame
    roi = frame[y:y+h, x:x+w]

    # Convert the ROI to HSV for color filtering
    color_mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    contrast_img = cv2.convertScaleAbs(color_mask, alpha=1.5, beta=-50)
    cv2.imshow("test", contrast_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

 

    # Overlay the result on the original ROI
    # final_roi = cv2.addWeighted(roi, 1.0, color_mask, 1.0, 0)


    # # Put the processed ROI back into the original frame
    # frame[y:y+h, x:x+w] = final_roi

    return frame


def process_video(video_path, annotation_path, annotation_fps=1.0):
    cap = cv2.VideoCapture(video_path)
    annotations = load_annotations(annotation_path)

    # Capture first frame only for testing
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    annot_frame = 0
    frame_annots = annotations.get(str(annot_frame), [])

    if frame_annots:
        # Apply Canny edge detection inside the first bbox
        frame = analyze_roi(frame, frame_annots[0])

    # Display the processed frame
    cv2.imshow("Processed Frame", frame)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

def main():
    #If data needs to be pulled
    # client = Client.from_api_key(V7_KEY)
    # dataset = client.get_remote_dataset('eos')
    # release = dataset.get_release()
    
    # if release:
    #     print("Release found. Pulling annotations...")
    #     dataset.pull(release=release)
    #     print("Annotations pulled successfully.")
    #     return

    video_files = [f for f in os.listdir(video_folder) if f.endswith('.MP4')]

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_folder, video_file)
        annotation_path = os.path.join(annotations_folder, f"{video_name}.json")

        if os.path.exists(annotation_path):
            print(f"➡️ Processing {video_file} with {video_name}.json")
            process_video(video_path, annotation_path)
        else:
            print(f"⚠️ No matching annotation file for {video_file}, skipping.")

if __name__ == "__main__":
    main()
