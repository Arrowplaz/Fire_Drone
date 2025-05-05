import os
import cv2
import json
import numpy as np
from darwin.client import Client

# === Set folder paths ===
video_folder = "../.darwin/datasets/honors/eos/images"
annotations_folder = "../.darwin/datasets/honors/eos/releases/test-2/annotations"

# === Load annotation JSON ===
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
        frame_annotations[int(frame_idx)] = [bbox]  # cast frame_idx to int

    return frame_annotations

# === Analyze a single ROI inside a mask ===
def analyze_roi_with_mask(frame, mask, annotation):
    x, y, w, h = map(int, annotation['bbox'])

    height, width = mask.shape
    x, y = max(0, x), max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)

    roi_mask = mask[y:y+h, x:x+w]
    edges = cv2.Canny(roi_mask, 100, 200)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.7, edges_bgr, 0.3, 0)
    return frame

# === Preprocessing logic ===
def preprocess(video_path, annotation_path, output_path='output.mp4'):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorKNN()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    annotations = load_annotations(annotation_path)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)

        if frame_idx in annotations:
            for ann in annotations[frame_idx]:
                frame = analyze_roi_with_mask(frame, fgmask, ann)

        out.write(frame)

        resized_frame = cv2.resize(frame, (960, 540))  # resize for display
        cv2.imshow('Processed Video', resized_frame)

        if cv2.waitKey(30) in [ord('q'), 27]:
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# === Main execution with optional data pull ===
def main():
    # === Optional: Pull data from V7 ===
    V7_KEY = "Wg78nME.WAvo-Sp-C7yriT1csJEg-f6nwhr57A8u"
    client = Client.from_api_key(V7_KEY)
    dataset = client.get_remote_dataset('eos')
    release = dataset.get_release(name="test-2")

    if release:
        print("✅ Release found. Pulling annotations...")
        dataset.pull(release=release)
        print("✅ Annotations pulled successfully.")

    print("📂 Processing videos...")
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.mov', '.avi'))]

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_folder, video_file)
        annotation_path = os.path.join(annotations_folder, f"{video_name}.json")

        if os.path.exists(annotation_path):
            print(f"➡️ Processing full video: {video_file}")
            output_path = os.path.join(video_folder, f"{video_name}_processed.mp4")
            preprocess(video_path, annotation_path, output_path=output_path)
        else:
            print(f"⚠️ No matching annotation for {video_file}, skipping.")

if __name__ == "__main__":
    main()
