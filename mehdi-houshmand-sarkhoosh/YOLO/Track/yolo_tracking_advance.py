from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

def find_dominant_color(image, k=1):
    """Find the dominant color in an image."""
    pixels = np.float32(image.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return tuple(int(c) for c in dominant)

# Load the YOLOv8 model
model = YOLO('/home/mehdihou/D1/YOLO-Train/Train/Thesis_detection_03/train/weights/best.pt')

# Open the video file
video_path = "/home/mehdihou/D1/AIProducer-Hoceky/AIProducer-Hockey/aiproducer/debug/debug/soccer0.mp4"
cap = cv2.VideoCapture(video_path)

# Get frame size from the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('tracking1.mp4', fourcc, 20.0, (frame_width, frame_height))

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Check if results are not None
        if results and results[0] and results[0].boxes:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id
            class_ids = results[0].boxes.cls

            if track_ids is not None and class_ids is not None:
                track_ids = track_ids.int().cpu().tolist()
                class_ids = class_ids.int().cpu().tolist()

                # Manually process and annotate the frame for Players and Goalkeepers
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if class_id in [0, 1]:  # Filter for Players (0) and Goalkeepers (1)
                        label = "Player" if class_id == 0 else "Goalkeeper"
                        x, y, w, h = box
                        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

                        # Extract the region of interest (ROI)
                        roi = frame[y1:y2, x1:x2]
                        if roi.size != 0:  # Ensure ROI is not empty
                            dominant_color = find_dominant_color(roi, k=1)

                            # Draw filled rectangle with the dominant color
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), dominant_color, -1)
                            alpha = 0.8  # Increased transparency factor for more intense color
                            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                            # Draw text label
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                        track = track_history[track_id]
                        track.append((int(x), int(y)))
                        if len(track) > 30:
                            track.pop(0)

                        # Draw tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=dominant_color, thickness=2)

                # Write the processed frame to the output video file
                out.write(frame)

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows
