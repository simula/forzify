from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

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

                # Manually process and annotate the frame for Goalkeepers only (class ID 1)
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if class_id == 1:  # Filter for Goalkeepers
                        x, y, w, h = box
                        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

                        # Draw filled rectangle with transparency
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red color fill
                        alpha = 0.3  # Transparency factor
                        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                        # Draw detection box border (optional, for better visibility)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw text label
                        label = "Goalkeeper"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        track = track_history[track_id]
                        track.append((int(x), int(y)))
                        if len(track) > 30:
                            track.pop(0)

                        # Draw tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

                # Write the processed frame to the output video file
                out.write(frame)

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
