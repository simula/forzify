import os
from ultralytics import YOLO  # Ensure correct import of YOLO
import cv2

def process_images(image_folder, model_path, output_file, selected_classes):
    # Initialize the YOLO model
    model = YOLO(model_path)

    # Define class names and find the IDs for the selected classes
    class_names = ['Player', 'Goalkeeper', 'Referee', 'Ball', 'Logo', 'Penalty Mark', 'Corner Flagpost', 'Goal Net']
    selected_class_ids = [class_names.index(cls) for cls in selected_classes]

    tracked_objects = []
    frame_number = 0

    # Process each image in the folder
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            frame = cv2.imread(image_path)

            # Perform tracking on the frame
            results = model.track(frame, persist=True)
            if results and results[0] and results[0].boxes:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id
                class_ids = results[0].boxes.cls

                # Check if tracking IDs and class IDs are available
                if track_ids is not None and class_ids is not None:
                    track_ids = track_ids.int().cpu().tolist()
                    class_ids = class_ids.int().cpu().tolist()

                    # Process each detection
                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        if class_id in selected_class_ids:
                            # Extract bounding box coordinates
                            x_center, y_center, w, h = box
                            x1 = int(x_center - w / 2)
                            y1 = int(y_center - h / 2)
                            width = int(w)
                            height = int(h)
                            
                            print(f'{track_id} is track id of goalkeeper')
                            print(f'{class_id} is class id of goalkeeper')

                            # Store the tracked object information
                            tracked_objects.append([frame_number, track_id, x1, y1, width, height, 1, -1, -1, -1])

            frame_number += 1

    # Save the tracked objects data to a file
    with open(output_file, 'w') as f:
        for obj in tracked_objects:
            f.write(','.join(map(str, obj)) + '\n')

# Usage example
process_images('/home/mehdihou/D1/soccernet_tracking/tracking/train/SNMOT-071/img1', '/home/mehdihou/D1/YOLO-Train/Train/750_Detection_01/train/weights/best.pt', '/home/mehdihou/D1/soccernet_tracking/tracking/train/SNMOT-071/soccersum_track.csv', ['Goalkeeper'])
