import numpy as np
import matplotlib.pyplot as plt 
import os
import json
from pathlib import Path
import cv2 

bbox_dir = "/itf-fi-ml/home/olekrus/master/master/dribbling-detection-pipeline/outputs_soccernet/"
output_dir = "/itf-fi-ml/home/olekrus/master/master/dribbling-detection-pipeline/"

for dir in os.listdir(bbox_dir):
    interpolated_dir = bbox_dir + dir + "/interpolated-predictions/"
    dir_name = [dir for dir in Path(interpolated_dir).iterdir()][0]
    file_path = Path(str(dir_name) + "/Labels-GameState.json")
    image_dir = str(dir_name) + "/img1/"

    with file_path.open("r") as file:
        bbox_data = json.load(file)

    fps = 25
    frame_width = 1920
    frame_height = 1080
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video_path = output_dir + "interpolated_bounding_boxes_soccernet/" + str(Path(dir_name).name) + ".mp4"
    print(output_video_path)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_counter = 0
    for image_info in bbox_data["images"]:
        print(f"Processing frame {frame_counter}/{len(bbox_data['images'])}", end="\r")
        frame_counter += 1
        image_path = os.path.join(image_dir, image_info["file_name"])
        image_id = image_info["image_id"]

        
        bboxes = []
        ids = []

        for annotation in bbox_data["annotations"]:
            if annotation["image_id"] == image_id and annotation["supercategory"] == "object":
                bboxes.append(annotation["bbox_image"])
                ids.append(annotation["track_id"])

        image = cv2.imread(image_path)
        for i, bbox in enumerate(bboxes):
            x = int(bbox['x'])
            y = int(bbox['y'])
            w = int(bbox['w'])
            h = int(bbox['h'])

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  

            track_id = str(ids[i])
            cv2.putText(image, track_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)  

        out.write(image)

    out.release()
    cv2.destroyAllWindows()

