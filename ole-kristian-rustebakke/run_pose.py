#import comet_ml
import torch
import time
#import logging
import psutil
import GPUtil
from ultralytics import YOLO
import os
import json
import numpy as np
import pandas as pd
import re
import cv2
import matplotlib.pyplot as plt

# Check CUDA availability
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available with {num_gpus} GPU{'s' if num_gpus > 1 else ''}.")

    for i in range(torch.cuda.device_count()):
        GPU_name = print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")


#2D keypoints:
keypoint_file = "labels_frames_for_training/keypoints/bounding_boxes_and_keypoints_from_labelstudio.csv"

df = pd.read_csv(keypoint_file)

pattern = r"([^/]+)(?=\.jpg)"
dst_keypoints = np.zeros((43, 2)) #39 possible keypoints
for index, row in df.iloc[-1:].iterrows():  #The last row is an example image
    filename = row["img"]
    match = re.search(pattern, filename)
    filename = match.group(1)
    print(filename, index)

    keypoints = json.loads(row["kp-1"])
    
    for dict in keypoints:
        keypointlabel = int(dict["keypointlabels"][0])
        dst_keypoints[keypointlabel - 1][0] = dict["x"]/100*1280
        dst_keypoints[keypointlabel - 1][1] = dict["y"]/100*720

    bounding_box = json.loads(row["pitch"])[0]
    x = bounding_box["x"]
    y = bounding_box["y"]
    w = bounding_box["width"]
    h = bounding_box["height"]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    x1y1x2y2 = (x1, y1, x2, y2)

        

# Define pose models dictionary - using pose-specific models
best_model = "/itf-fi-ml/home/olekrus/Deep-EIoU/runs/pose/train54/weights/best.pt"

model = YOLO(best_model)
#data_path = "/itf-fi-ml/home/olekrus/master/master/Data/test/images"
data = "/itf-fi-ml/home/olekrus/master/master/config_pose_test.yaml"
# save_path = "/itf-fi-ml/home/olekrus/Deep-EIoU/runs/pose/test"
# run_name = "test_run"
# metrics = model.val(data=data, project=save_path, name=run_name) #test metrics


data_folder = "/itf-fi-ml/home/olekrus/master/master/Data/test/images"
conf_threshold = 0.85
keypoints_list = []
dst_keypoints_list = []
bounding_box_list = []
image_list = os.listdir(data_folder)
for filename in image_list:
    if filename.endswith(".png"):
        image_path = os.path.join(data_folder, filename)
        results = model(image_path)

        box = results[0].boxes.xywh
        box = box.flatten().cpu().numpy()
        bounding_box_list.append(box)

        confidence_scores = results[0].keypoints.conf
        confidence_scores = confidence_scores.cpu().numpy().flatten()

        keypoints = results[0].keypoints.xy
        keypoints = keypoints.cpu().numpy().squeeze()
        for i in range(len(confidence_scores)):
            if confidence_scores[i] < conf_threshold:
                keypoints[i, :] = 0

        condition = np.all(keypoints != 0, axis=1)

        non_zero_indices = np.where(condition)[0]
        keypoints = keypoints[condition]

        keypoints_list.append(keypoints)
        dst_keypoints_list.append(dst_keypoints[non_zero_indices])



for i in range(len(keypoints_list)):
    bounding_box = bounding_box_list[i]
    x = bounding_box[0]
    y = bounding_box[1]
    w = bounding_box[2]
    h = bounding_box[3]
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2

    keypoints = keypoints_list[i]
    keypoints[:,0] = keypoints[:,0] - x1 #*w/1280
    keypoints[:,1] = keypoints[:,1] - y1#*h/720

    dst_keypoints = dst_keypoints_list[i]
    dst_keypoints[:,0] = dst_keypoints[:,0]*w/1280 #- x1#*w/1280
    dst_keypoints[:,1] = dst_keypoints[:,1]*h/720 #- y1#*h/720

    keypoints = keypoints.astype(np.float32)
    dst_keypoints = dst_keypoints.astype(np.float32)

    # Calculate the perspective transformation matrix
    matrix, mask = cv2.findHomography(keypoints, dst_keypoints, cv2.RANSAC)
    if matrix is None:
        print("Homography matrix computation failed!")
    else:
        image = cv2.imread(os.path.join(data_folder, image_list[i]))
        image_arr = np.array(image)

        # Extract the image region defined by the bounding box
        field_image = image_arr[round(y1):round(y2), round(x1):round(x2)]


        frame_height, frame_width = field_image.shape[0:2]
        transformed_frame = cv2.warpPerspective(field_image, matrix, (frame_width, int(frame_height)))
        transformed_frame = cv2.resize(transformed_frame, (1280, 720))

        plt.figure()
        plt.imshow(transformed_frame)
        # plt.scatter(keypoints[:, 0], keypoints[:, 1], color='red', label='Keypoints')  # Plot keypoints in red
        plt.scatter(dst_keypoints[:, 0]*1280/w, dst_keypoints[:, 1]*720/h, color='blue', label='Keypoints')
        plt.savefig(os.path.join("/itf-fi-ml/home/olekrus/master/master/Data/results", image_list[i]))





