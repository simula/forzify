import numpy as np 
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

keypoint_file = "labels_frames_for_training/keypoints/bounding_boxes_and_keypoints_from_labelstudio_2.csv"

df = pd.read_csv(keypoint_file)

pattern = r"([^/]+)(?=\.png)"
for index, row in df.iloc[:-1].iterrows():  #The last row is an example image
    filename = row["img"]
    match = re.search(pattern, filename)
    filename = match.group(1)
    print(filename, index)

    line = "0"  #start with class football pitch, only one possible class

    bounding_box = json.loads(row["pitch"])[0]

    # bounding_x = bounding_box['x']/100*1280
    # bounding_y = bounding_box['y']/100*720
    # bounding_width = bounding_box['width']/100*1280
    # bounding_height = bounding_box['height']/100*720
    # if index == 0:
    #     image_file = "Data/frames_for_training/train/1752_abu41w8te4nzl_105x65_frame_38.png"
    #     image = Image.open(image_file)
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(image)

    #     # Create a rectangle patch
    #     rect = patches.Rectangle((bounding_x, bounding_y), bounding_width, bounding_height, linewidth=2, edgecolor='r', facecolor='none')

    #     # Add the rectangle to the plot
    #     ax.add_patch(rect)

    #     # Display the plot
    #     plt.show()



    line += f" {(bounding_box['x'] + bounding_box['width']/2)/100} {(bounding_box['y'] + bounding_box['height']/2)/100} {bounding_box['width']/100} {bounding_box['height']/100}"
    print(bounding_box["x"], bounding_box["y"])
    keypoints = json.loads(row["kp-1"])
    keypoint_list = np.zeros((43, 3)) #43 possible keypoints
    prev_keypointlabel = None
    for dict in keypoints:
        keypointlabel = int(dict["keypointlabels"][0])
        keypoint_list[keypointlabel - 1][0] = dict["x"]/100
        keypoint_list[keypointlabel - 1][1] = dict["y"]/100
        keypoint_list[keypointlabel - 1][2] = 1

        prev_keypointlabel = keypointlabel


        #line += f" {dict['x']/100} {dict['y']/100}"

    for i in range(len(keypoint_list)):
        for j in range(len(keypoint_list[0])):
            line += f" {keypoint_list[i][j]}"

    with open("labels_frames_for_training/keypoints/txt/" + filename + ".txt", 'w') as file:
        file.write(line)
        