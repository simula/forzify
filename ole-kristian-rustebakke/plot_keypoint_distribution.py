import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

keypoint_file = "labels_frames_for_training/keypoints/bounding_boxes_and_keypoints_from_labelstudio.csv"

df = pd.read_csv(keypoint_file)

keypoint_count = np.zeros(40)
for index, row in df.iloc[:-2].iterrows(): 
    keypoints = json.loads(row["kp-1"])

    for dict in keypoints:
        keypointlabel = int(dict["keypointlabels"][0])
        keypoint_count[keypointlabel] += 1

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(keypoint_count)), keypoint_count[1:])
plt.xlabel('Keypoint Index')
plt.ylabel('Count')
plt.title('Keypoint Count Histogram')
plt.xticks(range(1, len(keypoint_count)))  # Set x-ticks to match the keypoint indices
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional: Add gridlines for better readability
plt.savefig("keypoints.png")

