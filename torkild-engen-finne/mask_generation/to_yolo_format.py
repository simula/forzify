import cv2
import numpy as np
import os

def mask_to_yolo_segmentation(mask, class_id=0):
    """
    Convert a binary mask to YOLO format segmentations
    """
    h, w = mask.shape

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []
    for contour in contours:
        if len(contour) < 3:
            continue

        line = [str(class_id)]
        for point in contour.squeeze():
            x, y = point
            line.append(f"{x / w:.6f}") # 6 decimals
            line.append(f"{y / h:.6f}")
        yolo_lines.append(" ".join(line))

    return yolo_lines


glob_dir = ""
out_dir_img = "/images/"
out_dir_mask = "/labels/"
geometry_groups_mask = {"Full field": None, "Big rect. left": None, "Big rect. right": None, "Goal left": None, "Goal right": None, "Circle right": None, "Circle left": None, "Circle central": None, "Small rect. left": None, "Small rect. right": None}
geometry_groups_mask_class_id = {"Full field": 0, "Big rect. left": 1, "Big rect. right": 2, "Goal left": 3, "Goal right": 4, "Circle right": 5, "Circle left": 6, "Circle central": 7, "Small rect. left": 8, "Small rect. right": 9}
geometry_groups_mask_names = ["Full field", "Big rect. left", "Big rect. right", "Goal left", "Goal right", "Circle right", "Circle left", "Circle central", "Small rect. left", "Small rect. right"]



all_imgs = []
all_masks = []

seq_dir = os.listdir(glob_dir)

seq_dir = [d for d in seq_dir if os.path.isdir(os.path.join(glob_dir, d))]
seq_dir.sort()

i = 0
for seq in seq_dir:
    #Going through each sequence and generating masks on yolo format and images
    print(f"Processing sequence {seq}")
    seq_dir_full = os.path.join(glob_dir, seq, "img1")
    
    seq_list = os.listdir(seq_dir_full)

    seq_list = [img for img in seq_list if img.endswith(".jpg")]
    seq_list.sort()

    for img in seq_list:
        img_path = os.path.join(seq_dir_full, img)
        img = cv2.imread(img_path)
        
        mask_path = os.path.join(seq_dir_full, img_path.replace(".jpg", ".npy"))
        mask = np.load(mask_path, allow_pickle=True)
        mask = mask.item()
        all_masks = []
        for key in mask.keys():
            if key not in geometry_groups_mask.keys():
                print(f"Key {key} not in geometry_groups_mask")

                continue
            mask_id = geometry_groups_mask_class_id[key]
            lines = mask_to_yolo_segmentation(mask[key], class_id=mask_id)
            all_masks.extend(lines)

        output_txt = out_dir_mask + str(i) + ".txt"

        cv2.imwrite(out_dir_img + str(i) + ".jpg", img)
        with open(output_txt, "w") as f:
            f.write("\n".join(all_masks))
        
        i += 1