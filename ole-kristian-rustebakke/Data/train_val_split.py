import shutil
import os
import random


parent_dir_frames = "/itf-fi-ml/home/olekrus/master/master/frames_for_training/"
parent_dir_txt = "/itf-fi-ml/home/olekrus/master/master/labels_frames_for_training/keypoints/txt/"

frame_files = sorted(os.listdir(parent_dir_frames)) #Alphabetical order
txt_files = sorted(os.listdir(parent_dir_txt)) #alphabetical order
indices = list(range(len(frame_files)))
random.shuffle(indices)
frame_files = [frame_files[i] for i in indices]
txt_files = [txt_files[i] for i in indices]

outcome = [0, 1, 2]
weights = [0.7, 0.15, 0.15]  # 70% chance for 0, 15% chance for 1, 15% chance for 2

train_folder = "/itf-fi-ml/home/olekrus/master/master/Data/train/"
val_folder = "/itf-fi-ml/home/olekrus/master/master/Data/val/"
test_folder = "/itf-fi-ml/home/olekrus/master/master/Data/test/"

for file_ind in range(len(frame_files)):
    print(frame_files[file_ind], txt_files[file_ind])

    new_image_name = f"img_{file_ind:04d}_{frame_files[file_ind]}"
    new_txt_name = f"img_{file_ind:04d}_{txt_files[file_ind]}"

    train_val_test = random.choices(outcome, weights=weights, k=1)[0]
    if train_val_test == 0: #Train
        shutil.copy(parent_dir_frames + frame_files[file_ind], train_folder + "images/" + new_image_name)
        shutil.copy(parent_dir_txt + txt_files[file_ind], train_folder + "labels/" + new_txt_name)

    elif train_val_test == 1: #Val
        shutil.copy(parent_dir_frames + frame_files[file_ind], val_folder + "images/" + new_image_name)
        shutil.copy(parent_dir_txt + txt_files[file_ind], val_folder + "labels/" + new_txt_name)

    else: #Test
        shutil.copy(parent_dir_frames + frame_files[file_ind], test_folder + "images/" + new_image_name)
        shutil.copy(parent_dir_txt + txt_files[file_ind], test_folder + "labels/" + new_txt_name)