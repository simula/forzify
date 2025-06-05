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
from funcs import *

#2D keypoints:
keypoint_file = "labels_frames_for_training/keypoints/bounding_boxes_and_keypoints_from_labelstudio_2.csv"

df = pd.read_csv(keypoint_file)

pattern = r"([^/]+)(?=\.jpg)"
dst_keypoints = np.zeros((43, 2)) #43 possible keypoints
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


bounding_box_ids = {"1738_avxeiaxxw6ocr-main": [[37]*11], 
                    "1738_aw3kercrl5skd-main": [[14]*11], 
                    "1742_ab2m9r2lve8xx-main": [[4]*11], 
                    "1744_ai0bkwrj1o6d3-main": [[34]*11], 
                    "1747_aas7m6x9khzm6-main": [[18]*11],
                    "1757_aaeaaztoinvzx-main": [[12]*11], 
                    "1757_aebqf8o3tg8wm-main": [[30]*11], 
                    "1765_ayt7hdvmhndz1-main": [[29]*11], 
                    "1767_ak7ro2ewn2hen-main": [[7]*11], 
                    "1769_akarclpi8dv1z-main": [[9]*11, [None, None, None, None, 9, None, None, None, None, None, None]], 
                    "1771_au5v9dd1iya1u-main": [[20]*11], 
                    "1773_a5k6fgshvjmvl-main": [[17, 17, 17, 17, 17, 17, 17, 17, 17, 19, 19]], 
                    "1776_as3pvx4kblaoy-main": [[16]*11], 
                    "1777_akd6w6rwzvbnw-main": [[28]*11], 
                    "1783_akxegh8qnmh0c-main": [[3]*11],
                    "1784_a309lpf9tgfjk-main": [[5, 4, None, 4, None, 4, None, 4, 4, None, 4]], 
                    "1786_aslryruw4a18u-main": [[4]*11], 
                    "1814_abujpl1vbmlpv-main": [[8]*11], 
                    "1814_av54fv6tckm63-main": [[9]*11], 
                    "1816_alz6d3trmjh1f-main": [[4, 4, 4, 4, 4, 4, 4, 4, 4, None, None]], 
                    "1821_ah8wqifyc7s7d-main": [[15, 15, 15, 15, 15, 15, 15, 15, 39, 15, 15]],
                    "1825_aq5qrn71iw73p-main": [[9]*11], 
                    "1827_aa7wtwlt86673-main": [[26]*11], 
                    "1834_amoai9g7gbzs3-main": [[32]*11], 
                    "1837_agt9x37n1f90h-main": [[28]*11], 
                    "1845_a6p35pr2oc066-main": [[9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5]], 
                    "1845_aa29zru0nnbn-main": [[20]*11], 
                    "1845_alj72ynli7zbn-main": [[23]*11], 
                    "1858_akkzn5c9u4u1b-main": [[5]*11], 
                    "1862_a8rmubj0ok7zm-main": [[9]*11], 
                    "1871_aieelpjpc384t-main": [[30]*11],
                    "1875_ajbz8ndcnzars-main": [[14]*11], 
                    "1877_aey3e3ta279ce-main": [[5]*11], 
                    "1902_a39ztm4o3824v-main": [[15]*11], 
                    "1915_aoui433sscrt2-main": [[13, 13, 13, 13, 8, 8, 8, 8, 8, 8, 8]], 
                    "1959_arh0h7tsjh69c-main": [[8]*11], 
                    "1962_a6kcrd3r8ovs-main": [[10]*11], 
                    "1968_agc6a87zu9ii1-main": [[7]*11], 
                    "1975_axao1rs21074-main": [[12]*11], 
                    "3181_agkzjzgidsgd2-main": [[18]*11],
                    "3181_agvu1d4f8ybf3-main": [[8]*11], 
                    "3182_a4vdmqg3cwuyn-main": [[35]*11], 
                    "3183_adg5id5325cuv-main": [[5]*11], 
                    "3183_akkgiec7wkss7-main": [[9]*11], 
                    "3185_ac7aak7jrcxm8-main": [[19, 19, 19, 19, 19, 19, 19, 19, 19, 12, 12]], 
                    #"3185_as2w8o9cprai7-main": [[12]*11], 
                    "3188_a4h5fuedm2xzr-main": [[None, None, None, 6, 6, None, None, 6, 6, 6, None]], 
                    "3194_aaqhazydn9id8-main": [[8]*11], 
                    "3197_a7mq9mlnjzq1x-main": [[4]*11],
                    "3198_a90f88xq8t1ut-main": [[None, 11, 11, None, None, None, None, None, None, None, None]], 
                    "3201_ab79q1e5zh9cd-main": [[8, 8, 9, 9, 9, 9, 9, 9, 8, 9, None]], 
                    "3206_a5wjsg0bq1ngj-main": [[36, 36, None, 36, 36, 36, 36, 36, 36, 36, None]], 
                    "3208_a4d6n4uu08j6p-main": [[40, 40, 4, 40, 40, None, 40, 7, 40, 40, 40]], 
                    "3211_aq2zvwssruck4-main": [[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 27], [27]*11], 
                    "3212_aet2zo7nas12j-main": [[8]*11], 
                    "3213_awytedvil97mu-main": [[2]*11], 
                    "3217_a81f6sk050e72-main": [[22, 22, 22, 22, 22, 22, 22, 22, 15, 15, 22]], 
                    "3218_agffo3bhwew2y-main": [[15]*11], 
                    "3219_ayq7zm7excchf-main": [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4]], 
                    "3220_avplqkhgzsp0a-main": [[32]*11],
                    "3221_aows07a24hzso-main": [[19, 19, 31, 31, 31, 31, 31, 31, 31, 31, 18]], 
                    "3223_aoat0rpbec3jz-main": [[26]*11], 
                    "3223_aohfwf10lp1dg-main": [[28]*11, [8]*11], 
                    "3227_ackegq4w6hv6-main": [[3]*11], 
                    "3227_alflphjllmv60-main": [[8]*11, [1]*11], 
                    "3227_as3pxmkpfaq5p-main": [[5]*11], 
                    "3227_avm4p96cuq29b-main": [[8]*11],
                    "3230_akfiynywg8rc9-main": [[9]*11], 
                    "3236_asnnfeyppx4bs-main": [[10]*11], 
                    "3237_aqmh3owxt46ji-main": [[1]*11], 
                    "3239_a96rowrzxa9kl-main": [[7]*11, [5]*11], 
                    "3239_awlqa708ufga8-main": [[7, 7, 7, 14, 7, 7, 7, 7, 7, 7, 7, 7], [30, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29]], 
                    "3243_akkx7abi0qnwy-main": [[None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 
                    "3246_a7q2of9a0pd6a-main": [[16]*11],
                    "3246_ahrs3qyzhc04m-main": [[30]*11], 
                    "3247_a7rii7bfxn52t-main": [[2, None, 2, 2, 2, 2, 2, 2, None, None, 2]], 
                    "3250_auw8h8bpm7vab-main": [[9, 9, 9, 9, 9, 9, 9, 9, 9, 2, 2]], 
                    "3251_asofekemaqjtq-main": [[7]*11], 
                    "3252_a3bxtm2ke0cl1-main": [[15]*11], 
                    #"3255_acmh6ytfvezfu-main": [[8]*11], 
                    "3265_ah78uxdwg7ol0-main": [[11]*11], 
                    "3266_ag95e7qiyar9f-main": [[1]*11], 
                    "3268_ak8tp20ta0usv-main": [[12]*11], 
                    "3269_acvc2894kckby-main": [[32]*11],
                    "3270_ak3og7x638lfd-main": [[14]*11], 
                    "3271_ahjx9rzchewnj-main": [[1, 1, 1, 1, 1, 1, 1, None, 15, 15, 1]], 
                    "3275_ae6yqpy36wawy-main": [[22, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]], 
                    "3276_aq0vkqdnewz3j-main": [[3]*11], 
                    "3278_av2482qp96f94-main": [[15, 15, None, 15, 15, 15, 15, 15, 15, 15, 15]], 
                    "3279_appg029alt88v-main": [[13]*11], 
                    "3284_ao7b1fdw9gcio-main": [[11]*11], 
                    "3284_aqvzrpx4l4ug8-main": [[10]*11], 
                    "3287_aebn04cxvaya4-main": [[6]*11], 
                    "3291_am1fnjz8bhv73-main": [[25, 25, 25, 25, 25, 3, 3, 3, 25, 25, 25]], 
                    "3295_a7p42zcq5gp9k-main": [[31]*11], 
                    "3295_als72y8ksh3d0-main": [[7]*11],
                    "3299_a7pdtki1rirvk-main": [[15]*11], 
                    "3299_ajqg5w0xt2tzd-main": [[5]*11], 
                    "3300_ae2o2j2s0ghdh-main": [[3]*11], 
                    "3309_adip6nafqmwt2-main": [[2]*11], 
                    "3313_ax2oaplec7ypx-main": [[16]*11], 
                    "3313_axcep3pyh09y0-main": [[7]*11], 
                    "3318_a6kvvchxp3n02-main": [[3]*11], 
                    "3323_aqc0fw2i8km9q-main": [[15]*11], 
                    "3324_ayikh1kn1jf1-main": [[17]*11], 
                    "3325_ahi4gu1oo5u7g-main": [[6, 6, 6, None, None, 6, 6, 6, 6, 6, 4]], 
                    "3326_av1hfseysxvlz-main": [[11]*11],
                    "3329_a5g9fv7h61tyu-main": [[34]*11],
                    "3332_ai3tt4k6dhczb-main": [[4]*11], 
                    "3334_ab2jsop8zqw3p-main": [[None, None, None, None, 1, 1, 1, 1, 1, 1, 1]], 
                    "3338_a8liptbgsg8hl-main": [[8]*11], 
                    "3338_aaji73b5nainn-main": [[19]*11], 
                    "3338_al8kgl5ooyk7a-main": [[18, 18, 18, 11, 11, 11, 18, 18, None, 18, 18]], 
                    "3342_ac38qpotfv65y-main": [[35]*11], 
                    "3342_atosyyiv9wnhn-main": [[None, None, None, None, 18, 12, 12, 12, 12, 12, 12]], 
                    #"3343_aragn0yk1ipap-main": [[8, None, 8, 8, 8, 8, 8, 8, 8, 8, 8]], 
                    "3348_ad1c3funotaad-main": [[25]*11], 
                    "3351_ays4y42mdh0bf-main": [[3]*11], 
                    "3353_a42b7leuixe38-main": [[6]*11],
                    "3354_awur9crbaqckv-main": [[None]*11], 
                    "3355_am3m8v45ee0fi-main": [[17]*11], 
                    "3355_atu85xsbh0xoc-main": [[3, 3, 3, 22, 22, 22, 22, 22, 22, 22, 22]], 
                    "3364_ar3n9z5v3atdy-main": [[10]*11, [20, 20, 20, 20, 20, 20, 28, 28, 28, 28, 28], [28]*11],
                    "3366_akb38710r8kyc-main": [[2]*11], 
                    "3370_aabltdwvl604m-main": [[10]*11, [24]*11], 
                    "3370_ae6qndhmjt9wt-main": [[12]*11], 
                    "3370_ava4o66x5yt3e-main": [[5, 5, 5, None, None, 5, 5, 5, 5, 5, 5]], 
                    "3375_ac0r17bq4c97h-main": [[1]*11], 
                    "3375_akdgamca3n5me-main": [[23]*11], 
                    "3377_abstqd3kdmelk-main": [[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, None]],
                    "3383_ab9fy58to0fue-main": [[1]*11, [22, 22, 22, 22, 22, 22, 22, 22, None, None, 22]], 
                    "3383_ai9saev3ivyte-main": [[8]*11], 
                    "3385_akcz43ge3mpec-main": [[2]*11], 
                    "3386_a47m0uqqg8f5e-main": [[2]*11], 
                    "3389_a7a5u3ihj8v9t-main": [[5]*11], 
                    "3389_ahom8t5szkj7-main": [[6]*11], 
                    "3389_ammj6l4t44a04-main": [[9]*11], 
                    "3391_aimiuhf1beuti-main": [[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 14]], 
                    "3392_aise66a88lgxp-main": [[9]*11], 
                    "3392_aut0dhwa5iaxi-main": [[1]*11], 
                    "3393_armwr1xhhvydt-main": [[5]*11], 
                    "3396_aih1r69jxq58z-main": [[17]*11], 
                    "3398_awulwqe6gqod-main": [[13]*11],
                    "3399_aolb2vfy7lf3x-main": [[19]*11], 
                    "3402_aj28827ic4y4z-main": [[6]*11], 
                    "3405_avjr7cotyfbtf-main": [[4]*11, [14, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]], 
                    "3406_ad79v02olrv6h-main": [[9]*11], 
                    "3409_aik0pl1ylihuz-main": [[8]*11], 
                    "3410_aqq8aeahwsxe-main": [[19]*11, [19]*11], 
                    "3411_aaf40mb2t015p-main": [[3, 3, 3, 3, 3, 3, 3, None, 10, 7, None]], 
                    "3412_a9863uj1lxmbq-main": [[5, 5, 5, 5, 5, 5, 5, 12, 12, 12, 12]], 
                    "3413_asok8wqxa4b5-main": [[5]*11], 
                    "3415_anub7qf1erg12-main": [[32]*11], 
                    "3417_ay8cjudnvafzl-main": [[9, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]]}


# bounding_box_ids = [[[37]*11], [[14]*11], [[4]*11], [[34]*11], [[18]*11], [[12]*11], [[30]*11], [[29]*11], [[7]*11], [[9]*11, 
#                 [None, None, None, None, 9, None, None, None, None, None, None]], [[20]*11], [[17, 17, 17, 17, 17, 17, 17, 17, 17, 19, 19]],
#                 [[16]*11], [[28]*11], [[3]*11], [[5, 4, None, 4, None, 4, None, 4, 4, None, 4]], [[4]*11], [[8]*11], [[9]*11], 
#                 [[4, 4, 4, 4, 4, 4, 4, 4, 4, None, None]], [[15, 15, 15, 15, 15, 15, 15, 15, 39, 15, 15]], [[9]*11], [[26]*11], [[32]*11],
#                 [[28]*11], [[9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5]], [[20]*11], [[23]*11], [[5]*11], [[9]*11], [[30]*11], [[14]*11], [[5]*11], [[15]*11],
#                 [[13, 13, 13, 13, 8, 8, 8, 8, 8, 8, 8]], [[8]*11], [[10]*11], [[7]*11], [[12]*11], [[18]*11], [[8]*11], [[35]*11], [[5]*11], [[9]*11],
#                 [[19, 19, 19, 19, 19, 19, 19, 19, 19, 12, 12]], [[12]*11], [[None, None, None, 6, 6, None, None, 6, 6, 6, None]], [[8]*11],
#                 [[4]*11], [[None, 11, 11, None, None, None, None, None, None, None, None]], [[8, 8, 9, 9, 9, 9, 9, 9, 8, 9, None]],
#                 [[36, 36, None, 36, 36, 36, 36, 36, 36, 36, None]], [[40, 40, 4, 40, 40, None, 40, 7, 40, 40, 40]], [[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 27], [27]*11],
#                 [[8]*11], [[2]*11], [[22, 22, 22, 22, 22, 22, 22, 22, 15, 15, 22]], [[15]*11], [[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4]], [[32]*11], 
#                 [[19, 19, 31, 31, 31, 31, 31, 31, 31, 31, 18]], [[26]*11], [[28]*11, [8]*11], [[3]*11], [[8]*11, [1]*11], [[5]*11], [[8]*11], [[9]*11],
#                 [[10]*11], [[1]*11], [[7]*11, [5]*11], [[7, 7, 7, 14, 7, 7, 7, 7, 7, 7, 7], [30, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29]], [[None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
#                 [[16]*11], [[30]*11], [[2, None, 2, 2, 2, 2, 2, 2, None, None, 2]], [[9, 9, 9, 9, 9, 9, 9, 9, 9, 2, 2]], [[7]*11], [[15]*11], [[8]*11],
#                 [[11]*11], [[1]*11], [[12]*11], [[32]*11], [[14]*11], [[1, 1, 1, 1, 1, 1, 1, None, 15, 15, 1]], [[22, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]],
#                 [[3]*11], [[15, 15, None, 15, 15, 15, 15, 15, 15, 15, 15]], [[13]*11], [[11]*11], [[10]*11], [[6]*11], [[25, 25, 25, 25, 25, 3, 3, 3, 25, 25, 25]],
#                 [[31]*11], [[7]*11], [[15]*11], [[5]*11], [[3]*11], [[2]*11], [[16]*11], [[7]*11], [[3]*11], [[15]*11], [[17]*11], [[6, 6, 6, None, None, 6, 6, 6, 6, 6, 4]],
#                 [[11]*11], [[34]*11], [[4]*11], [[None, None, None, None, 1, 1, 1, 1, 1, 1, 1]], [[8]*11], [[19]*11], [[18, 18, 18, 11, 11, 11, 18, 18, None, 18, 18]],
#                 [[35]*11], [[None, None, None, None, 18, 12, 12, 12, 12, 12, 12]], [[8, None, 8, 8, 8, 8, 8, 8, 8, 8, 8]], [[25]*11], [[3]*11], [[6]*11],
#                 [[None]*11], [[17]*11], [[3, 3, 3, 22, 22, 22, 22, 22, 22, 22, 22]], [[10]*11, [20, 20, 20, 20, 20, 20, 28, 28, 28, 28, 28], [28]*11],
#                 [[2]*11], [[10]*11, [24]*11], [[12]*11], [[5, 5, 5, None, None, 5, 5, 5, 5, 5, 5]], [[1]*11], [[23]*11], [[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, None]],
#                 [[1]*11, [22, 22, 22, 22, 22, 22, 22, 22, None, None, 22]], [[8]*11], [[2]*11], [[2]*11], [[5]*11], [[6]*11], [[9]*11], [[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 14]],
#                 [[9]*11], [[1]*11], [[5]*11], [[17]*11], [[13]*11], [[19]*11], [[6]*11], [[4]*11, [14, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]], [[9]*11],
#                 [[8]*11], [[19]*11, [19]*11], [[3, 3, 3, 3, 3, 3, 3, None, 10, 7, None]], [[5, 5, 5, 5, 5, 5, 5, 12, 12, 12, 12]], [[5]*11], [[32]*11],
#                 [[9, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]]]


#Sorting the list in alphabetical order in terms of filename:
#bounding_box_ids[5], bounding_box_ids[6] = bounding_box_ids[5], bounding_box_ids[6]
parent_dir_vids = "/itf-fi-ml/home/olekrus/master/master/Data/own_dataset/videos/"
parent_dir_labels = "/itf-fi-ml/home/olekrus/master/master/Data/own_dataset/labels_local_grid/"
parent_dir_bounding_boxes =  "/itf-fi-ml/home/olekrus/master/master/Data/own_dataset/bounding_boxes/Txt/"

video_files = sorted(os.listdir(parent_dir_vids)) #Alphabetical order
label_files = sorted(os.listdir(parent_dir_labels)) #alphabetical order
bounding_box_files = sorted(os.listdir(parent_dir_bounding_boxes)) #alphabetical order


# Define pose models dictionary - using pose-specific models
best_model = "/itf-fi-ml/home/olekrus/Deep-EIoU/runs/pose/train54/weights/best.pt"
model = YOLO(best_model)
conf_threshold_kp = 0.85 #Confidence score threshold for keypoints 

speeds = []
gt_speeds = []
for file_ind in range(len(video_files)):
    video = parent_dir_vids + video_files[file_ind]
    label = parent_dir_labels + label_files[file_ind]
    bounding_box_file = bounding_box_files[file_ind]

    filename = os.path.splitext(os.path.basename(video))[0]
    print(filename)
    bounding_box_id = bounding_box_ids[filename] 

    with open(label, 'r') as file:
        data = json.load(file)

    stadium_dim = data["metadata"]["team_home"]["stadium_dim"]
    height = data["media_attributes"]["height"]
    width = data["media_attributes"]["width"]

    position_list = calc_position_list(stadium_dim[0], stadium_dim[1])
    dst_keypoints, scale_x, scale_y = scale_keypoint_grid(position_list, dst_keypoints)

    plt.figure()
    # plt.scatter(keypoints[:, 0], keypoints[:, 1], color='red', label='Keypoints')  # Plot keypoints in red
    plt.scatter(dst_keypoints[:,0], dst_keypoints[:,1], color='blue', label='Keypoints')
    plt.savefig("/itf-fi-ml/home/olekrus/master/master/Data/results/keypoints/" + video_files[file_ind] + ".png")

    for i in range(len(bounding_box_id)):
        gt_speed = data["momentum_tackler"][i]["speed"]
        frame_start = data["events"][i]["frame_start"] - 10
        frame_end = data["events"][i]["frame_start"]

        start_index = next((j for j in range(len(bounding_box_id[i])) if bounding_box_id[i][j] is not None), None)
        end_index = next((j for j in reversed(range(len(bounding_box_id[i]))) if bounding_box_id[i][j] is not None), None)

        if start_index is None or end_index is None:
            print("Player not detected")
            continue

        start_id = bounding_box_id[i][start_index]
        end_id = bounding_box_id[i][end_index]

        frame_start += start_index
        frame_end -= 10 - end_index 

        if frame_start != frame_end:
            frames = extract_frame(video, frame_start, frame_end)
            start_frame_results, end_frame_results = model(frames[0]), model(frames[-1]) 
        else: 
            print("We don't have two frames to calculate speed!")
            continue
        box_start, box_end = start_frame_results[0].boxes.xywh, end_frame_results[0].boxes.xywh
        box_start, box_end = box_start.flatten().cpu().numpy(), box_end.flatten().cpu().numpy()

        conf_scores_box_start, conf_scores_box_end = start_frame_results[0].boxes.conf.cpu().numpy().flatten(), end_frame_results[0].boxes.conf.cpu().numpy().flatten()
        if conf_scores_box_start.size == 0 or conf_scores_box_end.size == 0:
            print("No field detected!")
            continue

        start_box_ind = np.argmax(conf_scores_box_start)
        end_box_ind = np.argmax(conf_scores_box_end)

        confidence_scores_start, confidence_scores_end = start_frame_results[0].keypoints.conf, end_frame_results[0].keypoints.conf
        confidence_scores_start, confidence_scores_end = confidence_scores_start.cpu().numpy().flatten(), confidence_scores_end.cpu().numpy().flatten()

        keypoints_start, keypoints_end = start_frame_results[0].keypoints.xy, end_frame_results[0].keypoints.xy
        keypoints_start, keypoints_end = keypoints_start.cpu().numpy().squeeze(), keypoints_end.cpu().numpy().squeeze()

        if len(conf_scores_box_start) > 1:
            keypoints_start = keypoints_start[start_box_ind]
            confidence_scores_start = confidence_scores_start[39*start_box_ind:39*(start_box_ind + 1)]
            
        if len(conf_scores_box_end) > 1:
            keypoints_end = keypoints_end[end_box_ind]
            confidence_scores_end = confidence_scores_end[39*end_box_ind:39*(end_box_ind + 1)]
 
        for i in range(len(confidence_scores_start)):
            if confidence_scores_start[i] < conf_threshold_kp:
                keypoints_start[i, :] = 0

        for i in range(len(confidence_scores_end)):
            if confidence_scores_end[i] < conf_threshold_kp:
                keypoints_end[i, :] = 0

        condition_start = np.all(keypoints_start != 0, axis=1)
        condition_end = np.all(keypoints_end != 0, axis=1)

        non_zero_indices_start = np.where(condition_start)[0]
        non_zero_indices_end = np.where(condition_end)[0]

        keypoints_start = keypoints_start[condition_start]
        keypoints_end = keypoints_end[condition_end]

        dst_keypoints_start = dst_keypoints[non_zero_indices_start]
        dst_keypoints_end = dst_keypoints[non_zero_indices_end]

        x_start, y_start, w_start, h_start = box_start[0:4]
        x1_start, y1_start, x2_start, y2_start = x_start - w_start/2, y_start - h_start/2, x_start + w_start/2, y_start + h_start/2
        x_end, y_end, w_end, h_end = box_end[0:4]
        x1_end, y1_end, x2_end, y2_end = x_end - w_end/2, y_end - h_end/2, x_end + w_end/2, y_end + h_end/2


        keypoints_start_plot = keypoints_start.copy() #To see how well the keypoints are predicted
        keypoints_end_plot = keypoints_end.copy()

        keypoints_start[:,0] = keypoints_start[:,0] - x1_start
        keypoints_start[:,1] = keypoints_start[:,1] - y1_start
        keypoints_end[:,0] = keypoints_end[:,0] - x1_end
        keypoints_end[:,1] = keypoints_end[:,1] - y1_end

        # plt.figure()
        # plt.imshow(frames[0])
        # # plt.scatter(keypoints[:, 0], keypoints[:, 1], color='red', label='Keypoints')  # Plot keypoints in red
        # plt.scatter(dst_keypoints_start[:,0], dst_keypoints_start[:,1], color='blue', label='Keypoints')
        # plt.savefig("/itf-fi-ml/home/olekrus/master/master/Data/results/keypoints/" + video_files[file_ind] + f"_{i}.png")


        dst_keypoints_start[:,0] = dst_keypoints_start[:,0]*w_start/1280 
        dst_keypoints_start[:,1] = dst_keypoints_start[:,1]*h_start/720
        dst_keypoints_end[:,0] = dst_keypoints_end[:,0]*w_end/1280 
        dst_keypoints_end[:,1] = dst_keypoints_end[:,1]*h_end/720


        if keypoints_start.shape[0] >= 4 and keypoints_end.shape[0] >= 4:
            matrix_start, mask_start = cv2.findHomography(keypoints_start, dst_keypoints_start, cv2.RANSAC)
            matrix_end, mask_end = cv2.findHomography(keypoints_end, dst_keypoints_end, cv2.RANSAC)
        else: 
            print("Not enough keypoints!")
            continue

        if matrix_start is None or matrix_end is None or keypoints_start.shape[0] <= 4 or keypoints_end.shape[0] <= 4:
            print("Homography matrix computation failed!")
        else:
            start_point = get_xy_from_txt(parent_dir_bounding_boxes + bounding_box_file, frame_start, start_id)
            end_point = get_xy_from_txt(parent_dir_bounding_boxes + bounding_box_file, frame_end, end_id)

            start_point_homogenous = np.array([start_point[0], start_point[1], 1])
            end_point_homogenous = np.array([end_point[0], end_point[1], 1])

            transformed_start_point = matrix_start.dot(start_point_homogenous)
            transformed_end_point = matrix_end.dot(end_point_homogenous)
            
            # Convert back to Cartesian coordinates by dividing by the homogeneous coordinate
            xy_start = np.array([transformed_start_point[0] / transformed_start_point[2], transformed_start_point[1] / transformed_start_point[2]])
            xy_end = np.array([transformed_end_point[0] / transformed_end_point[2], transformed_end_point[1] / transformed_end_point[2]])


            dist_travelled = (xy_start - xy_end)/np.array([scale_x, scale_y])

            print(video)
            speed = np.linalg.norm(dist_travelled)/((end_index - start_index)*1/25)
            print("Speed: ", speed, "gt: ", gt_speed)

            if speed < 13: #we don't want the impossibly high speeds
                speeds.append(speed)
                gt_speeds.append(gt_speed)



            plt.figure()
            plt.imshow(frames[0])
            # plt.scatter(keypoints[:, 0], keypoints[:, 1], color='red', label='Keypoints')  # Plot keypoints in red
            plt.scatter(keypoints_start_plot[:, 0], keypoints_start_plot[:,1], color='blue', label='Keypoints')
            plt.savefig("/itf-fi-ml/home/olekrus/master/master/Data/results/" + video_files[file_ind] + f"_0_{i}.png")


            plt.figure()
            plt.imshow(frames[-1])
            # plt.scatter(keypoints[:, 0], keypoints[:, 1], color='red', label='Keypoints')  # Plot keypoints in red
            plt.scatter(keypoints_end_plot[:, 0], keypoints_end_plot[:, 1], color='blue', label='Keypoints')
            plt.savefig("/itf-fi-ml/home/olekrus/master/master/Data/results/" + video_files[file_ind] + f"_1_{i}.png")

speeds = np.array(speeds)
gt_speeds = np.array(gt_speeds)
print(len(gt_speeds))
print(f"Avg relative error: {np.sum(abs(speeds - gt_speeds)/gt_speeds)/len(gt_speeds)*100}%")
print(f"Root mean squared error (RMSE): {np.sqrt(1/len(gt_speeds)*np.sum((speeds - gt_speeds)**2))}")
print(f"Mean of gt: {np.mean(gt_speeds)}")



