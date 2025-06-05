import numpy as np
import json
from pathlib import Path 
import os 
from coords_to_pos import *
#from speed_extraction import *
import matplotlib.patches as patches
from funcs import *
import matplotlib.pyplot as plt

bounding_box_ids = {"1738_avxeiaxxw6ocr-main": [[None, None, None, None, None, 67, 67, 67, 67, 67, 67]], 
                    "1738_aw3kercrl5skd-main": [[10]*11], 
                    "1742_ab2m9r2lve8xx-main": [[2]*11], 
                    "1744_ai0bkwrj1o6d3-main": [[40]*11], 
                    "1747_aas7m6x9khzm6-main": [[None, None, None, 14, 14, 14, 14, 14, 14, 14, None]],
                    "1757_aaeaaztoinvzx-main": [[67]*11], 
                    "1757_aebqf8o3tg8wm-main": [[101]*11], 
                    "1765_ayt7hdvmhndz1-main": [[7]*11], 
                    "1767_ak7ro2ewn2hen-main": [[5]*11], 
                    "1769_akarclpi8dv1z-main": [[9]*11, [9]*11], 
                    "1771_au5v9dd1iya1u-main": [[20]*11], 
                    "1773_a5k6fgshvjmvl-main": [[4]*11], 
                    "1776_as3pvx4kblaoy-main": [[3]*11], 
                    "1777_akd6w6rwzvbnw-main": [[5]*11], 
                    "1783_akxegh8qnmh0c-main": [[78]*11],
                    "1784_a309lpf9tgfjk-main": [[2, 2, 2, 2, 2, 2, 2, None, None, None, None]], 
                    "1786_aslryruw4a18u-main": [[10]*11], 
                    "1814_abujpl1vbmlpv-main": [[1]*11], 
                    "1814_av54fv6tckm63-main": [[None]*11], 
                    "1816_alz6d3trmjh1f-main": [[4]*11], 
                    "1821_ah8wqifyc7s7d-main": [[14, 14, 8, 8, 8, 8, 8, 8, 8, 53, 53]],
                    "1825_aq5qrn71iw73p-main": [[6, 6, 6, 6, 6, 6, 6, 6, 6, None, 19]], 
                    "1827_aa7wtwlt86673-main": [[33]*11], 
                    "1834_amoai9g7gbzs3-main": [[85]*11], 
                    "1837_agt9x37n1f90h-main": [[87]*11], 
                    "1845_a6p35pr2oc066-main": [[14]*11], 
                    "1845_aa29zru0nnbn-main": [[31]*11], 
                    "1845_alj72ynli7zbn-main": [[26]*11], 
                    "1858_akkzn5c9u4u1b-main": [[3]*11], 
                    "1862_a8rmubj0ok7zm-main": [[4]*11], 
                    "1871_aieelpjpc384t-main": [[51]*11],
                    "1875_ajbz8ndcnzars-main": [[3, 3, 3, 3, 3, 3, 3, 3, None, None, None]], 
                    "1877_aey3e3ta279ce-main": [[15]*11], 
                    "1902_a39ztm4o3824v-main": [[6]*11], 
                    "1915_aoui433sscrt2-main": [[14]*11], 
                    "1959_arh0h7tsjh69c-main": [[9]*11], 
                    "1962_a6kcrd3r8ovs-main": [[None]*11], 
                    "1968_agc6a87zu9ii1-main": [[1]*11], 
                    "1975_axao1rs21074-main": [[8]*11], 
                    "3181_agkzjzgidsgd2-main": [[None]*11],
                    "3181_agvu1d4f8ybf3-main": [[7]*11], 
                    "3182_a4vdmqg3cwuyn-main": [[71]*11], 
                    "3183_adg5id5325cuv-main": [[2]*11], 
                    "3183_akkgiec7wkss7-main": [[1]*11], 
                    "3185_ac7aak7jrcxm8-main": [[15, 15, 15, 15, 15, 15, 15, 15, 36, 36, 36]], 
                    #"3185_as2w8o9cprai7-main": [[None]*11], 
                    "3188_a4h5fuedm2xzr-main": [[None]*11], 
                    "3194_aaqhazydn9id8-main": [[None]*11], 
                    "3197_a7mq9mlnjzq1x-main": [[35, 35, 35, 35, 35, 35, 35, 35, None, None, None]],
                    "3198_a90f88xq8t1ut-main": [[None]*11], 
                    "3201_ab79q1e5zh9cd-main": [[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, None]], 
                    "3206_a5wjsg0bq1ngj-main": [[8]*11], 
                    "3208_a4d6n4uu08j6p-main": [[None]*11], 
                    "3211_aq2zvwssruck4-main": [[2]*11, [36]*11], 
                    "3212_aet2zo7nas12j-main": [[4]*11], 
                    "3213_awytedvil97mu-main": [[13]*11], 
                    "3217_a81f6sk050e72-main": [[23]*11], 
                    "3218_agffo3bhwew2y-main": [[7]*11], 
                    "3219_ayq7zm7excchf-main": [[5]*11], 
                    "3220_avplqkhgzsp0a-main": [[None, None, None, None, None, None, 73, None, None, None, None]],
                    "3221_aows07a24hzso-main": [[None, None, None, 15, 30, 30, 30, 30, 30, 30, 30]], 
                    "3223_aoat0rpbec3jz-main": [[58]*11], 
                    "3223_aohfwf10lp1dg-main": [[2]*11, [20]*11], 
                    "3227_ackegq4w6hv6-main": [[1]*11], 
                    "3227_alflphjllmv60-main": [[15]*11, [9]*11], 
                    "3227_as3pxmkpfaq5p-main": [[3]*11], 
                    "3227_avm4p96cuq29b-main": [[3]*11],
                    "3230_akfiynywg8rc9-main": [[None, None, None, None, 2, 2, 2, 2, 2, None, None]], 
                    "3236_asnnfeyppx4bs-main": [[4]*11], 
                    "3237_aqmh3owxt46ji-main": [[10]*11], 
                    "3239_a96rowrzxa9kl-main": [[5]*11, [18]*11], 
                    "3239_awlqa708ufga8-main": [[3]*11, [34]*11], 
                    "3243_akkx7abi0qnwy-main": [[6]*11], 
                    "3246_a7q2of9a0pd6a-main": [[31]*11],
                    "3246_ahrs3qyzhc04m-main": [[50]*11], 
                    "3247_a7rii7bfxn52t-main": [[None]*11], 
                    "3250_auw8h8bpm7vab-main": [[4]*11], 
                    "3251_asofekemaqjtq-main": [[7]*11], 
                    "3252_a3bxtm2ke0cl1-main": [[14]*11], 
                    #"3255_acmh6ytfvezfu-main": [[None]*11], 
                    "3265_ah78uxdwg7ol0-main": [[18]*11], 
                    "3266_ag95e7qiyar9f-main": [[6]*11], 
                    "3268_ak8tp20ta0usv-main": [[10]*11], 
                    "3269_acvc2894kckby-main": [[None, None, None, None, None, None, None, 65, 65, 65, 65]],
                    "3270_ak3og7x638lfd-main": [[4]*11], 
                    "3271_ahjx9rzchewnj-main": [[9]*11], 
                    "3275_ae6yqpy36wawy-main": [[9]*11], 
                    "3276_aq0vkqdnewz3j-main": [[78]*11], 
                    "3278_av2482qp96f94-main": [[None]*11], 
                    "3279_appg029alt88v-main": [[7]*11], 
                    "3284_ao7b1fdw9gcio-main": [[3]*11], 
                    "3284_aqvzrpx4l4ug8-main": [[20]*11], 
                    "3287_aebn04cxvaya4-main": [[14]*11], 
                    "3291_am1fnjz8bhv73-main": [[44]*11], 
                    "3295_a7p42zcq5gp9k-main": [[32]*11], 
                    "3295_als72y8ksh3d0-main": [[14]*11],
                    "3299_a7pdtki1rirvk-main": [[13]*11], 
                    "3299_ajqg5w0xt2tzd-main": [[14]*11], 
                    "3300_ae2o2j2s0ghdh-main": [[6]*11], 
                    "3309_adip6nafqmwt2-main": [[62]*11], 
                    "3313_ax2oaplec7ypx-main": [[14]*11], 
                    "3313_axcep3pyh09y0-main": [[3]*11], 
                    "3318_a6kvvchxp3n02-main": [[None]*11], 
                    "3323_aqc0fw2i8km9q-main": [[35]*11], 
                    "3324_ayikh1kn1jf1-main": [[3]*11], 
                    "3325_ahi4gu1oo5u7g-main": [[None]*11], 
                    "3326_av1hfseysxvlz-main": [[6]*11],
                    "3329_a5g9fv7h61tyu-main": [[62]*11],
                    "3332_ai3tt4k6dhczb-main": [[43]*11], 
                    "3334_ab2jsop8zqw3p-main": [[3]*11], 
                    "3338_a8liptbgsg8hl-main": [[None]*11], 
                    "3338_aaji73b5nainn-main": [[None]*11], 
                    "3338_al8kgl5ooyk7a-main": [[3]*11], 
                    "3342_ac38qpotfv65y-main": [[None]*11], 
                    "3342_atosyyiv9wnhn-main": [[7]*11], 
                    #"3343_aragn0yk1ipap-main": [[None]*11], 
                    "3348_ad1c3funotaad-main": [[52]*11], 
                    "3351_ays4y42mdh0bf-main": [[7]*11], 
                    "3353_a42b7leuixe38-main": [[9]*11],
                    "3354_awur9crbaqckv-main": [[None, None, None, 11, 11, 11, 11, 11, 11, 11]], 
                    "3355_am3m8v45ee0fi-main": [[20, 20, None, 36, 36, 36, 36, 36, 36, 36, 36]], 
                    "3355_atu85xsbh0xoc-main": [[11]*11], 
                    "3364_ar3n9z5v3atdy-main": [[15, 15, 15, 15, 15, 15, 15, 15, 15, 15, None], [None]*11, [40, 40, 40, None, None, None, None, None, None, None, None]],
                    "3366_akb38710r8kyc-main": [[13]*11], 
                    "3370_aabltdwvl604m-main": [[None, None, None, None, None, None, 71, 71, 71, 71, 71], [71]*11], 
                    "3370_ae6qndhmjt9wt-main": [[None]*11], 
                    "3370_ava4o66x5yt3e-main": [[13]*11], 
                    "3375_ac0r17bq4c97h-main": [[2]*11], 
                    "3375_akdgamca3n5me-main": [[25]*11], 
                    "3377_abstqd3kdmelk-main": [[6]*11],
                    "3383_ab9fy58to0fue-main": [[2]*11, [15]*11], 
                    "3383_ai9saev3ivyte-main": [[39]*11], 
                    "3385_akcz43ge3mpec-main": [[3]*11], 
                    "3386_a47m0uqqg8f5e-main": [[12]*11], 
                    "3389_a7a5u3ihj8v9t-main": [[36]*11], 
                    "3389_ahom8t5szkj7-main": [[16]*11], 
                    "3389_ammj6l4t44a04-main": [[3]*11], 
                    "3391_aimiuhf1beuti-main": [[4]*11], 
                    "3392_aise66a88lgxp-main": [[9]*11], 
                    "3392_aut0dhwa5iaxi-main": [[2]*11], 
                    "3393_armwr1xhhvydt-main": [[42]*11], 
                    "3396_aih1r69jxq58z-main": [[10]*11], 
                    "3398_awulwqe6gqod-main": [[14]*11],
                    "3399_aolb2vfy7lf3x-main": [[12]*11], 
                    "3402_aj28827ic4y4z-main": [[16]*11], 
                    "3405_avjr7cotyfbtf-main": [[4]*11, [3]*11], 
                    "3406_ad79v02olrv6h-main": [[None]*11], 
                    "3409_aik0pl1ylihuz-main": [[3]*11], 
                    "3410_aqq8aeahwsxe-main": [[21, None, None, None, 8, 8, 8, 8, 8, 8, 8], [8]*11], 
                    "3411_aaf40mb2t015p-main": [[3]*11], 
                    "3412_a9863uj1lxmbq-main": [[4]*11], 
                    "3413_asok8wqxa4b5-main": [[8]*11], 
                    "3415_anub7qf1erg12-main": [[None]*11], 
                    "3417_ay8cjudnvafzl-main": [[None]*11]}



bbox_dir = "/itf-fi-ml/home/olekrus/master/master/dribbling-detection-pipeline/outputs/"
parent_dir_labels = "/itf-fi-ml/home/olekrus/master/master/Data/own_dataset/labels_local_grid/"
parent_dir_vids = "/itf-fi-ml/home/olekrus/master/master/Data/own_dataset/videos/"

speeds = []
gt_speeds = []
for dir in os.listdir(bbox_dir):
    interpolated_dir = bbox_dir + dir + "/interpolated-predictions/"
    dir_name = [dir for dir in Path(interpolated_dir).iterdir()][0]
    file_path = Path(str(dir_name) + "/Labels-GameState.json")
    vid = str(dir_name.name)

    with file_path.open("r") as file:
        bbox_data = json.load(file)

    bounding_box_id = bounding_box_ids[vid]

    label = parent_dir_labels + vid 
    label = label.replace("-main", "") + ".json"


    with open(label, 'r') as file:
        data = json.load(file)

    stadium_dim = data["metadata"]["team_home"]["stadium_dim"]
    height = data["media_attributes"]["height"]
    width = data["media_attributes"]["width"]

    for i in range(len(bounding_box_id)):
        print(vid)
        fps = 25
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

        frame_start += start_index + 1
        frame_end -= 10 - end_index - 1

        if frame_start == frame_end:
            print("We don't have two frames to calculate speed!")
            continue


        annotations = bbox_data["annotations"]

        image_start_id = f"{frame_start:06d}"
        image_end_id = f"{frame_end:06d}"

        annotation_start = find_annotation(image_start_id, start_id, annotations)
        annotation_end = find_annotation(image_end_id, end_id, annotations)

        start_position = np.array([annotation_start["bbox_pitch"]["x_bottom_middle"], annotation_start["bbox_pitch"]["y_bottom_middle"]])
        end_position = np.array([annotation_end["bbox_pitch"]["x_bottom_middle"], annotation_end["bbox_pitch"]["y_bottom_middle"]])

        dist_travelled = end_position - start_position
        speed = np.linalg.norm(dist_travelled)/((end_index - start_index)*1/25)
        print("Speed: ", speed, "gt: ", gt_speed)
        
        if speed < 13: #we don't want the impossibly high speeds
            speeds.append(speed)
            gt_speeds.append(gt_speed)
        else:
            print("Speed exceeds 13 m/s!")

speeds = np.array(speeds)
gt_speeds = np.array(gt_speeds)
print(len(gt_speeds))
print(f"Avg. relative error homography: {np.sum(abs(speeds - gt_speeds)/gt_speeds)/len(gt_speeds)*100}%")
print(f"Root mean squared error (RMSE): {np.sqrt(1/len(gt_speeds)*np.sum((speeds - gt_speeds)**2))}")
print(f"Mean of gt: {np.mean(gt_speeds)}")


lines = ["Side line left", "Side line top", "Side line right", "Side line bottom", "Big rect. left top", "Big rect. left main",
         "Big rect. left bottom", "Big rect. right top", "Big rect. right main", "Big rect. right bottom", "Small rect. left top",
         "Small rect. left main", "Small rect. left bottom", "Small rect. right top", "Small rect. right main", "Small rect. right bottom",
         "Middle line", "Circle central", "Circle left", "Circle right"]

speeds = []
gt_speeds = []
start_count = 0
end_count = 0
both_count = 0
for dir in os.listdir(bbox_dir):
    interpolated_dir = bbox_dir + dir + "/interpolated-predictions/"
    dir_name = [dir for dir in Path(interpolated_dir).iterdir()][0]
    file_path = Path(str(dir_name) + "/Labels-GameState.json")
    vid = str(dir_name.name)

    with file_path.open("r") as file:
        bbox_data = json.load(file)

    bounding_box_id = bounding_box_ids[vid]

    label = parent_dir_labels + vid 
    label = label.replace("-main", "") + ".json"

    video = parent_dir_vids + vid + ".mp4"

    with open(label, 'r') as file:
        data = json.load(file)

    stadium_dim = data["metadata"]["team_home"]["stadium_dim"]
    height_vid = data["media_attributes"]["height"]
    width_vid = data["media_attributes"]["width"]

    for i in range(len(bounding_box_id)):
        print(vid)
        real_start_pos = None
        real_end_pos = None 
        
        fps = 25
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

        frame_start += start_index + 1
        frame_end -= 10 - end_index - 1

        if frame_start == frame_end:
            print("We don't have two frames to calculate speed!")
            continue


        height_pnl = bbox_data["images"][0]["height"]
        width_pnl = bbox_data["images"][0]["width"]

        annotations = bbox_data["annotations"]

        image_start_id = f"{frame_start:06d}"
        image_end_id = f"{frame_end:06d}"

        lines_start = find_lines(image_start_id, annotations)
        lines_end = find_lines(image_end_id, annotations)

        if lines_start is None or lines_end is None:
            print("Couldn't find any lines")
            continue

        possible_squares_start, keypoint_numbers_start = find_possible_squares(lines_start.keys())
        possible_squares_end, keypoint_numbers_end = find_possible_squares(lines_end.keys())

        annotation_start = find_annotation(image_start_id, start_id, annotations)
        annotation_end = find_annotation(image_end_id, end_id, annotations)

        start_x = annotation_start["bbox_image"]["x"]
        start_y = annotation_start["bbox_image"]["y"]
        start_w = annotation_start["bbox_image"]["w"]
        start_h = annotation_start["bbox_image"]["h"]
        start_position = np.array([(start_x + start_w/2)*width_vid/width_pnl, (start_y + start_h)*height_vid/height_pnl])

        end_x = annotation_end["bbox_image"]["x"]
        end_y = annotation_end["bbox_image"]["y"]
        end_w = annotation_end["bbox_image"]["w"]
        end_h = annotation_end["bbox_image"]["h"]
        end_position = np.array([(end_x + end_w/2)*width_vid/width_pnl, (end_y + end_h)*height_vid/height_pnl])

        position_list = calc_position_list_extra(stadium_dim[0], stadium_dim[1]) 
        #print(possible_squares_start)
        #print(possible_squares_end)
        start_count += len(possible_squares_start) > 0
        end_count += len(possible_squares_end) > 0
        both_count += (len(possible_squares_start) > 0 and len(possible_squares_end) > 0)
        for i, square in enumerate(possible_squares_start):
            if len(square) == 4:
                try:
                    point_1, point_2 = lines_start[lines[square[0]]][0], lines_start[lines[square[0]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l1 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_start[lines[square[1]]][0], lines_start[lines[square[1]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l2 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_start[lines[square[2]]][0], lines_start[lines[square[2]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l3 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_start[lines[square[3]]][0], lines_start[lines[square[3]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l4 = increase_and_constant(point_1, point_2)

                    P1, P2, P3, P4 = find_corner_points(l1, l2, l3, l4)

                    _, _, _, _, d1, d2 = calc_lines_and_diag(P1, P2, P3, P4)
                    CP1, CP2, CP3, MP1, MP2, MP3, MP4, m1, m2 = find_cross_midpoints_and_midlines(l1, l2, l3, l4, d1, d2)
                    R, C = find_quadrant(start_position, P1, P2, P3, P4, CP1, CP2, CP3)

                    keypoints = keypoint_numbers_start[i]
                    x_len = abs(position_list[keypoints[1] - 1][0] - position_list[keypoints[0] - 1][0])
                    y_len = abs(position_list[keypoints[2] - 1][1] - position_list[keypoints[0] - 1][1])

                    real_start_pos = find_pos(start_position, R, C, x_len, y_len, P3, P4, l1, l2, l3, l4, m1, m2)
                    x_displacement, y_displacement = position_list[keypoints[0] - 1] #Point P1 is "origin" of perspective grid
                    #print(real_start_pos)
                    real_start_pos[0], real_start_pos[1] = real_start_pos[0] + x_displacement, y_displacement - real_start_pos[1] 
                    #print(P1, P2, P3, P4, start_position)
                    #print(real_start_pos)

                    x = np.linspace(0, 1500, 500)

                    if frame_start != frame_end:
                        frames = extract_frame(video, frame_start - 1, frame_end - 1)
                        start_frame_results, end_frame_results = model(frames[0]), model(frames[-1]) 
                    else: 
                        print("We don't have two frames to calculate speed!")
                        continue
                    print(frames)
                    # Calculate y values for each line
                    # plt.figure()
                    # plt.imshow(frames[0])

                    # y1 = l1[0] * x + l1[1]
                    # y2 = l2[0] * x + l2[1]
                    # y3 = l3[0] * x + l3[1]
                    # y4 = l4[0] * x + l4[1]
                    # y5 = m1[0] * x + m1[1]
                    # y6 = m2[0] * x + m2[1]
                    # y7 = d1[0] * x + d1[1]
                    # y8 = d2[0] * x + d2[1]

                    # points = np.array([P1, P2, P3, P4, CP1, MP1, MP2, MP3, MP4, start_position])  # Create a numpy array for easy plotting
                    # #print(points)
                    # #plt.scatter(points[:, 0], points[:, 1], color='black', zorder=5, label="Points")

                    # # Plot each line
                    # plt.plot(x, y1, label="l1")
                    # plt.plot(x, y2, label="l2")
                    # plt.plot(x, y3, label="l3")
                    # plt.plot(x, y4, label="l4")
                    # plt.plot(x, y5, label="m1")
                    # plt.plot(x, y6, label="m2")
                    # plt.plot(x, y7, label="d1")
                    # plt.plot(x, y8, label="d2")

                    # # Add labels and title
                    # plt.xlabel("x")
                    # plt.ylabel("y")
                    # plt.title(f"{vid}, {frame_start}")
                    # plt.legend()

                    # plt.savefig("perspgrid.png")
                    break
                except:
                    print("One of the lines is not complete")

            elif len(square) == 2:
                try: 
                    circle_points = lines_start[lines[square[1]]]
                    A, B, C, D, E = find_ellipse(circle_points, height_vid, width_vid)

                    point_1, point_2 = lines_start[lines[square[0]]][0], lines_start[lines[square[0]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    m1 = increase_and_constant(point_1, point_2)

                    x_1, x_2 = find_intersect_ellipse_line(A, B, C, D, E, m1)
                    y_1 = m1[0]*x_1 + m1[1]
                    y_2 = m1[0]*x_2 + m1[1] 

                    if y_1 > y_2:
                        MP2 = np.array([x_1, y_1])
                        MP4 = np.array([x_2, y_2])
                    else:
                        MP4 = np.array([x_1, y_1])
                        MP2 = np.array([x_2, y_2])

                    l3 = find_ellipse_tangent(A, B, C, D, E, MP2)
                    l4 = find_ellipse_tangent(A, B, C, D, E, MP4)

                    CP3 = find_intersection(l3, l4)

                    if frame_start != frame_end:
                        frames = extract_frame(video, frame_start - 1, frame_end - 1)
                        start_frame_results, end_frame_results = model(frames[0]), model(frames[-1]) 
                    else: 
                        print("We don't have two frames to calculate speed!")
                        continue
                    keypoints_start, keypoints_end = start_frame_results[0].keypoints.xy, end_frame_results[0].keypoints.xy
                    
                    if keypoints_start[0][16][0]!=0.0:
                        MP1 = np.array(keypoints_start[0][16])
                        
                        m2 = increase_and_constant(MP1, CP3)
                        CP1 = find_intersection(m1, m2)
                        point_1, point_2 = find_intersect_ellipse_line(A, B, C, D, E, m2)

                        if np.linalg.norm(point_1 - MP1) > np.linalg.norm(point_2 - MP1):
                            MP3 = point_1 
                        else:
                            MP3 = point_2 

                    elif keypoints_start[0][19][0]!=0.0:
                        CP1 = np.array(keypoints_start[0][19])

                        m2 = increase_and_constant(CP1, CP3)
                        point_1, point_2 = find_intersect_ellipse_line(A, B, C, D, E, m2)
                        
                        if point_1[0] > point_2[0]:
                            MP3 = point_1 
                            MP1 = point_2 

                    elif keypoints_start[0][22][0]!=0.0:
                        MP3 = np.array(keypoints_start[0][22])

                        m2 = increase_and_constant(MP3, CP3)
                        CP1 = find_intersection(m1, m2)
                        point_1, point_2 = find_intersect_ellipse_line(A, B, C, D, E, m2)

                        if np.linalg.norm(point_1 - MP3) > np.linalg.norm(point_2 - MP3):
                            MP1 = point_1 
                        else:
                            MP1 = point_2 

                    l1 = find_ellipse_tangent(A, B, C, D, E, MP1)
                    l2 = find_ellipse_tangent(A, B, C, D, E, MP3)

                    P1, P2, P3, P4 = find_corner_points(l1, l2, l3, l4)
                    CP2 = find_intersection(l1, l2)

                    R, C = find_quadrant(start_position, P1, P2, P3, P4, CP1, CP2, CP3)

                    keypoints = keypoint_numbers_start[i]
                    x_len = abs(position_list[keypoints[1] - 1][0] - position_list[keypoints[0] - 1][0])
                    y_len = abs(position_list[keypoints[2] - 1][1] - position_list[keypoints[0] - 1][1])

                    real_start_pos = find_pos(start_position, R, C, x_len, y_len, P3, P4, l1, l2, l3, l4, m1, m2)
                    x_displacement, y_displacement = position_list[keypoints[0] - 1] #Point P1 is "origin" of perspective grid
                    #print(real_start_pos)
                    real_start_pos[0], real_start_pos[1] = real_start_pos[0] + x_displacement, y_displacement - real_start_pos[1] 

                    break
                except:
                    print("Line or ellipse is not complete")

            

        for i, square in enumerate(possible_squares_end):
            if len(square) == 4:
                try:
                    point_1, point_2 = lines_end[lines[square[0]]][0], lines_end[lines[square[0]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l1 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_end[lines[square[1]]][0], lines_end[lines[square[1]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l2 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_end[lines[square[2]]][0], lines_end[lines[square[2]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l3 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_end[lines[square[3]]][0], lines_end[lines[square[3]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l4 = increase_and_constant(point_1, point_2)

                    P1, P2, P3, P4 = find_corner_points(l1, l2, l3, l4)

                    _, _, _, _, d1, d2 = calc_lines_and_diag(P1, P2, P3, P4)
                    CP1, CP2, CP3, MP1, MP2, MP3, MP4, m1, m2 = find_cross_midpoints_and_midlines(l1, l2, l3, l4, d1, d2)
                    R, C = find_quadrant(end_position, P1, P2, P3, P4, CP1, CP2, CP3)

                    keypoints = keypoint_numbers_end[i]
                    x_len = abs(position_list[keypoints[1] - 1][0] - position_list[keypoints[0] - 1][0])
                    y_len = abs(position_list[keypoints[2] - 1][1] - position_list[keypoints[0] - 1][1])

                    real_end_pos = find_pos(end_position, R, C, x_len, y_len, P3, P4, l1, l2, l3, l4, m1, m2)
                    x_displacement, y_displacement = position_list[keypoints[0] - 1] #Point P1 is "origin" of perspective grid
                    #print(real_end_pos)
                    real_end_pos[0], real_end_pos[1] = real_end_pos[0] + x_displacement, y_displacement - real_end_pos[1] 
                    #print(P1, P2, P3, P4, end_position, l1, l2, l3, l4, m1, m2, d1, d2, CP1, CP2, CP3)
                    #print(real_end_pos)

                    # plt.figure()
                    # plt.imshow(frames[-1])

                    # y1 = l1[0] * x + l1[1]
                    # y2 = l2[0] * x + l2[1]
                    # y3 = l3[0] * x + l3[1]
                    # y4 = l4[0] * x + l4[1]
                    # y5 = m1[0] * x + m1[1]
                    # y6 = m2[0] * x + m2[1]
                    # y7 = d1[0] * x + d1[1]
                    # y8 = d2[0] * x + d2[1]

                    # points = np.array([P1, P2, P3, P4, CP1, MP1, MP2, MP3, MP4, end_position])  # Create a numpy array for easy plotting
                    # #print(points)
                    # colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

                    # # Plot each point with a unique color and label
                    # # for i, point in enumerate(points):
                    # #     plt.scatter(point[0], point[1], color=colors[i], label=f"Point {i+1}")
                    # plt.scatter(points[:, 0], points[:, 1], color='black', zorder=5, label="Points")

                    # # Plot each line
                    # plt.plot(x, y1, label="l1")
                    # plt.plot(x, y2, label="l2")
                    # plt.plot(x, y3, label="l3")
                    # plt.plot(x, y4, label="l4")
                    # plt.plot(x, y5, label="m1")
                    # plt.plot(x, y6, label="m2")
                    # plt.plot(x, y7, label="d1")
                    # plt.plot(x, y8, label="d2")

                    # # Add labels and title
                    # plt.xlabel("x")
                    # plt.ylabel("y")
                    # plt.title(f"{vid}, {frame_end}")
                    # plt.legend()

                    # plt.savefig("/itf-fi-ml/home/olekrus/master/master/perspgrid2.png")
                    break

                except:
                    print("One of the lines is not complete")

            elif len(square) == 2:
                try: 
                    circle_points = lines_end[lines[square[1]]]
                    A, B, C, D, E = find_ellipse(circle_points, height_vid, width_vid)

                    point_1, point_2 = lines_end[lines[square[0]]][0], lines_start[lines[square[0]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    m1 = increase_and_constant(point_1, point_2)

                    x_1, x_2 = find_intersect_ellipse_line(A, B, C, D, E, m1)
                    y_1 = m1[0]*x_1 + m1[1]
                    y_2 = m1[0]*x_2 + m1[1] 

                    if y_1 > y_2:
                        MP2 = np.array([x_1, y_1])
                        MP4 = np.array([x_2, y_2])
                    else:
                        MP4 = np.array([x_1, y_1])
                        MP2 = np.array([x_2, y_2])

                    l3 = find_ellipse_tangent(A, B, C, D, E, MP2)
                    l4 = find_ellipse_tangent(A, B, C, D, E, MP4)

                    CP3 = find_intersection(l3, l4)

                    if frame_start != frame_end:
                        frames = extract_frame(video, frame_start - 1, frame_end - 1)
                        start_frame_results, end_frame_results = model(frames[0]), model(frames[-1]) 
                    else: 
                        print("We don't have two frames to calculate speed!")
                        continue
                    keypoints_start, keypoints_end = start_frame_results[0].keypoints.xy, end_frame_results[0].keypoints.xy
                    
                    if keypoints_end[0][16][0]!=0.0:
                        MP1 = np.array(keypoints_end[0][16])
                        
                        m2 = increase_and_constant(MP1, CP3)
                        CP1 = find_intersection(m1, m2)
                        point_1, point_2 = find_intersect_ellipse_line(A, B, C, D, E, m2)

                        if np.linalg.norm(point_1 - MP1) > np.linalg.norm(point_2 - MP1):
                            MP3 = point_1 
                        else:
                            MP3 = point_2 

                    elif keypoints_end[0][19][0]!=0.0:
                        CP1 = np.array(keypoints_end[0][19])

                        m2 = increase_and_constant(CP1, CP3)
                        point_1, point_2 = find_intersect_ellipse_line(A, B, C, D, E, m2)
                        
                        if point_1[0] > point_2[0]:
                            MP3 = point_1 
                            MP1 = point_2 

                    elif keypoints_end[0][22][0]!=0.0:
                        MP3 = np.array(keypoints_end[0][22])

                        m2 = increase_and_constant(MP3, CP3)
                        CP1 = find_intersection(m1, m2)
                        point_1, point_2 = find_intersect_ellipse_line(A, B, C, D, E, m2)

                        if np.linalg.norm(point_1 - MP3) > np.linalg.norm(point_2 - MP3):
                            MP1 = point_1 
                        else:
                            MP1 = point_2 

                    l1 = find_ellipse_tangent(A, B, C, D, E, MP1)
                    l2 = find_ellipse_tangent(A, B, C, D, E, MP3)

                    P1, P2, P3, P4 = find_corner_points(l1, l2, l3, l4)
                    CP2 = find_intersection(l1, l2)

                    R, C = find_quadrant(start_position, P1, P2, P3, P4, CP1, CP2, CP3)

                    keypoints = keypoint_numbers_end[i]
                    x_len = abs(position_list[keypoints[1] - 1][0] - position_list[keypoints[0] - 1][0])
                    y_len = abs(position_list[keypoints[2] - 1][1] - position_list[keypoints[0] - 1][1])

                    real_end_pos = find_pos(end_position, R, C, x_len, y_len, P3, P4, l1, l2, l3, l4, m1, m2)
                    x_displacement, y_displacement = position_list[keypoints[0] - 1] #Point P1 is "origin" of perspective grid
                    #print(real_start_pos)
                    real_end_pos[0], real_end_pos[1] = real_end_pos[0] + x_displacement, y_displacement - real_end_pos[1] 
                    
                    break
                except:
                    print("Line or ellipse is not complete")

        if real_end_pos is not None and real_start_pos is not None:
            dist_travelled = real_end_pos - real_start_pos
            speed = np.linalg.norm(dist_travelled)/((end_index - start_index)*1/25)
            print("Speed: ", speed, "gt: ", gt_speed)
        else: 
            print("Could not find positions!")
            continue
        
        if speed < 13: #we don't want the impossibly high speeds
            speeds.append(speed)
            gt_speeds.append(gt_speed)
        else:
            print("Speed exceeds 13 m/s!")

#print(start_count, end_count, both_count)
speeds = np.array(speeds)
gt_speeds = np.array(gt_speeds)
print(len(gt_speeds))
print(f"Avg. relative error perspective grid: {np.sum(abs(speeds - gt_speeds)/gt_speeds)/len(gt_speeds)*100}%")
print(f"Root mean squared error (RMSE): {np.sqrt(1/len(gt_speeds)*np.sum((speeds - gt_speeds)**2))}")
print(f"Mean of gt: {np.mean(gt_speeds)}")



