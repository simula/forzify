import numpy as np 
#from speed_extraction_pnl_calib import find_annotation, find_lines, find_possible_rectangles, calc_position_list_extra
import os 
from pathlib import Path 
import json 
import re
import matplotlib.pyplot as plt 
from coords_to_pos import *
from funcs import *
from ultralytics import YOLO
from io import StringIO
import trackeval
from trackeval.datasets._base_dataset import _BaseDataset
from collections import defaultdict
from trackeval.metrics import HOTA

class Dataset_for_HOTA(_BaseDataset):
    """
    Dataset class for evaluating the tracking results using HOTA (Higher Order Tracking Accuracy)
    """
    def __init__(self, gt_sio, tracker_dict):
        self.gt_data = self._parse_data(gt_sio) #Ground truth data
        self.tracker_data = {name: self._parse_data(sio) for name, sio in tracker_dict.items()} #Predicted tracklets
        self.name = "MyCustomDataset"
        self.should_classes_combine = False
        self.use_super_categories = False
        self.output_fol = "./trackeval_output"  
        self.output_sub_fol = ""  

    def get_name(self):
        return self.name

    def get_eval_info(self):
        """
        Retrieves the keys for the ground truth and predicted sequences
        """
        sequences = list(self.gt_data.keys()) #List of keys in ground truth data representing the event sequences
        trackers = list(self.tracker_data.keys()) #List of keys in predicted tracklets representing the event sequences
        classes = ["class1"]
        return trackers, sequences, classes

    def get_raw_seq_data(self, tracker, seq):
        """
        Retrieves tracking and ground truth data for a specific event seqence
        """
        gt = self.gt_data.get(seq, [])
        tracker_dets = self.tracker_data.get(tracker, {}).get(seq, [])
        return {'gt': gt, 'tracker': tracker_dets}

    def get_preprocessed_seq_data(self, raw_data, cls):
        """
        Inputs raw data containing ground truth and predicted tracklets, and class, i.e. "class1" or just a player. Retrieves ids, 
        bounding boxes and a similarity matrix, and the number of ids and predictions, the number of timesteps.
        """

        gt_ids = []
        tracker_ids = []
        gt_dets = []
        tracker_dets = []

        #Loop through all the frames in the raw data, and get ids and bounding boxes:
        for frame_id in sorted(raw_data['gt'].keys()):
            gt_frame = raw_data['gt'][frame_id]
            trk_frame = raw_data['tracker'].get(frame_id, [])

            gt_ids.append([d['obj_id'] for d in gt_frame])
            tracker_ids.append([d['obj_id'] for d in trk_frame])

            gt_dets.append([[d['bbox']['x'], d['bbox']['y'], d['bbox']['w'], d['bbox']['h']] for d in gt_frame])
            tracker_dets.append([[d['bbox']['x'], d['bbox']['y'], d['bbox']['w'], d['bbox']['h']] for d in trk_frame])

        similarity_scores = []

        #Go through the bounding boxes of the ground truth and predicted tracklets, and calculate IoU and similarity matrix:
        for gt_frame_dets, tracker_frame_dets in zip(gt_dets, tracker_dets):
            frame_scores = []
            for trk_box in tracker_frame_dets:
                row = []
                for gt_box in gt_frame_dets:
                    iou = self._calculate_iou(trk_box, gt_box)
                    row.append(iou)
                frame_scores.append(row)
            similarity_scores.append(np.array(frame_scores))

        gt_ids_np = [np.array(f) for f in gt_ids]
        tracker_ids_np = [np.array(f) for f in tracker_ids]

        # Flatten id-lists and get only unique ids 
        all_gt_ids = set(obj_id for frame in gt_ids for obj_id in frame)
        all_tracker_ids = set(obj_id for frame in tracker_ids for obj_id in frame)

        # Map ids to indexes in the id-lists
        gt_id_map = {obj_id: idx for idx, obj_id in enumerate(sorted(all_gt_ids))}
        tracker_id_map = {obj_id: idx for idx, obj_id in enumerate(sorted(all_tracker_ids))}

        # Remap ids such that the indexes are ordered
        gt_ids_mapped = [np.array([gt_id_map[obj_id] for obj_id in frame]) for frame in gt_ids]
        tracker_ids_mapped = [np.array([tracker_id_map[obj_id] for obj_id in frame]) for frame in tracker_ids]


        return {
            'gt_ids': gt_ids_mapped,
            'tracker_ids': tracker_ids_mapped,
            'gt_dets': gt_dets,
            'tracker_dets': tracker_dets,
            'num_gt_ids': len(gt_id_map),
            'num_tracker_ids': len(tracker_id_map),
            'num_gt_dets': sum(len(f) for f in gt_dets),
            'num_tracker_dets': sum(len(f) for f in tracker_dets),
            'similarity_scores': similarity_scores,
            'num_timesteps': len(gt_ids_mapped),
        }

    def _parse_data(self, sio):
        """
        Organise tracking data into a dictionary with ids and bounding boxes.
        """
        data = defaultdict(lambda: defaultdict(list))  
        sio.seek(0)
        for line in sio:
            fields = line.strip().split(',')
            frame_id = int(fields[0])
            obj_id = int(fields[1])
            x, y, w, h = map(float, fields[2:6])
            det = {'obj_id': obj_id, 'bbox': {'x': x, 'y': y, 'w': w, 'h': h}}
            data['seq1'][frame_id].append(det)  
        return data

    def _calculate_similarities(self, pred_data, gt_data):
        """
        Compute the similarity matrices between the predictions and ground truth data using IoU.
        """
        similarities = []
        for pred in pred_data:
            row = []
            for gt in gt_data:
                row.append(self._calculate_iou(pred['bbox'], gt['bbox']))
            similarities.append(row)
        return similarities

    def _calculate_iou(self, pred_bbox, gt_bbox):
        """
        Calculates IoU between a predicted bounding box and the corresponding groun truth
        """
        x1, y1, w1, h1 = pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]
        x2, y2, w2, h2 = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union > 0 else 0

    def _load_raw_file(self, file_path):
        """
        Load ground truth data from a file
        """
        return self.gt_data  # Not used if we use SIO

    def get_default_dataset_config(self):
        """
        Returns a default dataset configuration. Here it isnt implemented, meant to be overridden.
        """
        return {}


#Start of events, end is just start + 10 frames
frames_start = {"SNGS-140": [0, 173, 195, 240, 335, 535],
                "SNGS-141": [0, 70, 105, 180, 275, 360, 525],
                "SNGS-142": [120, 170, 225, 350, 565, 600, 625, 660],
                "SNGS-143": [25, 240, 270, 395, 615, 630],
                "SNGS-144": [160, 170, 290, 385],
                "SNGS-146": [15, 60, 100, 170, 275],
                "SNGS-147": [290, 310, 370, 700],
                "SNGS-150": [75, 240, 430, 530],
                "SNGS-187": [50, 280, 280, 330, 395, 520],
                "SNGS-188": [450, 730],
                "SNGS-189": [50, 115, 200, 250, 290, 325, 350, 380],
                "SNGS-190": [25, 110, 155, 235, 365],
                "SNGS-191": [25, 95, 335, 360, 510],
                "SNGS-192": [20, 90, 240, 365, 490],
                "SNGS-193": [160, 225, 330, 450, 510, 595],
                "SNGS-194": [235, 400, 470, 500],
                "SNGS-195": [30, 197, 391, 610],
                "SNGS-196": [140, 290, 360, 425, 635, 710],
                "SNGS-197": [30, 140, 275, 305, 640],
                "SNGS-198": [164, 250, 295, 475, 490, 620],
                "SNGS-199": [155, 250, 337, 565],
                "SNGS-200": [215, 300, 370, 570, 730]}

#The bounding box id of the player we look at in the ground truth
bounding_box_ids_gt = {"SNGS-140": [8, 20, 19, 3, 12, 20],
                       "SNGS-141": [8, 16, 21, 20, 8, 22, 6],
                       "SNGS-142": [12, 10, 21, 20, 19, 23, 24, 14],
                       "SNGS-143": [7, 4, 18, 17, 1, 4],
                       "SNGS-144": [11, 7, 8, 20],
                       "SNGS-146": [6, 1, 12, 2, 12],
                       "SNGS-147": [3, 5, 4, 21],
                       "SNGS-150": [4, 24, 18, 28],
                       "SNGS-187": [8, 2, 6, 8, 5, 1],
                       "SNGS-188": [26, 5],
                       "SNGS-189": [8, 15, 13, 11, 8, 11, 21, 13],
                       "SNGS-190": [19, 3, 1, 20, 10],
                       "SNGS-191": [11, 4, 5, 4, 7],
                       "SNGS-192": [8, 2, 2, 18, 4],
                       "SNGS-193": [1, 11, 1, 11, 14, 3],
                       "SNGS-194": [6, 13, 14, 17],
                       "SNGS-195": [10, 15, 12, 6],
                       "SNGS-196": [3, 14, 3, 6, 3, 16],
                       "SNGS-197": [1, 13, 13, 21, 6],
                       "SNGS-198": [1, 18, 23, 4, 6, 2],
                       "SNGS-199": [13, 20, 23, 3],
                       "SNGS-200": [7, 10, 15, 11, 17]}

#The bounding box id, as taken from the prediction video
bounding_box_ids = {"SNGS-140": [[5]*11, [12]*11, [7]*11, [19]*11, [59]*11, [112]*11],
                    "SNGS-141": [[10]*11, [30]*11, [31]*11, [75]*11, [65]*11, [82]*11, [220]*11],
                    "SNGS-142": [[None]*11, [43]*11, [37]*11, [41, 41, 41, 41, 41, 41, 41, 41, 41, None, None], [200]*11, [154]*11, [149]*11, [199]*11],
                    "SNGS-143": [[3]*11, [118]*11, [110]*11, [109]*11, [197]*11, [196]*11],
                    "SNGS-144": [[8]*11, [21]*11, [11]*11, [None, None, None, None, None, None, 105, 105, 105, 105, 105]],
                    "SNGS-146": [[10]*11, [2]*11, [6, 6, 6, 6, 6, 6, 6, 6, None, None, None], [5]*11, [2]*11],
                    "SNGS-147": [[118]*11, [72]*11, [119]*11, [238]*11],
                    "SNGS-150": [[1]*11, [35]*11, [56]*11, [151]*11],
                    "SNGS-187": [[3]*11, [40]*11, [5]*11, [3]*11, [6]*11, [17]*11],
                    "SNGS-188": [[24]*11, [10]*11],
                    "SNGS-189": [[2]*11, [18]*11, [1]*11, [6]*11, [2]*11, [6]*11, [42]*11, [55]*11],
                    "SNGS-190": [[None]*11, [5]*11, [4]*11, [67]*11, [138]*11],
                    "SNGS-191": [[7]*11, [8]*11, [13]*11, [8]*11, [1]*11],
                    "SNGS-192": [[7]*11, [1]*11, [1]*11, [89]*11, [148]*11],
                    "SNGS-193": [[1]*11, [5]*11, [68]*11, [107]*11, [37]*11, [113]*11],
                    "SNGS-194": [[42]*11, [6]*11, [55]*11, [43]*11],
                    "SNGS-195": [[1]*11, [6]*11, [59]*11, [113]*11],
                    "SNGS-196": [[2]*11, [60]*11, [55]*11, [62]*11, [55]*11, [203]*11],
                    "SNGS-197": [[2]*11, [7]*11, [7]*11, [19]*11, [104]*11],
                    "SNGS-198": [[1]*11, [12]*11, [21]*11, [56]*11, [5]*11, [123]*11],
                    "SNGS-199": [[9]*11, [45]*11, [67]*11, [122]*11],
                    "SNGS-200": [[3]*11, [5]*11, [38]*11, [2]*11, [132]*11]}

bbox_dir = "/itf-fi-ml/home/olekrus/master/master/dribbling-detection-pipeline/outputs_soccernet/"
parent_dir_labels = "/itf-fi-ml/home/olekrus/master/master/soccernet/data/SoccerNetGS/test/"
parent_dir_vids = "/itf-fi-ml/home/olekrus/master/master/soccernet/data/SoccerNetGS/test/"

#The model predicts some additional keypoints, so that we can find perspective grids for example for the center circle
best_model = "/itf-fi-ml/home/olekrus/Deep-EIoU/runs/pose/train54/weights/best.pt"
model = YOLO(best_model)

gt_entries = []
pred_entries = []

number_of_events = 0
speeds = []
gt_speeds = []
absolute_position_error = []
video_and_frame_start_bad_pred = []
video_and_frame_start_good_pred = []
#Go through each video
for dir in os.listdir(bbox_dir):
    interpolated_dir = bbox_dir + dir + "/interpolated-predictions/"
    dir_name = [dir for dir in Path(interpolated_dir).iterdir()][0]
    file_path = Path(str(dir_name) + "/Labels-GameState.json")
    vid = str(dir_name.name)


    with file_path.open("r") as file:
        bbox_data = json.load(file)   #Predicted bounding boxes
    
    label_dir = vid.replace("video_", "").upper()
    label = parent_dir_labels + label_dir
    label += "/Labels-GameState.json"

    bounding_box_id = bounding_box_ids[label_dir] #Name is equal to label_dir value

    with open(label, 'r') as file:
        data = json.load(file) #Ground truth bounding boxes

    stadium_dim = [105, 65]
    height = data["images"][0]["height"]
    width = data["images"][0]["width"]

    for i in range(len(bounding_box_id)):
        number_of_events += 1
        fps = 25

        #Get gt speed
        bounding_box_id_gt = bounding_box_ids_gt[label_dir][i]
        frame_start = frames_start[label_dir][i]
        frame_end = frame_start + 10

        annotations_gt = data["annotations"]
        annotations = bbox_data["annotations"]

        #Format image and video id, to extract the correct ground truth bounding box
        vid_id = re.findall(r'\d+',label_dir)
        image_start_id = f"3{vid_id[0]}{(frame_start + 1):06d}"
        image_end_id = f"3{vid_id[0]}{(frame_end + 1):06d}"

        annotation_gt_start = find_annotation(image_start_id, bounding_box_id_gt, annotations_gt)
        annotation_gt_end = find_annotation(image_end_id, bounding_box_id_gt, annotations_gt)

        #Skip iteration if gt not available
        if annotation_gt_start["bbox_pitch"] == None or annotation_gt_end["bbox_pitch"] == None:
            print("Gt not found.")
            continue

        #Save all bounding box predictions and ground truths
        for frame in range(frame_start, frame_end + 1):
            image_id_gt = f"3{vid_id[0]}{(frame + 1):06d}"
            image_id = f"{(frame + 1):06d}"

            id = bounding_box_id[i][frame - frame_start]

            annotation = find_annotation(image_id, id, annotations)
            annotation_gt = find_annotation(image_id_gt, bounding_box_id_gt, annotations_gt)
            if annotation is not None and annotation_gt is not None:
                if annotation["bbox_image"] is not None and annotation_gt["bbox_image"] is not None:
                    bbox = annotation["bbox_image"]
                    pred_entries.append((frame, id, bbox["x"], bbox["y"], bbox["w"], bbox["h"]))

                    gt_bbox = annotation_gt["bbox_image"]
                    gt_entries.append((frame, bounding_box_id_gt, gt_bbox["x"], gt_bbox["y"], gt_bbox["w"], gt_bbox["h"]))

            elif annotation is None and annotation_gt is not None:
                if annotation["bbox_image"] is None and annotation_gt["bbox_image"] is not None:
                    pred_entries.append((frame, id, -1, -1, -1, -1)) 
                    gt_bbox = annotation_gt["bbox_image"] 
                    gt_entries.append((frame, bounding_box_id_gt, gt_bbox["x"], gt_bbox["y"], gt_bbox["w"], gt_bbox["h"]))

            elif annotation is not None and annotation_gt is None:
                if annotation["bbox_image"] is not None and annotation_gt["bbox_image"] is None:
                    bbox = annotation["bbox_image"]
                    pred_entries.append((frame, id, bbox["x"], bbox["y"], bbox["w"], bbox["h"])) 
                    gt_entries.append((frame, bounding_box_id_gt, -1, -1, -1, -1))
                    
            elif annotation is None and annotation_gt is None: 
                if annotation["bbox_image"] is None and annotation_gt["bbox_image"] is None:
                    pred_entries.append((frame, id, -1, -1, -1, -1)) 
                    gt_entries.append((frame, bounding_box_id_gt, -1, -1, -1, -1))
                    
        #Extract ground truth position from the annotations        
        gt_start_pos = np.array([annotation_gt_start["bbox_pitch"]["x_bottom_middle"], annotation_gt_start["bbox_pitch"]["y_bottom_middle"]])
        gt_end_pos = np.array([annotation_gt_end["bbox_pitch"]["x_bottom_middle"], annotation_gt_end["bbox_pitch"]["y_bottom_middle"]])

        #Calculate ground truth speed
        dist_travelled = gt_end_pos - gt_start_pos
        gt_speed = np.linalg.norm(dist_travelled)/(10/25)


        #Predicted speed:
        #Find indices of sequence that are not None
        start_index = next((j for j in range(len(bounding_box_id[i])) if bounding_box_id[i][j] is not None), None)
        end_index = next((j for j in reversed(range(len(bounding_box_id[i]))) if bounding_box_id[i][j] is not None), None)

        #If there are no bounding box detections, skip to next iteration
        if start_index is None or end_index is None:
            print("Player not detected")
            continue

        #Get bounding box id of start and end of sequence (they can have changed, but it is possible to do a run where that is not allowed)
        start_id = bounding_box_id[i][start_index]
        end_id = bounding_box_id[i][end_index]

        #If the start is None, the start index will be added to find the first non None value, same for the end
        frame_start += start_index + 1
        frame_end -= 10 - end_index - 1

        #It is possible that only a single frame had a bounding box prediction, and then a speed calculation would be impossible
        if frame_start == frame_end:
            print("We don't have two frames to calculate speed!")
            continue
        
        #Format ids
        image_start_id = f"{frame_start:06d}"
        image_end_id = f"{frame_end:06d}"

        #Find annotations
        annotation_start = find_annotation(image_start_id, start_id, annotations)
        annotation_end = find_annotation(image_end_id, end_id, annotations)

        #Get predicted positions
        start_position = np.array([annotation_start["bbox_pitch"]["x_bottom_middle"], annotation_start["bbox_pitch"]["y_bottom_middle"]])
        end_position = np.array([annotation_end["bbox_pitch"]["x_bottom_middle"], annotation_end["bbox_pitch"]["y_bottom_middle"]])

        #Calculate predicted speed
        dist_travelled = end_position - start_position
        speed = np.linalg.norm(dist_travelled)/((end_index - start_index)*1/25)
        print(vid)
        print("Speed: ", speed, "gt: ", gt_speed)
        
        if speed < 13: #we don't want the impossibly high speeds
            speeds.append(speed)
            gt_speeds.append(gt_speed)
        else:
            print("Speed exceeds 13 m/s!")

        #See how many speeds have a relative error higher than 100% or lower than 10%
        if abs(speed - gt_speed)/gt_speed > 1:
            video_and_frame_start_bad_pred.append([label_dir, frame_start, abs(speed - gt_speed)/gt_speed, speed])
        if abs(speed - gt_speed)/gt_speed < 0.1:
            video_and_frame_start_good_pred.append([label_dir, frame_start, abs(speed - gt_speed)/gt_speed, speed]) 
 
#The name of the lines in the json file
lines = ["Side line left", "Side line top", "Side line right", "Side line bottom", "Big rect. left top", "Big rect. left main",
         "Big rect. left bottom", "Big rect. right top", "Big rect. right main", "Big rect. right bottom", "Small rect. left top",
         "Small rect. left main", "Small rect. left bottom", "Small rect. right top", "Small rect. right main", "Small rect. right bottom",
         "Middle line", "Circle central", "Circle left", "Circle right"]

speeds_pg = []
gt_speeds_pg = []
video_and_frame_start_bad_pred_pg = []
video_and_frame_start_good_pred_pg = []
line_predictions = []
line_gt = []
for dir in os.listdir(bbox_dir):
    interpolated_dir = bbox_dir + dir + "/interpolated-predictions/"
    dir_name = [dir for dir in Path(interpolated_dir).iterdir()][0]
    file_path = Path(str(dir_name) + "/Labels-GameState.json")
    vid = str(dir_name.name)

    #Load predicted bounding boxes
    with file_path.open("r") as file:
        bbox_data = json.load(file)
    
    label_dir = vid.replace("video_", "").upper()
    label = parent_dir_labels + label_dir

    video = parent_dir_vids + label_dir + "/video_" + label_dir + ".mp4"
    label += "/Labels-GameState.json"


    bounding_box_id = bounding_box_ids[label_dir] #Name is equal to label_dir value

    #Load ground truth bounding boxes
    with open(label, 'r') as file:
        data = json.load(file)

    stadium_dim = [105, 65]
    height_vid = data["images"][0]["height"]
    width_vid = data["images"][0]["width"]

    for i in range(len(bounding_box_id)):
        real_start_pos = None
        real_end_pos = None 
        
        fps = 25

        #Get gt speed
        bounding_box_id_gt = bounding_box_ids_gt[label_dir][i]
        frame_start = frames_start[label_dir][i]
        frame_end = frame_start + 10

        annotations_gt = data["annotations"]

        #Format image id
        vid_id = re.findall(r'\d+',label_dir)
        image_start_id = f"3{vid_id[0]}{(frame_start + 1):06d}"
        image_end_id = f"3{vid_id[0]}{(frame_end + 1):06d}"

        annotation_gt_start = find_annotation(image_start_id, bounding_box_id_gt, annotations_gt)
        annotation_gt_end = find_annotation(image_end_id, bounding_box_id_gt, annotations_gt)

        #If there is no ground truth found, skip iteration
        if annotation_gt_start["bbox_pitch"] == None or annotation_gt_end["bbox_pitch"] == None:
            print("Gt not found.")
            continue

        gt_start_pos = np.array([annotation_gt_start["bbox_pitch"]["x_bottom_middle"], annotation_gt_start["bbox_pitch"]["y_bottom_middle"]])
        gt_end_pos = np.array([annotation_gt_end["bbox_pitch"]["x_bottom_middle"], annotation_gt_end["bbox_pitch"]["y_bottom_middle"]])

        dist_travelled = gt_end_pos - gt_start_pos
        gt_speed_pg = np.linalg.norm(dist_travelled)/(10/25)

        #Find indices for predicted bounding boxes
        start_index = next((j for j in range(len(bounding_box_id[i])) if bounding_box_id[i][j] is not None), None)
        end_index = next((j for j in reversed(range(len(bounding_box_id[i]))) if bounding_box_id[i][j] is not None), None)

        #If there are no bounding box detections, skip to next iteration
        if start_index is None or end_index is None:
            print("Player not detected")
            continue

        #Get bounding box ids
        start_id = bounding_box_id[i][start_index]
        end_id = bounding_box_id[i][end_index]

        #Calculate the frames where there is a predicted sequence
        frame_start += start_index + 1
        frame_end -= 10 - end_index - 1

        #If only one bounding box is predicted, we cannot calculate speed
        if frame_start == frame_end:
            print("We don't have two frames to calculate speed!")
            continue

        #Height and width in predictions
        height_pnl = bbox_data["images"][0]["height"]
        width_pnl = bbox_data["images"][0]["width"]

        annotations = bbox_data["annotations"]


        #Save results to calculate OKS for the line predictions
        for frame in range(frame_start, frame_end + 1):
            new_image_id = f"{frame:06d}"
            new_lines = find_lines(new_image_id, annotations)
            line_predictions.append(new_lines)

            new_image_id_gt = f"3{vid_id[0]}{(frame + 1):06d}"
            new_lines_gt = find_lines(new_image_id_gt, annotations_gt)
            line_gt.append(new_lines_gt)

        #Format ids
        image_start_id = f"{frame_start:06d}"
        image_end_id = f"{frame_end:06d}"

        #Find all predicted lines for the id
        lines_start = find_lines(image_start_id, annotations)
        lines_end = find_lines(image_end_id, annotations)

        if lines_start is None or lines_end is None:
            print("Couldn't find any lines")
            continue

        #Identify all the possible perspective grids
        possible_rectangles_start, keypoint_numbers_start = find_possible_rectangles(lines_start.keys())
        possible_rectangles_end, keypoint_numbers_end = find_possible_rectangles(lines_end.keys())

        annotation_start = find_annotation(image_start_id, start_id, annotations)
        annotation_end = find_annotation(image_end_id, end_id, annotations)

        #Find start and end positions in pixel coordinates
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

        #Calculate the real positions
        position_list = calc_position_list_extra(stadium_dim[0], stadium_dim[1]) 

        if frame_start == frame_end:
            print("We don't have two frames to calculate speed!")
            continue


        #Calculate perspective grids
        for i, rectangle in enumerate(possible_rectangles_start):
            if len(rectangle) == 4:
                try:
                    point_1, point_2 = lines_start[lines[rectangle[0]]][0], lines_start[lines[rectangle[0]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l1 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_start[lines[rectangle[1]]][0], lines_start[lines[rectangle[1]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l2 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_start[lines[rectangle[2]]][0], lines_start[lines[rectangle[2]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l3 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_start[lines[rectangle[3]]][0], lines_start[lines[rectangle[3]]][-1]
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

                    x = np.linspace(0, 1920, 500)

                    frames = extract_frame(video, frame_start - 1, frame_end - 1)
                    #start_frame_results, end_frame_results = model(frames[0]), model(frames[-1]) 

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
                    #print(points)
                    # plt.scatter(points[:, 0], points[:, 1], color='black', zorder=5)#, label="Points")

                    # # Plot each line
                    # plt.plot(x, y1, color="red")
                    # plt.plot(x, y2, color="red")
                    # plt.plot(x, y3, color="red")
                    # plt.plot(x, y4, color="red")
                    # plt.plot(x, y5, color="red")
                    # plt.plot(x, y6, color="red")
                    # #plt.plot(x, y7, label="d1", color="red")
                    # #plt.plot(x, y8, label="d2", color="red")

                    # # Add labels and title
                    # plt.xlabel("x")
                    # plt.ylabel("y")
                    # plt.title(f"{vid}, frame {frame_start}")
                    # plt.legend()

                    # plt.savefig("perspgrid.png")
                    break
                except:
                    print("One of the lines is not complete")
            
            #If the center circle is the visible shape
            elif len(rectangle) == 2:
                try: 
                    circle_points = lines_start[lines[rectangle[1]]]
                    A, B, C, D, E = find_ellipse(circle_points, height_vid, width_vid)

                    point_1, point_2 = lines_start[lines[rectangle[0]]][0], lines_start[lines[rectangle[0]]][-1]
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
                    # print(points)
                    # plt.scatter(points[:, 0], points[:, 1], color='black', zorder=5, label="Points")

                    # # Plot each line
                    # plt.plot(x, y1, color="red")
                    # plt.plot(x, y2, color="red")
                    # plt.plot(x, y3, color="red")
                    # plt.plot(x, y4, color="red")
                    # plt.plot(x, y5, color="red")
                    # plt.plot(x, y6, color="red")
                    # #plt.plot(x, y7, label="d1", color="red")
                    # #plt.plot(x, y8, label="d2", color="red")

                    # # Add labels and title
                    # plt.xlabel("x")
                    # plt.ylabel("y")
                    # plt.title(f"{vid}, frame {frame_start}")
                    # plt.legend()

                    # plt.savefig("perspgrid.png")
                    break
                except:
                    print("Line or ellipse is not complete")

        #Same procedure for the end frame
        for i, rectangle in enumerate(possible_rectangles_end):
            if len(rectangle) == 4:
                try:
                    point_1, point_2 = lines_end[lines[rectangle[0]]][0], lines_end[lines[rectangle[0]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l1 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_end[lines[rectangle[1]]][0], lines_end[lines[rectangle[1]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l2 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_end[lines[rectangle[2]]][0], lines_end[lines[rectangle[2]]][-1]
                    point_1 = np.array([point_1["x"]*width_vid, point_1["y"]*height_vid])
                    point_2 = np.array([point_2["x"]*width_vid, point_2["y"]*height_vid])
                    l3 = increase_and_constant(point_1, point_2)

                    point_1, point_2 = lines_end[lines[rectangle[3]]][0], lines_end[lines[rectangle[3]]][-1]
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
                    # #colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

                    # # Plot each point with a unique color and label
                    # #for i, point in enumerate(points):
                    #  #   plt.scatter(point[0], point[1], color=colors[i], label=f"Point {i+1}")
                    # plt.scatter(points[:, 0], points[:, 1], color='black', zorder=5)#, label="Points")

                    # # Plot each line
                    # plt.plot(x, y1, color="red")#, label="l1")
                    # plt.plot(x, y2, color="red")#, label="l2")
                    # plt.plot(x, y3, color="red")#, label="l3")
                    # plt.plot(x, y4, color="red")#, label="l4")
                    # plt.plot(x, y5, color="red")#, label="m1")
                    # plt.plot(x, y6, color="red")#, label="m2")
                    # #plt.plot(x, y7, color="red")#, label="d1")
                    # #plt.plot(x, y8, color="red")#, label="d2")

                    # Add labels and title
                    dist_travelled = real_end_pos - real_start_pos
                    speed_pg = np.linalg.norm(dist_travelled)/((end_index - start_index)*1/25)
                    rel_error = abs(speed_pg - gt_speed_pg)/gt_speed_pg

                    # plt.xlabel("x")
                    # plt.ylabel("y")
                    # plt.title(f"{vid}, frame {frame_end}")

                    # plt.text(
                    #     0.05, 0.95,  # X, Y position in axes fraction (0-1)
                    #     f"Rel Error: {rel_error:.1%}",  # Format as percentage
                    #     transform=plt.gca().transAxes,  # Use axes fraction coords
                    #     fontsize=10,
                    #     verticalalignment='top',
                    #     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                    # )

                    # plt.legend()

                    # plt.savefig("perspgrid2.png")
                    break

                except:
                    print("One of the lines is not complete")

            elif len(rectangle) == 2:
                try: 
                    circle_points = lines_end[lines[rectangle[1]]]
                    A, B, C, D, E = find_ellipse(circle_points, height_vid, width_vid)

                    point_1, point_2 = lines_end[lines[rectangle[0]]][0], lines_start[lines[rectangle[0]]][-1]
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
                    real_end_pos[0], real_end_pos[1] = real_end_pos[0] + x_displacement, y_displacement - real_end_pos[1] 
                    
                    break
                except:
                    print("Line or ellipse is not complete")

        #Calculate speed
        print(video)
        if real_end_pos is not None and real_start_pos is not None:
            dist_travelled = real_end_pos - real_start_pos
            speed_pg = np.linalg.norm(dist_travelled)/((end_index - start_index)*1/25)
            print("Speed: ", speed_pg, "gt: ", gt_speed_pg)
        else: 
            print("Could not find positions!")
            continue
     
        if speed_pg < 13: #we don't want the impossibly high speeds
            speeds_pg.append(speed_pg)
            gt_speeds_pg.append(gt_speed_pg)
        else:
            print("Speed exceeds 13 m/s!")

        if abs(speed_pg - gt_speed_pg)/gt_speed_pg > 1:
            video_and_frame_start_bad_pred_pg.append([label_dir, frame_start, abs(speed_pg - gt_speed_pg)/gt_speed_pg, speed_pg]) 

        if abs(speed_pg - gt_speed_pg)/gt_speed_pg < 0.1:
            video_and_frame_start_good_pred_pg.append([label_dir, frame_start, abs(speed_pg - gt_speed_pg)/gt_speed_pg, speed_pg]) 

#Calculate HOTA
pred_sio = StringIO()
for entry in pred_entries:
    frame_id, obj_id, x, y, w, h = entry
    pred_sio.write(f"{frame_id},{obj_id},{x},{y},{w},{h},1,-1,-1,-1\n")
pred_sio.seek(0)
gt_sio = StringIO()
for entry in gt_entries:
    frame_id, obj_id, x, y, w, h = entry
    gt_sio.write(f"{frame_id},{obj_id},{x},{y},{w},{h},1,-1,-1,-1\n")
gt_sio.seek(0)

gt_dataset = Dataset_for_HOTA(gt_sio, {'Tracker1': pred_sio})

evaluator = trackeval.Evaluator()
metrics = [HOTA()]
results, _ = evaluator.evaluate(
    dataset_list=[gt_dataset],
    metrics_list=metrics,
    show_progressbar=True
)

print("HOTA Score:", results['MyCustomDataset']['Tracker1']['COMBINED_SEQ']['class1']['HOTA'])

speeds_pg = np.array(speeds_pg)
gt_speeds_pg = np.array(gt_speeds_pg)
print(f"{len(gt_speeds_pg)}/{number_of_events} with speed found.")
print(f"Avg. relative error perspective grid: {np.sum(abs(speeds_pg - gt_speeds_pg)/gt_speeds_pg)/len(gt_speeds_pg)*100}%")
print(f"Root mean rectangled error (RMSE): {np.sqrt(1/len(gt_speeds_pg)*np.sum((speeds_pg - gt_speeds_pg)**2))} m/s")
print(f"Mean of gt: {np.mean(gt_speeds_pg)} m/s")


speeds = np.array(speeds)
gt_speeds = np.array(gt_speeds)
print(f"{len(gt_speeds)}/{number_of_events} with speed found.")
print(f"Avg. relative error homography: {np.sum(abs(speeds - gt_speeds)/gt_speeds)/len(gt_speeds)*100}%")
print(f"Root mean rectangled error (RMSE): {np.sqrt(1/len(gt_speeds)*np.sum((speeds - gt_speeds)**2))} m/s")
print(f"Mean of gt: {np.mean(gt_speeds)} m/s")


#Print predictions above 100% relative error and below 10% relative error
sorted_bad_preds = sorted(video_and_frame_start_bad_pred, key=lambda x: x[2], reverse=True)
sorted_good_preds = sorted(video_and_frame_start_good_pred, key=lambda x: x[2], reverse=False)

print(sorted_bad_preds, len(sorted_bad_preds))
print(sorted_good_preds, len(sorted_good_preds))

sorted_bad_preds_pg = sorted(video_and_frame_start_bad_pred_pg, key=lambda x: x[2], reverse=True)
sorted_good_preds_pg = sorted(video_and_frame_start_good_pred_pg, key=lambda x: x[2], reverse=False)

print(sorted_bad_preds_pg, len(sorted_bad_preds_pg))
print((sorted_good_preds_pg), len(sorted_good_preds_pg))

#Calculate OKS:
def find_closest_keypoint(gt: list, pred: float):
    min_distance = 500
    ind = 500
    for i in range(len(gt)):
        euclidean_distance = np.sqrt((gt[i]["x"]-pred["x"])**2 + (gt[i]["y"]-pred["y"])**2)
        if euclidean_distance < min_distance:
            min_distance = euclidean_distance 
            ind = i
    return min_distance 


#s is the Root of the area of the line
max_length = 105 #The longest sideline will have the largest area
s_values = {"Side line left": np.sqrt(65/max_length),
            "Side line top": np.sqrt(105/max_length),
            "Side line right": np.sqrt(65/max_length),
            "Side line bottom": np.sqrt(105/max_length),
            "Big rect. left top": np.sqrt(16.46/max_length),
            "Big rect. left main": np.sqrt(40.23/max_length),
            "Big rect. left bottom": np.sqrt(16.46/max_length),
            "Big rect. right top": np.sqrt(16.46/max_length),
            "Big rect. right main": np.sqrt(40.23/max_length),
            "Big rect. right bottom": np.sqrt(16.46/max_length),
            "Small rect. left top": np.sqrt(5.49/max_length),
            "Small rect. left main": np.sqrt(18.29/max_length),
            "Small rect. left bottom": np.sqrt(5.49/max_length),
            "Small rect. right top": np.sqrt(5.49/max_length),
            "Small rect. right main": np.sqrt(18.29/max_length),
            "Small rect. right bottom": np.sqrt(5.49/max_length),
            "Middle line": np.sqrt(65/max_length),
            "Circle central": np.sqrt(57.43/max_length),
            "Circle left": np.sqrt(16.94/max_length), 
            "Circle right": np.sqrt(16.94/max_length),
            "Goal right post left": np.sqrt(2.44/max_length),
            "Goal right post right": np.sqrt(2.44/max_length),
            "Goal right crossbar": np.sqrt(7.32/max_length),
            "Goal left post left": np.sqrt(2.44/max_length),
            "Goal left post right": np.sqrt(2.44/max_length),
            "Goal left crossbar": np.sqrt(7.32/max_length)
            } 

all_lines = ["Side line left", "Side line top", "Side line right", "Side line bottom", "Big rect. left top", "Big rect. left main",
         "Big rect. left bottom", "Big rect. right top", "Big rect. right main", "Big rect. right bottom", "Small rect. left top",
         "Small rect. left main", "Small rect. left bottom", "Small rect. right top", "Small rect. right main", "Small rect. right bottom",
         "Middle line", "Circle central", "Circle left", "Circle right", "Goal right post left", "Goal right post right", "Goal right crossbar",
         "Goal left post left", "Goal left post right", "Goal left crossbar"]


keypoint_similarity = {}
num_gt_points = {}
distances = {}
avg_distances = {}
for line in all_lines:
    keypoint_similarity[line] = 0
    num_gt_points[line] = []
    distances[line] = []
    avg_distances[line] = []

for i in range(len(line_gt)):
    lines = line_predictions[i]
    gt_lines = line_gt[i]
    for true_line in gt_lines.keys():
        num_gt_points[true_line].append(len(gt_lines[true_line]))
        if true_line in lines.keys():
            predicted_line = lines[true_line]
            gt_line = gt_lines[true_line]

            distance_list = []
            for pred in predicted_line:
                distance_list.append(find_closest_keypoint(gt_line, pred))
                
            distances[true_line].append(distance_list)
            avg_distances[true_line].append(np.mean(np.array(distance_list)))

OKS = 0

for line in distances.keys():
    avg_distance_arr = np.array(avg_distances[line])
    std_dev = np.std(avg_distance_arr)
    k_value = 2*std_dev

    # num_points = 0
    # for i in range(len(num_gt_points[line])):
    #     #distance_arr = np.array(distances[line][i])
    #     num_points = num_gt_points[line][i]
    keypoint_similarity[line] = np.sum(np.exp(-avg_distance_arr**2/(2*s_values[line]**2*k_value**2)))

    OKS += keypoint_similarity[line]/len(num_gt_points[line])

OKS /= len(keypoint_similarity.keys())

print(f"Average OKS: {OKS}")

            

