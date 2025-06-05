import numpy as np
import os 
import json
from pathlib import Path 

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

for dir in os.listdir(bbox_dir):
    interpolated_dir = bbox_dir + dir + "/interpolated-predictions/"
    dir_name = [dir for dir in Path(interpolated_dir).iterdir()][0]
    file_path = Path(str(dir_name) + "/Labels-GameState.json")
    vid = str(dir_name.name)

    with file_path.open("r") as file:
        bbox_data = json.load(file)

    label_dir = vid.replace("video_", "").upper()
    label = parent_dir_labels + label_dir
    label += "/Labels-GameState.json"

    bounding_box_id = bounding_box_ids[label_dir] #Name is equal to label_dir value

    with open(label, 'r') as file:
        data = json.load(file)