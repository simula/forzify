import numpy as np 
import cv2

def calc_position_list_extra(field_width, field_height): 
    """
    Position of all the keypoints, including extra ones for line interpolation
    """
    position_list = np.zeros((57, 2)) #Distance between all the points on the field
    position_list[0] = [0,0]
    position_list[1] = [0, field_height/2 - 20.12]
    position_list[2] = [0, field_height/2 - 9.14]
    position_list[3] = [0, field_height/2 - 3.66]
    position_list[4] = [0, field_height/2 + 3.66]
    position_list[5] = [0, field_height/2 + 9.14]
    position_list[6] = [0, field_height/2 + 20.12]
    position_list[7] = [0, field_height]
    position_list[8] = [5.49, field_height/2 - 9.14]
    position_list[9] = [5.49, field_height/2 + 9.14]
    position_list[10] = [10.97, field_height/2]
    position_list[11] = [16.46, field_height/2 - 20.12]
    position_list[12] = [16.46, field_height/2 - 7.3]
    position_list[13] = [16.46, field_height/2 + 7.3]
    position_list[14] = [16.46, field_height/2 + 20.12]
    position_list[15] = [20.11, field_height/2]
    position_list[16] = [field_width/2 - 9.14, field_height/2]
    position_list[17] = [field_width/2, 0]
    position_list[18] = [field_width/2, field_height/2 - 9.14]
    position_list[19] = [field_width/2, field_height/2]
    position_list[20] = [field_width/2, field_height/2 + 9.14]
    position_list[21] = [field_width/2, field_height]
    position_list[22] = [field_width/2 + 9.14, field_height/2]
    position_list[23] = [field_width - 20.11, field_height/2]
    position_list[24] = [field_width - 16.46, field_height/2 - 20.12]
    position_list[25] = [field_width - 16.46, field_height/2 - 7.3]
    position_list[26] = [field_width - 16.46, field_height/2 + 7.3]
    position_list[27] = [field_width - 16.46, field_height/2 + 20.12]
    position_list[28] = [field_width - 10.97, field_height/2]
    position_list[29] = [field_width - 5.49, field_height/2 - 9.14]
    position_list[30] = [field_width - 5.49, field_height/2 + 9.14]
    position_list[31] = [field_width, 0]
    position_list[32] = [field_width, field_height/2 - 20.12]
    position_list[33] = [field_width, field_height/2 - 9.14]
    position_list[34] = [field_width, field_height/2 - 3.66]
    position_list[35] = [field_width, field_height/2 + 3.66]
    position_list[36] = [field_width, field_height/2 + 9.14]
    position_list[37] = [field_width, field_height/2 + 20.12]
    position_list[38] = [field_width, field_height]
    position_list[39] = [field_width/2 - 4.57, field_height/2 - 7.92]
    position_list[40] = [field_width/2 - 4.57, field_height/2 + 7.92]
    position_list[41] = [field_width/2 + 4.57, field_height/2 - 7.92]
    position_list[42] = [field_width/2 + 4.57, field_height/2 + 7.92]
    position_list[43] = [field_width/2, 20.12]
    position_list[44] = [field_width/2, field_height/2 + 20.12]
    position_list[45] = [16.46, 0]
    position_list[46] = [16.46, field_height]
    position_list[47] = [field_width - 16.46, 0]
    position_list[48] = [field_width - 16.46, field_height]
    position_list[49] = [5.49, field_height/2 - 20.12]
    position_list[50] = [5.49, field_height/2 + 20.12]
    position_list[51] = [field_width - 5.49, field_height/2 - 20.12]
    position_list[52] = [field_width - 5.49, field_height/2 + 20.12]
    position_list[53] = [16.46, field_height/2 - 9.14]
    position_list[54] = [16.46, field_height/2 + 9.14]
    position_list[55] = [field_width - 16.46, field_height/2 - 9.14]
    position_list[56] = [field_width - 16.46, field_height/2 + 9.14]

    return position_list


def find_annotation(image_id, track_id, annotations):
    """
    Find annotation for a specific image and track id
    """
    for annotation in annotations:
        if annotation["image_id"] == image_id and annotation["track_id"] == track_id:
            return annotation
    return None

def find_lines(image_id, annotations):
    """
    Find predicted lines for an image id
    """
    for annotation in annotations:
        if annotation["image_id"] == image_id and annotation["supercategory"] == "pitch":
            return annotation["lines"]
    return None

def find_possible_rectangles(available_lines):
    """
    Find possible rectangles based on the lines that have been predicted
    """
    lines = ["Side line left", "Side line top", "Side line right", "Side line bottom", "Big rect. left top", "Big rect. left main",
         "Big rect. left bottom", "Big rect. right top", "Big rect. right main", "Big rect. right bottom", "Small rect. left top",
         "Small rect. left main", "Small rect. left bottom", "Small rect. right top", "Small rect. right main", "Small rect. right bottom",
         "Middle line", "Circle central", "Circle left", "Circle right"]
    
    possible_combinations= [[0, 5, 6, 4], [8, 2, 9, 7], [0, 11, 12, 10], [14, 2, 15, 13], [5, 16, 6, 4], [16, 8, 9, 7], [0, 5, 4, 1], [0, 5, 3, 6], [8, 2, 7, 1], 
                            [8, 2, 3, 9], [11, 5, 6, 4], [8, 14, 9, 7], [0, 5, 10, 4], [0, 5, 6, 12], [8, 2, 13, 7], [8, 2, 9, 15], [5, 16, 4, 1],
                            [5, 16, 3, 6], [16, 8, 7, 1], [16, 8, 3, 9]]
    
    keypoint_number_list = [[7, 15, 2, 12], [28, 38, 25, 33], [6, 10, 3, 9], [31, 37, 30, 34], [16, 45, 12, 44], [45, 28, 44, 25], [2, 12, 1, 46], [8, 47, 7, 15], [25, 33, 48, 32],
                            [49, 39, 28, 38], [51, 15, 50, 12], [28, 53, 25, 52], [3, 54, 2, 12], [7, 15, 6, 55], [56, 34, 25, 33], [28, 38, 57, 37], [12, 44, 46, 18],
                            [47, 22, 15, 45], [44, 25, 18, 48], [22, 49, 45, 28]]
    
    mid_combination = [16, 17]

    possible_rectangles = []
    keypoints = []
    line_1 = lines[mid_combination[0]]
    line_2 = lines[mid_combination[1]]
    if line_1 in available_lines and line_2 in available_lines:
        possible_rectangles.append(mid_combination)
        keypoints.append("midcircle")

    for i, combination in enumerate(possible_combinations):
        line_1 = lines[combination[0]]
        line_2 = lines[combination[1]]
        line_3 = lines[combination[2]]
        line_4 = lines[combination[3]]

        if line_1 in available_lines and line_2 in available_lines and line_3 in available_lines and line_4 in available_lines:
            possible_rectangles.append(combination)
            keypoints.append(keypoint_number_list[i])

    
    return possible_rectangles, keypoints

def calc_position_list(field_width, field_height): 
    """
    Position of all the keypoints
    """
    position_list = np.zeros((43, 2)) #Distance between all the points on the field
    position_list[0] = [0,0]
    position_list[1] = [0, field_height/2 - 20.12]
    position_list[2] = [0, field_height/2 - 9.14]
    position_list[3] = [0, field_height/2 - 3.66]
    position_list[4] = [0, field_height/2 + 3.66]
    position_list[5] = [0, field_height/2 + 9.14]
    position_list[6] = [0, field_height/2 + 20.12]
    position_list[7] = [0, field_height]
    position_list[8] = [5.49, field_height/2 - 9.14]
    position_list[9] = [5.49, field_height/2 + 9.14]
    position_list[10] = [10.97, field_height/2]
    position_list[11] = [16.46, field_height/2 - 20.12]
    position_list[12] = [16.46, field_height/2 - 7.3]
    position_list[13] = [16.46, field_height/2 + 7.3]
    position_list[14] = [16.46, field_height/2 + 20.12]
    position_list[15] = [20.11, field_height/2]
    position_list[16] = [field_width/2 - 9.14, field_height/2]
    position_list[17] = [field_width/2, 0]
    position_list[18] = [field_width/2, field_height/2 - 9.14]
    position_list[19] = [field_width/2, field_height/2]
    position_list[20] = [field_width/2, field_height/2 + 9.14]
    position_list[21] = [field_width/2, field_height]
    position_list[22] = [field_width/2 + 9.14, field_height/2]
    position_list[23] = [field_width - 20.11, field_height/2]
    position_list[24] = [field_width - 16.46, field_height/2 - 20.12]
    position_list[25] = [field_width - 16.46, field_height/2 - 7.3]
    position_list[26] = [field_width - 16.46, field_height/2 + 7.3]
    position_list[27] = [field_width - 16.46, field_height/2 + 20.12]
    position_list[28] = [field_width - 10.97, field_height/2]
    position_list[29] = [field_width - 5.49, field_height/2 - 9.14]
    position_list[30] = [field_width - 5.49, field_height/2 + 9.14]
    position_list[31] = [field_width, 0]
    position_list[32] = [field_width, field_height/2 - 20.12]
    position_list[33] = [field_width, field_height/2 - 9.14]
    position_list[34] = [field_width, field_height/2 - 3.66]
    position_list[35] = [field_width, field_height/2 + 3.66]
    position_list[36] = [field_width, field_height/2 + 9.14]
    position_list[37] = [field_width, field_height/2 + 20.12]
    position_list[38] = [field_width, field_height]
    position_list[39] = [field_width/2 - 4.57, field_height/2 - 7.92]
    position_list[40] = [field_width/2 - 4.57, field_height/2 + 7.92]
    position_list[41] = [field_width/2 + 4.57, field_height/2 - 7.92]
    position_list[42] = [field_width/2 + 4.57, field_height/2 + 7.92]

    return position_list


def scale_keypoint_grid(position_list, dst_keypoints):
    #Move all points inside the field, so that the pixel distances are to scale.
    tl_pixel, tl_meters = dst_keypoints[0], position_list[0]
    bl_pixel, bl_meters = dst_keypoints[7], position_list[7]
    tr_pixel, tr_meters = dst_keypoints[31], position_list[31]
    br_pixel, br_meters = dst_keypoints[38], position_list[38]

    scale_x = (tl_pixel[0] - tr_pixel[0])/(tl_meters[0] - tr_meters[0])
    scale_y = (tl_pixel[1] - bl_pixel[1])/(tl_meters[1] - bl_meters[1])

    for i in range(dst_keypoints.shape[0]):
        if i not in [0, 7, 31, 38]:
            dst_keypoints[i] = dst_keypoints[0] + np.array([position_list[i][0]*scale_x, position_list[i][1]*scale_y]) 

    return dst_keypoints, scale_x, scale_y



def extract_frame(video_path, frame_start, frame_end):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Get the frame rate (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set the start frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    
    frame_idx = frame_start
    frames = []  # List to store frames in memory
    
    while frame_idx <= frame_end:
        ret, frame = cap.read()
        if not ret:
            break  
        frames.append(frame)
        
        frame_idx += 1
    cap.release()

    return frames

def get_xy_from_txt(file_path, frame, player_id):
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by commas to extract values
            values = line.strip().split(',')
            
            if len(values) >= 6:  # Ensure the line has the expected number of values
                current_frame = int(values[0])
                current_player_id = int(values[1])
                
                # Check if the current frame and player_id match
                if current_frame == frame and current_player_id == player_id:
                    # Extract the bounding box coordinates (x, y, width, height)
                    x = float(values[2])
                    y = float(values[3])
                    width = float(values[4])
                    height = float(values[5])

                    mid_bottom_x = x + width / 2
                    mid_bottom_y = y + height
                    
                    return (mid_bottom_x, mid_bottom_y)
    
    # If no matching frame/player_id is found, return None
    return None

def find_nearest_keypoint(xy, dst_keypoints):
    distances = np.linalg.norm(dst_keypoints - xy, axis=1)
    
    nearest_index = np.argmin(distances)

    return dst_keypoints[nearest_index]
        