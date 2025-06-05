import json
import cv2
import numpy as np
from argparse import ArgumentParser
import os



# Lines list from the json file
lines_list = ["Big rect. left bottom", "Big rect. left main", "Big rect. left top", "Big rect. right bottom",
                "Big rect. right main", "Big rect. right top", "Goal left crossbar", "Goal left post left ",
                "Goal left post right", "Goal right crossbar", "Goal right post left", "Goal right post right",
                "Middle line", "Side line bottom", "Side line left", "Side line right", "Side line top",
                "Small rect. left bottom", "Small rect. left main", "Small rect. left top", "Small rect. right bottom",
                "Small rect. right main", "Small rect. right top"]

# Mapping the spesific lines from the field to corresponding placement
full_field_lines = ["Side line bottom", "Side line left", "Side line right", "Side line top"]
mapping_full_field = { "Side line bottom": "bottom", "Side line left": "left", "Side line right": "right", "Side line top": "top"}

big_rect_right = ["Big rect. right bottom", "Big rect. right main", "Big rect. right top", "Side line right"]
mapping_big_rect_right = { "Big rect. right bottom": "bottom", "Big rect. right main": "left", "Big rect. right top": "top", "Side line right": "right"}

small_rect_right = ["Small rect. right bottom", "Small rect. right main", "Small rect. right top", "Side line right"]
mapping_small_rect_right = { "Small rect. right bottom": "bottom", "Small rect. right main": "left", "Small rect. right top": "top", "Side line right": "right"}

goal_right = ["Goal right crossbar", "Goal right post right", "Goal right post left", "Side line right"]
mapping_goal_right = { "Goal right crossbar": "top", "Goal right post right": "right", "Goal right post left": "left", "Side line right": "bottom"}

big_rect_left = ["Big rect. left bottom", "Big rect. left main", "Big rect. left top", "Side line left"]
mapping_big_rect_left = { "Big rect. left bottom": "bottom", "Big rect. left main": "right", "Big rect. left top": "top", "Side line left": "left"}

small_rect_left = ["Small rect. left bottom", "Small rect. left main", "Small rect. left top", "Side line left"]
mapping_small_rect_left = { "Small rect. left bottom": "bottom", "Small rect. left main": "right", "Small rect. left top": "top", "Side line left": "left"}

goal_left = ["Goal left crossbar", "Goal left post right", "Goal left post left", "Side line left"]
mapping_goal_left = { "Goal left crossbar": "top", "Goal left post right": "right", "Goal left post left": "left", "Side line left": "bottom"}

circle_right = ["Circle right", "Big rect. right main"]
mapping_circle_right = { "Circle right": "ellipse", "Big rect. right main": "line"}

circle_left = ["Circle left", "Big rect. left main"]
mapping_circle_left = { "Circle left": "ellipse", "Big rect. left main": "line"}

circle_central = ["Circle central"]
mapping_circle_central = { "Circle central": "ellipse"}

geometry_groups_mask = {"Full field": None, "Big rect. left": None, "Big rect. right": None, "Goal left": None, "Goal right": None, "Circle right": None, "Circle left": None, "Circle central": None, "Small rect. left": None, "Small rect. right": None}
color_for_group = {"Full field": [255, 255, 255], "Big rect. left": [0, 255, 0], "Big rect. right": [0, 0, 255], "Goal left": [255, 255, 0], "Goal right": [255, 0, 255], "Circle right": [0, 255, 255], "Circle left": [128, 0, 0], "Circle central": [0, 128, 0], "Small rect. left": [0, 0, 128], "Small rect. right": [128, 128, 0]}


# ----------------------------------------------------

def find_intersections(m, b, width, height):
    """
    Input: 
        m: slope of the line
        b: y-intercept of the line
        width: width of the image
        height: height of the image

    Output:
        intersections: list of intersection points with the image boundaries
    """
    intersections = []
    
    # Check x=0 (left edge)
    y_at_x0 = b
    if 0 <= y_at_x0 <= height:
        intersections.append({'x': 0, 'y': y_at_x0})

    # Check x=width (right edge)
    y_at_xw = m * width + b
    if 0 <= y_at_xw <= height:
        intersections.append({'x': width, 'y': y_at_xw})

    # Check y=0 (top edge)
    x_at_y0 = -b / m if m != 0 else None
    if x_at_y0 is not None and 0 <= x_at_y0 <= width:
        intersections.append({'x': x_at_y0, 'y': 0})

    # Check y=height (bottom edge)
    x_at_yh = (height - b) / m if m != 0 else None
    if x_at_yh is not None and 0 <= x_at_yh <= width:
        intersections.append({'x': x_at_yh, 'y': height})

    return intersections

def find_extremities(line, width=1920, height=1080):
    """
    Input:
        line: list of points in the line
    Output:
        boundary_intersections: list of intersection points with the image boundaries
    """

    # Extract x and y coordinates
    x_coords = np.array([p['x']*width for p in line])
    y_coords = np.array([p['y']*height for p in line])

    # Fitting a line (y = mx + b) using least squares
    m, b = np.polyfit(x_coords, y_coords, 1)

    # Finding intersections with the image boundaries
    boundary_intersections = find_intersections(m, b, width, height)
    return boundary_intersections

def get_scaled_extremities_points(data_input, width, height):
    new_lines = {}
    for key in data_input:
        if "Circle" in key:
            #scale the circle points
            new_lines[key] = []
            for point in data_input[key]:
                new_lines[key].append({'x': int(point['x']*width), 'y': int(point['y']*height)})
        else:
            try:
                #Finding the extremities of the line and scaling them
                new_lines[key] = find_extremities(data_input[key], width=width, height=height)
            except Exception as e:
                print(f"Error: Unable to find extremities for {key} at image number {i}. Exception: {e}")
                # new_lines[key] = []
                continue
    return new_lines
    

def generate_visible_lines(height, width, data_path):
    data_input = json.load(open(data_path))
    new_lines = get_scaled_extremities_points(data_input, width, height)

    # Mapping the lines to the corresponding groups
    geometry_groups_dict = { "Big rect. left": {}, "Big rect. right": {}, "Goal left": {}, "Goal right": {}, "Circle right": {}, "Circle left": {}, "Full field": {}, "Circle central": {}, "Small rect. left": {}, "Small rect. right": {}}
    for line in new_lines:
        if line in full_field_lines:
            geometry_groups_dict["Full field"][mapping_full_field[line]] = new_lines[line]
        
        if line in big_rect_right:
            geometry_groups_dict["Big rect. right"][mapping_big_rect_right[line]] = new_lines[line]

        if line in small_rect_right:
            geometry_groups_dict["Small rect. right"][mapping_small_rect_right[line]] = new_lines[line]
        
        if line in goal_right:
            geometry_groups_dict["Goal right"][mapping_goal_right[line]] = new_lines[line]
        
        if line in big_rect_left:
            geometry_groups_dict["Big rect. left"][mapping_big_rect_left[line]] = new_lines[line]
        
        if line in small_rect_left:
            geometry_groups_dict["Small rect. left"][mapping_small_rect_left[line]] = new_lines[line]

        if line in goal_left:
            geometry_groups_dict["Goal left"][mapping_goal_left[line]] = new_lines[line]

        if line in circle_right:
            geometry_groups_dict["Circle right"][mapping_circle_right[line]] = new_lines[line]
        
        if line in circle_left:
            geometry_groups_dict["Circle left"][mapping_circle_left[line]] = new_lines[line]
        
        if line in circle_central:
            geometry_groups_dict["Circle central"][mapping_circle_central[line]] = new_lines[line]
        
    return geometry_groups_dict

def make_dividing_line(visible_lines, key, xx, yy):
    # d = (xx - x) * vy - (yy - y) * vx
    return (xx - int(visible_lines[key][0]['x'])) * (int(visible_lines[key][1]['y']) - int(visible_lines[key][0]['y'])) - (yy - int(visible_lines[key][0]['y'])) * (int(visible_lines[key][1]['x']) - int(visible_lines[key][0]['x']))

def generate_mask_for_rect(visible_lines, height, width):
    #make bool masks for each line
    top_over = np.ones((height, width), dtype=bool)
    right_over = np.ones((height, width), dtype=bool)
    bottom_over = np.ones((height, width), dtype=bool)
    left_over = np.ones((height, width), dtype=bool)
    
    if 'top' in visible_lines:
        
        #sorted lines lowest x to highest x
        sorted_lines = sorted(visible_lines['top'], key=lambda x: x['x'])
        visible_lines['top'] = sorted_lines
    

        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask2 = np.zeros((height, width), dtype=np.uint8)

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))


        d = make_dividing_line(visible_lines, 'top', xx, yy)

        mask1[d >= 0] = 255  # Pixels on one side of the line.
        mask2[d < 0] = 255

        top_over = mask2.astype(bool)
        

    if 'right' in visible_lines:

        #sorted lines highest y to lowest y
        sorted_lines = sorted(visible_lines['right'], key=lambda x: x['y'], reverse=True)
        visible_lines['right'] = sorted_lines

        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask2 = np.zeros((height, width), dtype=np.uint8)

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        d = make_dividing_line(visible_lines, 'right', xx, yy)

        mask1[d >= 0] = 255
        mask2[d < 0] = 255

        right_over = mask1.astype(bool)


    if 'bottom' in visible_lines:

        #sorted lines lowest x to higest x
        sorted_lines = sorted(visible_lines['bottom'], key=lambda x: x['x'], reverse=False)
        visible_lines['bottom'] = sorted_lines

        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask2 = np.zeros((height, width), dtype=np.uint8)

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        d = make_dividing_line(visible_lines, 'bottom', xx, yy)

        mask1[d >= 0] = 255
        mask2[d < 0] = 255

        bottom_over = mask1.astype(bool)
        
    
    if 'left' in visible_lines:

        #sorted lines highest y to lowest y
        sorted_lines = sorted(visible_lines['left'], key=lambda x: x['y'], reverse=True)
        visible_lines['left'] = sorted_lines

        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask2 = np.zeros((height, width), dtype=np.uint8)

        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        d = make_dividing_line(visible_lines, 'left', xx, yy)

        mask1[d >= 0] = 255
        mask2[d < 0] = 255

        left_over = mask2.astype(bool)

    mask = top_over & right_over & bottom_over & left_over
    
    return mask.astype(bool)

def generate_mask_for_central_circle(visible_lines, height, width):

    ellipse_points = np.array([[int(p['x']), int(p['y'])] for p in visible_lines["ellipse"]], dtype=np.int32)
    ellipse_mask = np.zeros((height, width), dtype=np.uint8)

    try:
        ellipse = cv2.fitEllipse(ellipse_points)
    except Exception as e:
        print(f"Error: Unable to fit ellipse for 'Middle Circle' at image number {i}, Returning the fillPoly instead. Exception: {e}")
        # returning the polygon mask

        ellipse_mask = cv2.fillPoly(ellipse_mask, [ellipse_points], 255)
        ellipse_mask = ellipse_mask.astype(bool)
        return ellipse_mask


    
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)

    ellipse_mask = ellipse_mask.astype(bool)

    return ellipse_mask



def generate_mask_for_penalty_arc(visible_lines, height, width):

    if "ellipse" in visible_lines.keys():
        ellipse_points = np.array([[int(p['x']), int(p['y'])] for p in visible_lines["ellipse"]], dtype=np.int32)
        if "line" in visible_lines.keys():
            ellipse_mask = np.zeros((height, width), dtype=np.uint8)

            try:
                ellipse = cv2.fitEllipse(ellipse_points)
            except Exception as e:
                print(f"Error: Unable to fit ellipse for 'Penalty arc group' at image number {i}, Returning the fillPoly instead. Exception: {e}")
                
                # Returning the polygon mask
                ellipse_mask = cv2.fillPoly(ellipse_mask, [ellipse_points], 255)
                ellipse_mask = ellipse_mask.astype(bool)
                return ellipse_mask

            
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)
            
            line_points = np.array([[int(p['x']), int(p['y'])] for p in visible_lines["line"]], dtype=np.int32)

            #Finding the best fitting line between the points
            [vx, vy, x, y] = cv2.fitLine(line_points, cv2.DIST_L2, 0, 0.01, 0.01)


            mask1 = np.zeros((height, width), dtype=np.uint8)
            mask2 = np.zeros((height, width), dtype=np.uint8)

            xx, yy = np.meshgrid(np.arange(width), np.arange(height))

            # The line is defined by the point (x, y) and the direction (vx, vy).
            d = (xx - x) * vy - (yy - y) * vx

            # Desiding the side of the line
            mask1[d >= 0] = 255  # Pixels on one side of the line.
            mask2[d < 0] = 255

            #Useing the intersection of the two ellipse mask and the mask1
            over = cv2.bitwise_and(ellipse_mask, mask1)

            under = cv2.bitwise_and(ellipse_mask, mask2)

            under = under.astype(bool)

            return under
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [ellipse_points], 255)
            #bool mask
            mask = mask.astype(bool)

            return mask
    
    return np.zeros((height, width), dtype=np.uint8).astype(bool)


def generate_mask_for_group(visible_lines, height, width, group_name):

    # Rectangular groups
    rectangular_groups = ["Big rect. left", "Big rect. right", "Goal left", "Goal right", "Full field", "Small rect. left", "Small rect. right"]
    if group_name in rectangular_groups:
        if "Goal left" == group_name and len(visible_lines) == 1 and "bottom" in visible_lines.keys():
            return np.zeros((height, width), dtype=np.uint8).astype(bool)
        
        if "Goal right" == group_name and len(visible_lines) == 1 and "bottom" in visible_lines.keys():
            return np.zeros((height, width), dtype=np.uint8).astype(bool)
        
        if "Small rect. right" == group_name and len(visible_lines) == 1 and "right" in visible_lines.keys():
            return np.zeros((height, width), dtype=np.uint8).astype(bool)
        
        if "Small rect. left" == group_name and len(visible_lines) == 1 and "left" in visible_lines.keys():
            return np.zeros((height, width), dtype=np.uint8).astype(bool)
        
        
        return generate_mask_for_rect(visible_lines, height=height, width=width)

    # Circle groups
    penalty_arc_groups = ["Circle right", "Circle left"]
    if group_name in penalty_arc_groups:
        try:
            return generate_mask_for_penalty_arc(visible_lines, height=height, width=width)
        except Exception as e:
            print(f"Error: Unable to generate mask for 'Penalty arc' at image number {i}. Exception: {e}")
            return np.zeros((height, width), dtype=np.uint8).astype(bool)

    if group_name == "Circle central":
        try:
            return generate_mask_for_central_circle(visible_lines, height=height, width=width)
        except Exception as e:
            print(f"Error: Unable to generate mask for 'Circle central' at image number {i}. Exception: {e}")
            return np.zeros((height, width), dtype=np.uint8).astype(bool)

    return np.zeros((height, width), dtype=np.uint8).astype(bool)


def path_finder(data_dir):
    data=[]
    print("Finding paths in ", data_dir)
    seq_list = os.listdir(data_dir)
    print("Found ", len(seq_list), " sequences")

    #sort the sequences
    seq_list.sort()
    
    for seq in seq_list:
        seq = "SNGS-149"
        if not os.path.isdir(os.path.join(data_dir, seq)):
            continue
        images = os.listdir(data_dir+seq + "/img1")

        #filter out non image files
        images = [img for img in images if img.endswith(".jpg")]
        images.sort()
        for img in images:
            id = img.split(".")[0]
            data.append({"image":data_dir+seq + "/img1/"+img,"annotation":data_dir + seq + "/img1/" + id + ".json", "mask": data_dir + seq + "/img1/" + id + ".npy"})
    return data

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, required=False, default="train")
    parser.add_argument("--visualize", action="store_true", help="Visualize the masks on the images")
    parser.add_argument("--base_dir", type=str, required=False, default="")

    args = parser.parse_args()

    data_dir = args.base_dir + "/" + args.data_type + "/"


    data = path_finder(data_dir)
    print("Found ", len(data), " images")

    for i, ent in enumerate(data):
        image_path = ent["image"]
        save_path = ent["mask"]
        json_path = ent["annotation"]
            
        if i % 750 == 0:
            print("Processing ", json_path, "number ", i)

        if not os.path.exists(json_path) or not os.path.exists(image_path):
            print("No data or image for ", json_path, " or ", image_path)
            continue

        overlay = cv2.imread(image_path)

        height, width, _ = overlay.shape

        test_lines = generate_visible_lines(height, width, json_path)

        for group in geometry_groups_mask:
            if test_lines[group] == {}:
                mask = np.zeros((height, width), dtype=np.uint8).astype(bool)
            else:
                mask = generate_mask_for_group(test_lines[group], height=height, width=width, group_name=group)
            geometry_groups_mask[group] = mask

        
        #make dict a numpy array
        new_dict = {}
        for key in geometry_groups_mask.keys():
            new_dict[key] = geometry_groups_mask[key].astype(np.uint8)
        new_dict = np.array(new_dict)

        np.save(save_path, new_dict)




