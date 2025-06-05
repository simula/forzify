import numpy as np
import torch
import os
import tqdm
from argparse import ArgumentParser
import pandas as pd
from sam2.build_sam import build_sam2_video_predictor


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def gt_mapper(gt_path, use_mapping):
    gt_mask = np.load(gt_path, allow_pickle=True).item()
    mapped_gt_mask = {}
    
    if use_mapping:

        features = {'Full field':0, 'Big rect. left':1, 'Big rect. right':1, 'Goal left':2, 'Goal right':2, 'Circle right':3, 'Circle left':3, 'Circle central':4, 'Small rect. left':5, 'Small rect. right':5}
        for key in gt_mask.keys():
            if features[key] not in mapped_gt_mask.keys():
                mapped_gt_mask[features[key]] = gt_mask[key]
            elif features[key] in mapped_gt_mask.keys():
                if gt_mask[key].sum() > mapped_gt_mask[features[key]].sum():
                    mapped_gt_mask[features[key]] = gt_mask[key]
    
    else:
        features = {'Full field':0, 'Big rect. left':1, 'Big rect. right':2, 'Goal left':3, 'Goal right':4, 'Circle right':5, 'Circle left':6, 'Circle central':7, 'Small rect. left':8, 'Small rect. right':9}
        for key in gt_mask.keys():
            mapped_gt_mask[features[key]] = gt_mask[key]        

    return mapped_gt_mask

def match_masks(gt_masks, pred_masks):
    matched_pairs = {}
    for key in gt_masks:
        best_iou = 0.0
        best_pred = None
        for pred in pred_masks:
            iou = compute_iou(gt_masks[key], pred)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred
                
        matched_pairs[key] = [gt_masks[key], best_pred, best_iou]

    return matched_pairs

def print_metrics(TP, FP, FN, info, use_mapping=True):
    eps = 1e-6 

    precision = TP.float() / (TP + FP + eps)      
    recall    = TP.float() / (TP + FN + eps)

    IoU   = TP.float() / (TP + FP + FN + eps)
    Dice  = 2 * TP.float() / (2 * TP + FP + FN + eps)

    mIoU   = IoU.mean()
    mDice  = Dice.mean()    
    print(info, file=f)
    print("mIoU : ", mIoU.item(), file=f)
    print("mDice: ", mDice.item(), file=f)

    if use_mapping:
        class_names = ["Full field", "Big rect.", "Goal", "Penalty Arc", "Circle central", "Small rect"]
    else:
        class_names = ["Full field", "Big rect. left", "Big rect. right", "Goal left", "Goal right", "Circle right", "Circle left", "Circle central", "Small rect. left", "Small rect. right"]

    df = pd.DataFrame({
        "Class": class_names,
        "IoU":    IoU.cpu().numpy(),
        "Dice":   Dice.cpu().numpy(),
        "TP":     TP.cpu().numpy(),
        "FP":     FP.cpu().numpy(),
        "FN":     FN.cpu().numpy(),
        "Precision": precision.cpu().numpy(),
        "Recall":    recall.cpu().numpy(),
    })
    print(df.to_string(index=False), file=f)
    print("", file=f, flush=True)

if __name__ == "__main__":
    fild_types = ["Full field", "Big rect. left", "Big rect. right", "Goal left", "Goal right", "Circle right", "Circle left", "Circle central", "Small rect. left", "Small rect. right"]

    parser = ArgumentParser()

    # Arguments
    parser.add_argument(
        "-cp",
        "--config_path",
        required=True,
        type=str,
        help="Path to the config file",
    )
    parser.add_argument(
        "-mp",
        "--model_path",
        required=True,
        type=str,
        help="Path to the model file",
    )
    parser.add_argument(
        "-bm",
        "--base_model",
        required=True,
        type=str,
        help="Path to the base model file",
    )

    parser.add_argument(
        "-op",
        "--output_path",
        required=True,
        type=str,
        help="Path to the output file",
    )


    parser.add_argument(
        "-ft",
        "--field_type",
        required=False,
        default="Full field",
        type=str,
        help="Type of field to evaluate (" + str(fild_types) + ")",
        choices=fild_types,
    )
    parser.add_argument(
        "-mm",
        "--multimask_output",
        action="store_true",
        help="Use multimask output",
    )

    parser.add_argument(
        "-sf",
        "--sequence_frequency",
        required=False,
        default=25,
        type=int,
        help="Use sequence frequency",
    )

    parser.add_argument(
        "-uv",
        "--use_video",
        action="store_true",
        help="Use video (use of the memory block)",
    )

    parser.add_argument(
        "-ma",
        "--mapping",
        action="store_true",
        help="Use mapping for the masks. (mix left and right features)",
    )

    parser.add_argument(
        "-gt",
        "--from_gt_points",
        action="store_true",
        help="Use ground truth points for the masks. (mix left and right features)",
    )

    parser.add_argument(
        "-ef",
        "--eval_frequency",
        required=False,
        default=25,
        type=int,
        help="Use eval frequency",
    )

    parser.add_argument(
        "-cp",
        "--center_points_path",
        required=False,
        default="",
        type=str,
        help="Path to the center points file",
    )


    """
    Should be as follows:
        Folder:/sequnceId/img1/
                                000001.jpg
                                000001.npy
                                000002.jpg
                                000002.npy
                                ...
                ...

    where the npy files contain a dictionary with the key being the field type and the value being the mask.
    ["Full field", "Big rect. left", "Big rect. right", "Goal left", "Goal right", "Circle right", "Circle left", "Circle central", "Small rect. left", "Small rect. right"]
    """
    parser.add_argument(
        "-dp",
        "--data_path",
        required=True,
        type=str,
        help="Path to the data directory",
    )



    args = parser.parse_args()
    
    model_cfg = args.config_path
    sam2_checkpoint = args.base_model 
    model_path = args.model_path 
    outputPath = args.output_path 
    field_type = args.field_type
    multimask_output = args.multimask_output
    sequence_frequency = args.sequence_frequency
    use_video = args.use_video
    use_mapping = args.mapping
    from_gt_points = args.from_gt_points
    eval_frequency = args.eval_frequency
    center_points_path = args.center_points_path
    data_path = args.data_path

    number_of_classes = 10
    if use_mapping:
        number_of_classes = 6


    if not use_video and (eval_frequency != sequence_frequency):
        print("Eval frequency must be the same as sequence frequency if not using video")
        exit(1)


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        

    # from sam2.build_sam import build_sam2
    # sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)
    # predictor = SAM2ImagePredictor(sam2)

    # mask_generator = SAM2AutomaticMaskGenerator(sam2, multimask_output=multimask_output)

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    predictor.load_state_dict(torch.load(model_path))



    folders = os.listdir(data_path)
    folders = sorted(folders)

    #filter out folders that are not directories or do not contain "img1"
    folders = [os.path.join(data_path, f, "img1") for f in folders if os.path.isdir(os.path.join(data_path, f))]

    video_dir_list = {}
    for video_dir in folders:
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        # append the full path to the frame names
        video_dir_list[video_dir] = frame_names



    all_folders_iou = []
    all_folders_iou_only_predicted = []

    if not os.path.exists(outputPath):
        os.makedirs(os.path.dirname(outputPath), exist_ok=True)


    with open(outputPath, "a") as f:
        #Making hearder for printing to file
        print("---- Results ----", file=f)
        print("Config path: ", model_cfg, file=f)
        print("Model path: ", model_path, file=f)
        print("Base model path: ", sam2_checkpoint, file=f)
        print("Output path: ", outputPath, file=f)
        print("Field type: ", field_type, file=f)
        print("Multimask output: ", multimask_output, file=f)
        print("Sequence frequency: ", sequence_frequency, file=f)
        print("Use video: ", use_video, file=f)
        print("Use mapping: ", use_mapping, file=f)
        print("Use ground truth points: ", from_gt_points, file=f)
        print("Eval frequency: ", eval_frequency, file=f)
        print("Number of videoclips: ", len(video_dir_list), file=f)
        print("", file=f, flush=True)

        best_predicted_image = [None] * 10
        worst_predicted_image = [None] * 10
        best_iou = [0] * 10
        worst_iou = [1] * 10
        best_mask = [None] * 10
        worst_mask = [None] * 10

        number_of_not_predicted = 0
        number_of_no_gt = 0

        number_of_predicted = 0

        fild_types_IoU_dict = {
            "Full field": [],
            "Big rect. left": [],
            "Big rect. right": [],
            "Goal left": [],
            "Goal right": [],
            "Circle right": [],
            "Circle left": [],
            "Circle central": [],
            "Small rect. left": [],
            "Small rect. right": [],
        }
        TP = torch.zeros(number_of_classes, dtype=torch.long, device=device)
        FP = torch.zeros_like(TP)
        FN = torch.zeros_like(TP)
        for video_dir in tqdm.tqdm(video_dir_list.keys()):
            local_TP = torch.zeros_like(TP)
            local_FP = torch.zeros_like(TP)
            local_FN = torch.zeros_like(TP)

            print("Video dir: ", video_dir, file=f)
            if True: #use_video:
                inference_state = predictor.init_state(video_path=video_dir)
                predictor.reset_state(inference_state)
            prompts = {}
            
            mapper = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
            if use_mapping:
                mapper = {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:5, 9:5}



            #get center points from file
            video_dir_name = video_dir.split("/")[-2]
            center_points = os.path.join(center_points_path, video_dir_name + ".npy")
            print("Center points file: ", center_points, file=f)

            centers = np.load(center_points, allow_pickle=True).item()

            features = {'Full field':1, 'Big rect. left':2, 'Big rect. right':2, 'Goal left':3, 'Goal right':3, 'Circle right':4, 'Circle left':4, 'Circle central':5, 'Small rect. left':6, 'Small rect. right':6}
            mapped_index_to_feature = {0:'Full field', 1:'Big rect', 2:'Goal', 3:'Penalty Arc', 4:'Circle central', 5:'Small rect'}


            from_indx_to_feature = {0: "Full field", 1: "Big rect. left", 2: "Big rect. right", 3: "Goal left", 4: "Goal right", 5: "Circle right", 6: "Circle left", 7: "Circle central", 8: "Small rect. left", 9: "Small rect. right"}
            for ann_frame_idx in range(0, len(frame_names), sequence_frequency):    
                prompts = {}
                centers_maks = centers[ann_frame_idx]
                image_pts = []
                point_labels = []
                found_masks = []
                
                for cls in centers_maks:

                    x_point, y_point = centers_maks[cls]

                    points = np.array([[int(x_point), int(y_point)]], dtype=np.float32)
                    point_label = np.array([1], np.int32)

                    labels = point_label
                    prompts[mapper[cls]] = points, labels
                    # prompts[cls] = points, labels
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=mapper[cls],
                        # obj_id=cls,
                        points=points,
                        labels=labels,
                    )
                if not use_video:
                    gt_path = os.path.join(video_dir, frame_names[ann_frame_idx].replace(".jpg", ".npy"))
                    gt_mask = gt_mapper(gt_path, use_mapping)

                    for i, out_obj_id in enumerate(out_obj_ids):
                        img_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    
                        if gt_mask[out_obj_id].sum() == 0 and img_mask.sum() == 0:
                            continue
                        cur_gt_mask = gt_mask[out_obj_id]

                        IoU = compute_iou(cur_gt_mask, img_mask)
                        TP[out_obj_id] += (cur_gt_mask & img_mask).sum()
                        FP[out_obj_id] += (~cur_gt_mask & img_mask).sum()
                        FN[out_obj_id] += (cur_gt_mask & ~img_mask).sum()
                        
                        local_TP[out_obj_id] += (cur_gt_mask & img_mask).sum()
                        local_FP[out_obj_id] += (~cur_gt_mask & img_mask).sum()
                        local_FN[out_obj_id] += (cur_gt_mask & ~img_mask).sum()


            if use_video:    
                # run propagation throughout the video and collect the results in a dict
                video_segments = {}  # video_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }


                for out_frame_idx in range(0, len(frame_names), eval_frequency):
                    gt_path = os.path.join(video_dir, frame_names[out_frame_idx].replace(".jpg", ".npy"))
                    gt_mask = gt_mapper(gt_path, use_mapping)

                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():

                        if gt_mask[out_obj_id].sum() == 0 and out_mask.sum() == 0:
                            continue
                        cur_gt_mask = gt_mask[out_obj_id]

                        IoU = compute_iou(cur_gt_mask, out_mask)
                        TP[out_obj_id] += (cur_gt_mask & out_mask).sum()
                        FP[out_obj_id] += (~cur_gt_mask & out_mask).sum()
                        FN[out_obj_id] += (cur_gt_mask & ~out_mask).sum()

                        local_TP[out_obj_id] += (cur_gt_mask & out_mask).sum()
                        local_FP[out_obj_id] += (~cur_gt_mask & out_mask).sum()
                        local_FN[out_obj_id] += (cur_gt_mask & ~out_mask).sum()

            print_metrics(local_TP, local_FP, local_FN, "Final results for video: " + video_dir, use_mapping=use_mapping)
        
        print_metrics(TP, FP, FN, "Final results for all videos", use_mapping=use_mapping)
