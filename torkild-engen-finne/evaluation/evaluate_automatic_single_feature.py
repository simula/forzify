import numpy as np
import torch
from PIL import Image
import os
import tqdm
from argparse import ArgumentParser
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

if __name__ == "__main__":
    parser = ArgumentParser()
    fild_types = ["Full field", "Big rect. left", "Big rect. right", "Goal left", "Goal right", "Circle right", "Circle left", "Circle central", "Small rect. left", "Small rect. right"]

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
        required=False,
        default=False,
        type=bool,
        help="Use multimask output",
    )


    """
    Should be as follows:
     Folder: /sequnceId/img1/
                            000001.jpg
                            000001.npy
                            000002.jpg
                            000002.npy
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
    data_path = args.data_path

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        


    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)
    mask_generator = SAM2AutomaticMaskGenerator(sam2, multimask_output=multimask_output)
    mask_generator.predictor.model.load_state_dict(torch.load(model_path))



    base_path = data_path
    folders = os.listdir(base_path)
    folders = sorted(folders)
    #filet out non folders
    folders = [os.path.join(base_path, f, "img1") for f in folders if os.path.isdir(os.path.join(base_path, f))]

    all_folders_iou = []
    all_folders_iou_only_predicted = []

    if not os.path.exists(outputPath):
        os.makedirs(os.path.dirname(outputPath), exist_ok=True)


    with open(outputPath, "a") as f:
        #Header for printing to file
        print("---- Results ----", file=f)
        print("Config path: ", model_cfg, file=f)
        print("Model path: ", model_path, file=f)
        print("Base model path: ", sam2_checkpoint, file=f)
        print("Output path: ", outputPath, file=f)

        best_predicted_image = None
        worst_predicted_image = None
        best_iou = 0
        worst_iou = 1
        best_mask = None
        worst_mask = None

        number_of_not_predicted = 0
        number_of_no_gt = 0

        number_of_predicted = 0


        for folder_path in tqdm.tqdm(folders):
            images = os.listdir(folder_path)
            images = sorted(images)
            images = [os.path.join(folder_path, img) for img in images if img.endswith(".jpg")]

            #only do every 25 th image
            images = images[::25]

            iou_list = []
            for img_path in images:

                gt_path = img_path.replace(".jpg", ".npy")
                gt = np.load(gt_path, allow_pickle=True).item()
                gt_mask = gt[field_type]
                img = np.array(Image.open(img_path).convert("RGB"))
                masks = mask_generator.generate(img)
                if len(masks) == 0:
                    if gt_mask.sum() == 0:
                        number_of_no_gt += 1
                        continue
                    else:
                        number_of_not_predicted += 1
                        iou_list.append(None)
                        continue

                number_of_predicted += 1
                    
                
                predicted_mask = masks[0]["segmentation"]

                #iou between the two masks
                intersection = np.logical_and(gt_mask, predicted_mask)
                union = np.logical_or(gt_mask, predicted_mask)
                iou = np.sum(intersection) / np.sum(union)
                if iou > best_iou:
                    best_iou = iou
                    best_predicted_image = img_path
                    best_mask = predicted_mask
                if iou < worst_iou:
                    worst_iou = iou
                    worst_predicted_image = img_path
                    worst_mask = predicted_mask
                
                iou_list.append(iou)

            
            iou_only_all = [0 if iou is None else iou for iou in iou_list]

            iou_only_predicted = [iou for iou in iou_list if iou is not None]


            print("---- Results for folder: ", folder_path, "----", file=f)
            print("Mean IoU (excluded none predicted masks): ", np.mean(iou_only_all), file=f)
            print("Mean IoU (Included none predictd mask as 0): ", np.mean(iou_only_predicted), file=f)
            print("", file=f, flush=True)

            if iou_only_all == []:
                print("No predicted masks for folder: ", folder_path, file=f)
                continue

            all_folders_iou.append(np.mean(iou_only_all))

            all_folders_iou_only_predicted.append(np.mean(iou_only_predicted))

        all_folders_iou_only_predicted = [f_iou for f_iou in all_folders_iou_only_predicted if f_iou is not None]
        print("", file=f)
        print("Mean for every folder included 0 for no predicted", file=f)
        print("Length:", len(all_folders_iou), file=f)
        print(all_folders_iou, file=f)
        print("", file=f)
        print("Mean for every folder included removed for no predicted", file=f)
        print("Length:", len(all_folders_iou_only_predicted), file=f)
        print(all_folders_iou_only_predicted, file=f)
        print("", file=f)
        print("---- Results for all folders: ----", file=f)
        print("Mean IoU (excluded none predicted masks): ", np.mean(all_folders_iou), file=f)
        print("Mean IoU (Included none predictd mask as 0): ", np.mean(all_folders_iou_only_predicted), file=f)
        print("", file=f)
        print("Best predicted image: ", best_predicted_image, " with IoU: ", best_iou, file=f)
        print("Worst predicted image: ", worst_predicted_image, " with IoU: ", worst_iou, file=f)
        print("", file=f)
        print("Number of predicted masks: ", number_of_predicted, file=f)
        print("Number of not predicted masks: ", number_of_not_predicted, file=f)
        print("Number of no ground truth masks: ", number_of_no_gt, file=f)
        print("Total number of images: ", number_of_predicted + number_of_not_predicted + number_of_no_gt, file=f)
        print("---- End of results ----", file=f)

        
        