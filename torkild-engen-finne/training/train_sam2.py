from argparse import ArgumentParser
import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def datamaker(data_dir="train"):
    """
    Should be as follows:
     Folder:train/sequnceId/img1/
                                000001.jpg
                                000001.npy
                                000002.jpg
                                000002.npy
                                ...
    where the npy files contain a dictionary with the key being the field type and the value being the mask.
    ["Full field", "Big rect. left", "Big rect. right", "Goal left", "Goal right", "Circle right", "Circle left", "Circle central", "Small rect. left", "Small rect. right"]
    """
    data=[]

    for seq in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, seq)):
            continue
        images = os.listdir(data_dir+seq + "/img1")
        #filter out non image files
        images = [img for img in images if img.endswith(".jpg")]
        images.sort()
        for img in images:
            id = img.split(".")[0]
            data.append({"image":data_dir+seq + "/img1/"+img,"annotation":data_dir + seq + "/img1/" + id + ".npy"})
    return data

def read_batch(data, itr, remove_full_field, remove_list, random_point=False):
    # Read a batch of data from the dataset

    #  select image
    ent  = data[itr] # choose entry
    Img = cv2.cvtColor(cv2.imread(ent["image"]), cv2.COLOR_BGR2RGB) #Same as Img = cv2.imread(ent["image"])[...,::-1] 
    ann_map = np.load(ent["annotation"], allow_pickle=True) # load annotation map

    ann_map = ann_map.item()
    
    # if remove_full_field then remove full field mask
    if remove_full_field:
        if "Full field" in ann_map.keys():
            ann_map.pop("Full field") # remove full field mask

    if len(remove_list) > 0:
        new_ann_map = {}
        for key in ann_map.keys():
            if key in remove_list:
                continue
                # ann_map.pop(key)
            new_ann_map[key] = ann_map[key]
    
        ann_map = new_ann_map


    # resize image and mask
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]]) # scalling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r))) # resize image
     
    points= []

    masks = []
    for key in ann_map.keys():

        cur_mask = ann_map[key]
        cur_mask = cv2.resize(cur_mask, (int(cur_mask.shape[1] * r), int(cur_mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
        if cur_mask.sum() > 0: # check if mask is empty
            masks.append(cur_mask)
            coords = np.argwhere(cur_mask > 0) # get all coordinates in mask
            if random_point:
                yx = np.array(coords[np.random.randint(len(coords))]) # choose random point/coordinate
                points.append([[yx[1], yx[0]]])
            else:
                middel_yx = np.mean(coords, axis=0) # get the middle point of the mask
                points.append([[middel_yx[1], middel_yx[0]]])

        else:
            continue
            


    
    return Img, np.array(masks), np.array(points), np.ones([len(masks),1])



if __name__ == "__main__":
    parser = ArgumentParser()

    # Arguments
    parser.add_argument(
        "-mn",
        "--model_name",
        required=True,
        type=str,
        help="name of the saved model",
        default="no_name",
    ) 
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        type=str,
        help="path to config file (e.g. base_config/sam2_hiera_s.yaml)",
        default="/configs/sam2.1/sam2.1_hiera_s.yaml",
    )

    parser.add_argument(
        "-ch",
        "--checkpoint",
        required=False,
        type=str,
        help="path to model weight (e.g. base_models/sam2_hiera_small.pt)",
        default="base_models/sam2_hiera_small.pt",
    )

    parser.add_argument(
        "-d",
        "--data_dir",
        required=False,
        type=str,
        help="path to data directory",
        default="",
    )

    parser.add_argument(
        "-te",
        "--train_encoder",
        action="store_true",
        help="train encoder",
    )

    parser.add_argument(
        "-td",
        "--train_decoder",
        action="store_true",
        help="train decoder",
    )

    parser.add_argument(
        "-f",
        "--field_type",
        required=False,
        type=str,
        help="field type (e.g. Full field, goal left, goal right)",
        default="Full field",
    )

    parser.add_argument(
        "-mu",
        "--multi_mask",
        action="store_true",
        help="multi mask",
    )
    parser.add_argument(
        "-rf",
        "--remove_full_field",
        action="store_true",
        help="remove full field mask",
    )
    parser.add_argument(
        "-ps",
        "--print_step",
        required=False,
        type=int,
        help="print step (e.g. 1000)",
        default=1000,
    )

    parser.add_argument(
        "-me",
        "--max_epochs",
        required=False,
        type=int,
        help="max epochs (e.g. 50)",
        default=50,
    )

    parser.add_argument(
        "-sp",
        "--save_path",
        required=False,
        type=str,
        help="path to save model",
        default="/sam2_models",
    )

    parser.add_argument(
        "-rp",
        "--random_point",
        action="store_true",
        help="use random point in mask as prompt point instead of the middle point",
    )

    
    remove_list = [] #["Big rect. left", "Big rect. right", "Goal left", "Goal right", "Circle central"]




    geometry_groups_mask = {"Full field": None, "Big rect. left": None, "Big rect. right": None, "Goal left": None, "Goal right": None, "Circle right": None, "Circle left": None, "Circle central": None, "Small rect. left": None, "Small rect. right": None}

    geometry_groups_mask_list = list(geometry_groups_mask.keys())

    
    args = parser.parse_args()

    # Check if the field type is valid
    if args.field_type not in geometry_groups_mask_list:
        raise ValueError(f"Invalid field type. Choose from {geometry_groups_mask_list}.")
    
    # Set the field type
    fielt_type = args.field_type

    multi_mask = args.multi_mask

    remove_full_field = args.remove_full_field

    print_step = args.print_step

    max_epochs = args.max_epochs

    random_point = args.random_point

    base_save_path = args.save_path


    model_save_path = str(args.model_name)
    print("Model save name: ", model_save_path)
    model_save_path = os.path.join(base_save_path, model_save_path)
    print("Model save path: ", model_save_path)

    data = datamaker(data_dir=args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if available
    print("Device: ",device)
    device = str(device)

    sam2_checkpoint = args.checkpoint # path to model weight
    model_cfg = args.config # model config

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device) # load model
    predictor = SAM2ImagePredictor(sam2_model) # load net

    # auto_model = SAM2AutomaticMaskGenerator(sam2_model)
    # auto_model.predictor.model.load_state_dict(torch.load(""))
    # predictor = auto_model.predictor


    predictor.model.sam_mask_decoder.train(args.train_decoder) # enable training of mask decoder 
    predictor.model.sam_prompt_encoder.train(args.train_encoder) # enable training of prompt encoder

    #Hyperparameters
    lerning_rate = 1e-5 # means 0.00001
    weight_decay = 4e-5 # means 0.00004
    optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=lerning_rate,weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device=device) # use mixed precision training





    length = len(data) -1
    total_iou = 0
    mean_iou = 0
    for epoch in range(max_epochs):
        random_int_list = np.random.randint(0, len(data), size=(len(data),))
        print("Epoch: ",epoch)
        print("length of data: ",len(data))

        for itr in range(len(data)):
            #get random number between 0 and length of data
            r_itr = random_int_list[itr] # random number

            with torch.amp.autocast(device_type=device): # cast to mix precision
                    image,mask,input_point, input_label = read_batch(data, r_itr, remove_full_field, remove_list, random_point) # load data batch

                    if image is None: 
                        print("1: Ignore empty image, skipping img nr", itr, "path: ", data[r_itr]["image"])
                        continue # ignore empty batches
                    
                    if mask.shape[0]==0: 
                        print("2: Ignore empty mask, skipping img nr", itr, "path: ", data[r_itr]["image"])
                        continue # ignore empty batches

                    predictor.set_image(image) # apply SAM image encoder to the image

                    # prompt encoding
                    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point,
                                                                                            input_label,
                                                                                            box=None,
                                                                                            mask_logits=None,
                                                                                            normalize_coords=True
                                                                                            )
                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)


                    batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=multi_mask,
                            repeat_image=batched_mode,
                            high_res_features=high_res_features,
                            )
                    
                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1]) #Upscale the masks to the original image resolution

                    # Segmentaion Loss caclulation
                    gt_mask = torch.tensor(mask.astype(np.float32)).cuda() 
                    prd_mask = torch.sigmoid(prd_masks[:, 0]) # Turn logit map to probability map
                    seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # Cross entropy loss

                    # Score loss calculation (intersection over union) IOU
                    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                    union = (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter) # Union calculation
                    iou = inter / union # IOU calculation
                    
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean() # Score loss from predicted scores and IOU
                    loss=seg_loss+score_loss*0.05  # Mix losses

                    # apply back propogation
                    predictor.model.zero_grad() # empty gradient
                    scaler.scale(loss).backward()  # Backpropogate
                    scaler.step(optimizer)
                    scaler.update() # Mix precision


                    # Display results
                    if itr==0: 
                        mean_iou=0
                        total_iou = 0

                    total_iou += iou.sum() / len(iou) # sum IOU for all masks

                    if itr%print_step==0:
                        mean_iou = total_iou / print_step
                        mean_iou_save = str(round(mean_iou.mean().item(), 4))
                        model_save_path_epoch = model_save_path + str(epoch) + ".torch"
                        torch.save(predictor.model.state_dict(), model_save_path_epoch)
                        print("IOU: ", iou, "itr: ", itr)
                        print("model saved as ", model_save_path_epoch + " at step ", itr, " with mean IOU for last", print_step, "steps: ", mean_iou_save, " and current IOU: ", np.mean(iou.cpu().detach().numpy()))
                        mean_iou=0
                        total_iou=0

            
                




                        