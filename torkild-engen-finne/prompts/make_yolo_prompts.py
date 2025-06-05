from ultralytics import YOLO
import os
import numpy as np
from argparse import ArgumentParser
if __name__ == "__main__":


    parser = ArgumentParser()

    parser.add_argument(
        "-mp",
        "--model_path",
        required=True,
        type=str,
        help="Path to the YOLO model file",
    )
    args = parser.parse_args()

    model = YOLO(args.model_path)

    def get_all_video_folders():
        base_path = ""
        folders = os.listdir(base_path)
        folders = [os.path.join(base_path, f, "img1") for f in folders if os.path.isdir(os.path.join(base_path, f))]
        return folders


    folders = get_all_video_folders()

    #class names 
    geometry_groups_mask = {"Full field": 0, "Big rect. left": 1, "Big rect. right": 2, "Goal left": 3, "Goal right": 4, "Circle right": 5, "Circle left": 6, "Circle central": 7, "Small rect. left": 8, "Small rect. right": 9}
    features = {'Full field':1, 'Big rect. left':2, 'Big rect. right':2, 'Goal left':3, 'Goal right':3, 'Circle right':4, 'Circle left':4, 'Circle central':5, 'Small rect. left':6, 'Small rect. right':6}
    base_save_path = "YOLO_center_points"
    clss_on_off = {
        "Full field": True,
        "Big rect. left": True,
        "Big rect. right": True,
        "Goal left": True,
        "Goal right": True,
        "Circle right": True,
        "Circle left": True,
        "Circle central": True,
        "Small rect. left": True,
        "Small rect. right": True
    }

    class_list = []
    class_num = 0
    for key, value in geometry_groups_mask.items():
        if clss_on_off[key]:
            class_list.append(value)




    frame_rate = 1
    batch_size = 50
    for video_dir in folders[:]:
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort()

        video_name = video_dir.split("/test/")[1]
        video_name = video_name.split("/img1")[0]


        img_paths = []

        all_center_points = {}
        for ann_frame_idx in range(0, len(frame_names), frame_rate):
            img_paths.append(os.path.join(video_dir, frame_names[ann_frame_idx]))

        frame = 0
        for i in range(0, len(img_paths), batch_size):
            img_paths_batch = img_paths[i:i+batch_size]
            yolo_out = model.predict(
                source=img_paths_batch,
                classes=class_list,
                save=False,
                retina_masks=True
            )
            
            
            for result in yolo_out:
                all_center_points[frame] = {}
                masks = result.masks.data.cpu().numpy()

                classes = result.boxes.cls.cpu().numpy()

                for i, mask in enumerate(masks):
                    ys, xs = np.nonzero(mask)
                
                    x_center = int(np.mean(xs))
                    y_center = int(np.mean(ys))
                    classe = int(classes[i])

                    all_center_points[frame][classe] = [x_center, y_center]
                frame += frame_rate

        save_path = os.path.join(base_save_path, os.path.basename(video_name))

        np.save(save_path, all_center_points)
        print(f"Saved {save_path} with {len(all_center_points)} frames")