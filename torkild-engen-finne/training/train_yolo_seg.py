from ultralytics import YOLO
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()

    # Arguments
    parser.add_argument(
        "-mn",
        "--model_name",
        required=True,
        type=str,
        help="name of the saved model",
        default="yolo11x-seg.pt",
    ) 
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        type=str,
        help="path to config file (e.g. base_config/sam2_hiera_s.yaml)",
        default="config/yolo_segmentation_config.yaml",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        required=False,
        type=int,
        help="number of epochs to train",
        default=30,
    )
    parser.add_argument(
        "-sf",
        "--save_folder",
        required=False,
        type=str,
        help="path to save folder",
        default="",
    )
    parser.add_argument(
        "-n",
        "--name",
        required=False,
        type=str,
        help="name of the model",
        default="yoloSeg_no_name",
    )
    
    parser.add_argument(
        "-is",
        "--img_size",
        required=False,
        type=int,
        help="image size for training",
        default=640,
    )

    parser.add_argument(
        "-f",
        "--fliplr",
        required=False,
        type=int,
        help="whether to flip images horizontally (0 or 1)",
        default=0,
    )

    args = parser.parse_args()

    model_name = args.model_name
    config = args.config
    epochs = args.epochs
    save_folder = args.save_folder
    name = args.name
    img_size = args.img_size
    fliplr = args.fliplr

    model = YOLO(model_name)

    model.train(data=config, epochs=epochs, batch=8, imgsz=img_size, device=0, workers=8, project=save_folder, name=name, fliplr=fliplr)