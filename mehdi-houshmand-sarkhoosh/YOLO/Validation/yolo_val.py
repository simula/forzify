from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/home/mehdihou/D1/YOLO-Train/Train/Thesis_detection_02/train/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val(data='/home/mehdihou/D1/YOLO-Train/Train/config.yaml',
                    imgsz=1280,
                    batch=18,
                    device=[0,1,2,3,4,5],
                    plots=True
                    )  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category