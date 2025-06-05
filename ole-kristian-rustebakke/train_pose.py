#import comet_ml
import torch
import time
#import logging
import psutil
import GPUtil
from ultralytics import YOLO
import os
import json

num_gpus = torch.cuda.device_count()
print(num_gpus)
# Set environment variable for Comet ML
#os.environ['COMET_EVAL_LOG_CONFUSION_MATRIX'] = 'true'

# Initialize Logging
#logging.basicConfig(filename='pose_training_log.log', level=logging.INFO)

# Initialize Comet ML with your pose project
#comet_ml.init(project_name='hockey_rink_pose_detection')

# Check CUDA availability
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available with {num_gpus} GPU{'s' if num_gpus > 1 else ''}.")

    for i in range(torch.cuda.device_count()):
        GPU_name = print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

# Define pose models dictionary - using pose-specific models
models_to_load = {
    'yolov8n-pose': {
        'path': 'yolov8n-pose.pt',
        'load': False
    },
    'yolov8s-pose': {
        'path': 'yolov8s-pose.pt',
        'load': False
    },
    'yolov8m-pose': {
        'path': 'yolov8m-pose.pt', #train43, train432: Pose Metrics: mAP50: 0.9908732394366199, mAP50-95: 0.9321311423036402, Box: mAP50: 0.995, mAP50-95: 0.995
        'load': True 
    },
    'yolov8l-pose': {       #train42, train422: Pose Metrics: mAP50: 0.995, mAP50-95: 0.9480612574981206, Box:  mAP50: 0.995, mAP50-95: 0.995
        'path': 'yolov8l-pose.pt', #train52, train522: Pose Metrics: mAP50: 0.9578458239685861, mAP50-95: 0.8640074772225139, Box:  mAP50: 0.995, mAP50-95: 0.995
        'load': False    #train54, train 542: Pose Metrics: mAP50: 0.9860469795740981, mAP50-95: 0.8847166676082509, Box: mAP50: 0.995, mAP50-95: 0.995
    },
    'yolov8x-pose': {
        'path': 'yolov8x-pose.pt',
        'load': False
    }
}

# Example hyperparameters for pose detection
pose_hyperparameters = {
    "data": "/itf-fi-ml/home/olekrus/master/master/config_pose.yaml",  
    "epochs": 500,
    "batch": 16,
    "imgsz": 640,
    "device": [3],  
    "workers": 1,
    "patience": 500,
    "pose": 15.0,  # Pose-specific loss weight
    "kobj": 25.0,   # Keypoint obj loss weight
    "save": True,
    "save_period": 100,
    "cache": False,
    "exist_ok": False,
    "pretrained": False,
    "plots": True,
    "close_mosaic": 0,
    "mosaic": 0.0,
    "fliplr": 0,
}

# Find the model to load
selected_model_path = next((model['path'] for model in models_to_load.values() 
                          if model['load']), None)

if selected_model_path:
    #logging.info(f"Loading pose model from: {selected_model_path}")
    print(f"Loading pose model from: {selected_model_path}")
    
    # Initialize YOLO model for pose detection
    model = YOLO(selected_model_path)
    model.info()

    start_time = time.time()

    # Check resource usage
    cpu_usage = psutil.cpu_percent()
    gpu_usage = GPUtil.getGPUs()[0].load
    #logging.info(f"Training will be started with {GPU_name}")
    print(f"Training will be started with {GPU_name}")

    # Modified validation section in your script
    try:
        # Train the pose model
        results = model.train(**pose_hyperparameters)
        
        # Log training completion
        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        #logging.info(f"Pose detection training completed in {execution_time_minutes:.2f} minutes")
        print(f"Pose detection training completed in {execution_time_minutes:.2f} minutes")
        
        # Validate the model - with proper metrics handling
        try:
            metrics = model.val()
            # Log specific metrics that are available
            if hasattr(metrics, 'pose'):
                # logging.info(f"Pose Metrics:")
                print(f"Pose Metrics:")
                # logging.info(f"mAP50: {metrics.pose.map50}")
                print(f"mAP50: {metrics.pose.map50}")
                # logging.info(f"mAP50-95: {metrics.pose.map}")
                print(f"mAP50-95: {metrics.pose.map}")
                # logging.info(f"Precision: {metrics.pose.precision}")
                print(f"Precision: {metrics.pose.precision}")
                # logging.info(f"Recall: {metrics.pose.recall}")
                print(f"Recall: {metrics.pose.recall}")
            
            if hasattr(metrics, 'box'):
                #logging.info(f"Box Metrics:")
                print(f"Box Metrics:")
                #logging.info(f"mAP50: {metrics.box.map50}")
                print(f"mAP50: {metrics.box.map50}")
                #logging.info(f"mAP50-95: {metrics.box.map}")
                print(f"mAP50-95: {metrics.box.map}")
            
            if hasattr(metrics, 'speed'):
                #logging.info(f"Speed Metrics:")
                print(f"Speed Metrics:")
                #logging.info(f"Preprocess: {metrics.speed['preprocess']}ms")
                print(f"Preprocess: {metrics.speed['preprocess']}ms")
                #logging.info(f"Inference: {metrics.speed['inference']}ms")
                print(f"Inference: {metrics.speed['inference']}ms")
                #logging.info(f"Loss: {metrics.speed['loss']}ms")
                print(f"Loss: {metrics.speed['loss']}ms")
                #logging.info(f"Postprocess: {metrics.speed['postprocess']}ms")
                print(f"Postprocess: {metrics.speed['postprocess']}ms")
                
        except AttributeError as ae:
            #logging.warning(f"Some metrics were not available: {str(ae)}")
            print(f"Some metrics were not available: {str(ae)}")
        except Exception as ve:
            #logging.error(f"Validation error: {str(ve)}")
            print(f"Validation error: {str(ve)}")
            
    except Exception as e:
        #logging.error(f"Training error occurred: {str(e)}")
        print(f"Training error occurred: {str(e)}")
        raise e

else:
    print("No pose model selected for training!")