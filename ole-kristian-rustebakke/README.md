# Ole Kristian Rustebakke - Extracting Player Speed from Football Videos


## Abstract

One of the key metrics for player and team performance in football is speed. Traditionally, measuring player speed either requires extensive human effort or expensive equipment. In this work, we investigate methods for manual and automatic player speed extraction directly from broadcast football videos. We implement a pipeline combining player detection, tracking, field mapping, and position measurement. We experiment with different configurations of the pipeline, concluding that a setting with YOLOv11 for detection, StrongSORT for tracking, PnLCalib for field mapping and keypoint detection, and position measurement through homography yields the best results for our current scope and test datasets. This work serves as a proof-of-concept analytics solution for player speed extraction, which can be adopted by teams of all sizes and means, and built upon by researchers. Our pipeline implementation and two newly curated player speed datasets (Begnadalen and TACDEC++) are openly available for the sports and scientific community.   


## Pipeline Overview

The figures below show the two pipelines that are used for player speed extraction. Starting with an input video sequence of a TV broadcast, we first track the players using bounding boxes, then detect keypoints or lines of a field for either a homography transformation of the field and the players, or to find a perspective grid that can be used to find the player position using the preservation of cross section. The speeds are then calculated by finding the distance travelled between the frames of the sequence. Due to lacking tracking information in some frames, we do not use a strict time interval of 10 frames for every event, but as close as possible to 10 frames and the start of the speed event.

<img width="978" height="582" alt="pipeline1" src="https://github.com/user-attachments/assets/a5c99de6-e334-40cc-bf62-88b7a1771f45" />   

_Figure 1: Pipeline for player speed extraction using homography. (a) Starting with a raw input video from a TV broadcast, (b) Tracking players using bounding boxes, (c) Detecting keypoints or lines on the field, (d) Using the keypoints or lines for a transformation into 2D, so that player positions can be measured and used for speed calculation._   


<img width="788" height="591" alt="pipeline2" src="https://github.com/user-attachments/assets/291d2610-40de-495b-accf-693eafacc0fd" />   

_Figure 2: Pipeline for player speed extraction using perspective grid. (a) Starting with a raw input video from a TV broadcast, (b) Tracking players using bounding boxes, (c) Detecting keypoints and lines on the field, (d) Using the keypoints and lines to make a 2x2 rectangular grid, so that player positions can be measured using the preservation of cross section and used for speed calculation._


## Datasets

We present two new player speed datasets, which are openly available for the sports and scientific communities under: https://zenodo.org/records/17849442

- **Begnadalen** is composed of speed annotations on custom videos which were recorded on a football pitch with cones placed in a grid formation.

<img width="1593" height="191" alt="begnadalen-grids" src="https://github.com/user-attachments/assets/bbfe521b-2db5-42e9-a96c-e8127aa0efe2" />

_Figure 3: Different grids that were used for the Begnadalen dataset. Burgundy marks indicate the cone positions._

- **TACDEC++** is composed of speed annotations on an existing video collection from the Norwegian Eliteserien called [TACDEC](https://zenodo.org/records/10611979). It contains 163 annotated speed events from 151 of the TACDEC videos. In addition, for the special use case of looking at player speeds going into tackles, the dataset contains 163 annotated tackle events, with 142 yellow cards, 19 successful tackles, 1 tackle miss, and 1 foul.


## Code

This repository contains the source code for our work on extracting player speed from football videos, including the pipelines described above. 

### Dependencies

Create and activate environment
```bash
python -m venv myenv
source venv/bin/activate
```

The code runs with python 3.10, but potentially with other versions too. The repo was used in [ml-node](https://www.uio.no/tjenester/it/forskning/kompetansehuber/uio-ai-hub-node-project/it-resources/ml-nodes/), which uses GCCcore. Therefore, the following packages were downloaded from ml-node, compatible with GCCcore:
- **PyTorch**: 1.7.0
- **bzip2**: 1.0.8 (built with GCCcore 10.3.0)
- **GCCcore**: 10.3.0
- **FFmpeg**: 4.4.2

The other packages can be installed with:

```bash
pip install -r requirements.txt
```

### Run

Run ```run_pose.py``` to estimate keypoints (can train the model using ```train_pose.py```, or use the weights [here](https://drive.google.com/file/d/1L19dFxpzKHyzABrjgbb7GpkMg6xxiYUN/view?usp=drive_link)) and get the bounding box ids and coordinates from running [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU), run the speed prediction based on this data with ```speed_extraction.py```. ```speed_extraction.py``` calculates the homography transformation and predicts the player speeds for the events.

Get bounding boxes, keypoint and line predictions from the pipeline [using PnLCalib](https://github.com/eirikeg1/dribbling-detection-pipeline), and run ```speed_extraction_pnl_calib.py``` to get the predicted speeds (```speed_extraction_gamestate.py``` is the same script, but runs on the SoccerNet Gamestate Reconstruction dataset). 


## Results

You can find more detailed results in [Ole Kristian Rustebakke, _Extracting Player Speed from Football Videos_. Master's Thesis, University of Oslo (UiO), 2025.](https://home.simula.no/~paalh/students/2025-UiO-OleKristianRustebakke.pdf)
