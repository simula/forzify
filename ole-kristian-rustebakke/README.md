# Ole Kristian Rustebakke - Extracting Player Speed from Football Videos

## Abstract

Football is a sport of passion and entertainment, with a lot of planning going into creating grand moments for the fans to enjoy. This involves statistics and analysis of all the details in the game, and speed is an important quality to measure for any player, as attackers need it to pass the defence, and the defenders need it to recover. In many leagues, advanced equipment such as GPS or multiple high quality cameras are used to measure the position and speed of players, but lower leagues do not necessarily have access to such equipment, and teams scouting another league do not necessarily have access to that league's data. Therefore, measuring speed from a video as seen in a TV-broadcast (or simular) of a match can be useful, as it is available for anyone. In addition, the amount of data currently annotated for the sake of speed extraction is limited, so the curation of a speed dataset is necessary. Manual methods of measuring speeds are tested on the Alfheim dataset, and the original experimental Begnadalen dataset, to find the best method for making the dataset. The dataset curated in this project is based on videos from the TACDEC dataset, and to measure player speeds a perspective grid is used for distance calibration, as this was the best manual method with an average relative error of 11.8\% on the Begnadalen dataset. The curated dataset contains 151 videos, with 163 annotated speed events. In addition, because of the intended additional use of looking at player speeds into tackles, the dataset contains 163 annotated tackle events, with 142 yellow cards, 19 successful tackles, 1 foul and 1 tackle miss. Then, the dataset is tested along with the SoccerNet Gamestate Reconstruction data (SoccerNet GSR), on machine learning pipelines automatically extracting the player speeds. The automatic pipeline performing best used YOLOv11 and StrongSORT for player detection and tracking, with an additional linear interpolation of the detected bounding boxes, and two encoder-decoder convolutional networks, the encoder being HRNetV2-w48, for keypoint and line detection on the field, with the Levenberg-Marquardt method to refine the homography matrix based off these keypoints. The pipeline is referred to as PnLCalib, and on the SoccerNet GSR data it achieved an average relative error of 27.3\%. Since the performance of the pipelines are not good enough to deem them usable in a real match scenario, further improvements are discussed along with possible extensions to the application of the methods in this project. In addition to researching the objectives that are set based on the current state of the issue, code for the different pipelines tested is available, and the dataset is available.

## Code

Code for speed extraction of players in football videos. 

### Dependencies
Create and activate environment
```bash
python -m venv myenv
source venv/bin/activate
```

The code runs with python 3.10, but potentially with other versions too. The repo was used in ml-node (https://www.uio.no/tjenester/it/forskning/kompetansehuber/uio-ai-hub-node-project/it-resources/ml-nodes/), which uses GCCcore. Therefore, the following packages were downloaded from ml-node, compatible with GCCcore:
- **PyTorch**: 1.7.0
- **bzip2**: 1.0.8 (built with GCCcore 10.3.0)
- **GCCcore**: 10.3.0
- **FFmpeg**: 4.4.2

The other packages can be installed with:

```bash
pip install -r requirements.txt
```

### Run
Run ```run_pose.py``` to estimate keypoints (can train the model using ```train_pose.py```, or use the weights at https://drive.google.com/file/d/1L19dFxpzKHyzABrjgbb7GpkMg6xxiYUN/view?usp=drive_link) and get the bounding box ids and coordinates from running Deep-EIoU (https://github.com/hsiangwei0903/Deep-EIoU), and run the speed prediction based on this data with ```speed_extraction.py```. ```speed_extraction.py``` calculates the homography transformation and predicts the player speeds for the events.

Get bounding boxes, keypoint and line predictions from the pipeline using PnLCalib (https://github.com/eirikeg1/dribbling-detection-pipeline), and run ```speed_extraction_pnl_calib.py``` to get the predicted speeds (```speed_extraction_gamestate.py``` is the same script, but runs on the SoccerNet Gamestate Reconstruction data). 