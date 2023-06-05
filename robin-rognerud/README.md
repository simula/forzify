# AI-based clipping for booking events in soccer

## Abstract
Manual clipping is currently the gold standard for extracting highlight clips from soccer games. However, it is a costly, tedious, and time-consuming task that is impractical and unfeasible for, at least, lower-league games with limited resources. Today, manual clipping is either used to trim away undesired video frames in a custom manner per video (high cost), or by employing a preset time interval leading to non-custom static clips (low quality). To address this issue, this thesis aims to automate the generation of highlight clips for booking events, in a custom and dynamic manner. In our pipeline, we will implement logo detection, scene boundary detection, and multimedia processing. We will also do a statistical analysis of current highlight clips, and perform a subjective evaluation. Full games are used as input, where detection modules will locate possible timestamps to produce an intruguing highlight clip. Through experimentation and results from state-of-the-art research, we will use neural network architectures and different datasets to suggest two models that can automatically detect appropriate timestamps for extracting booking events. These models are evaluated both qualitatively and quantitatively, demonstrating high accuracy in detecting logo and scene transitions and generating viewer-friendly highlight clips. When looking at state-of-the-art research and the results in the thesis, the conclusion is that automating the soccer video clipping process has significant potential.

## Train the model
You must add your own data for training in the images folder. And it must be in this structure:

```
.
└── images/
    ├── train/
    │   ├── logo
    │   └── game
    ├── validation/
    │   ├── logo
    │   └── game
    └── test/
        ├── logo
        └── game
```

## Run pipeline

### Clone to repository
Clone the repository to your chosen folder 

```
git clone 
```

### Prerequisites

#### Versions
 
| Package       | Version  |
|---------------|----------|
| Keras         | 2.10.0   |
| Tensorflow    | 2.10.0    |
| MoviePy       | 1.0.3    |
| opencv-python | 4.5.5.62 |
| matplotlib    | 3.7.0    |
| Numpy         | 1.24.2   |

#### Add TransNetV2
We need to add TransNetV2 in under the sbd_classification folder for the pipeline to work.

Go to https://github.com/soCzech/TransNetV2 and clone the repo, then clone it into the sbd_classification folder

```
git clone https://github.com/soCzech/TransNetV2.git
```

#### Data folder
Then you must switch the path_to_game string in pipeline.py, to where you store your full games and text-files.
And you must switch filename and txt strings in the same file to the game you wish to clip.

### Run
To run the pipeline, you simply run pipeline.py

### Output
The output is the clipped games, and can be found in the new_clips folder.


### Potential TransNetV2 error
You may need to install the python package TransNetV2 offers. Do this by:

From the root folder of TransNetV2 under sbd_classification, do:
   ```
   python setup.py install
   ```




