# Eirik Eggset - From Pixels to Play: Dribble and Tackle Detection in Football

## Repository Description

This repository contains the Finite State Machine (FSM) used to detect dribbles and tackles from association football (soccer) videos, which is in the `dribbling-detection-algorithm` folder. The full pipeline going from video to predictions is in the `dribbling-detection-pipeline` folder. The full pipeline folder also contains the video sources as well as the annotations.

## Abstract

The automated analysis of fine-grained interactions in sports videos, such as dribbles and tackles in football (soccer), faces significant challenges, primarily the scarcity of large-scale annotated datasets for these specific, non-official events. This thesis confronts this gap by developing and evaluating a novel methodology to automatically detect dribble and tackle events directly from raw football broadcast video.

The core of this solution is a knowledge-driven Finite State Machine (FSM) that models observable player movement patterns. This FSM utilizes player and ball positional data, which are automatically extracted through a comprehensive multi-stage pipeline. This pipeline includes video preprocessing, customized YOLO-based object detection, and annotation interpolation—a technique found to critically enhance both detection accuracy (as measured by mAP) and computational efficiency.

The integrated end-to-end system underwent rigorous evaluation using diverse video content, including event compilations, match highlights, and full broadcast matches. Findings demonstrate the system's capability to identify the target events, achieving an overall precision of approximately 0.31. A key observation was the significant impact of input positional data quality on FSM performance: image-based bounding box coordinates proved more robust and reliable for the FSM than 2D homography-transformed pitch coordinates, largely due to instabilities encountered with the latter on varied footage.

While the achieved precision indicates clear paths for future refinement, particularly in the robustness of the upstream positional data generation. This research firmly establishes a viable foundational methodology for automatically detecting these complex events. Moreover, despite its moderate precision on uncurated footage, the system demonstrated practical effectiveness for semi-automated dataset creation.

Ultimately, this research contributes a novel rule-based FSM detection algorithm, an open-source data processing and annotation pipeline, and valuable insights into the intricacies of analyzing nuanced football interactions. This work provides significant groundwork for future advancements in detailed, automated football event analysis and facilitates the creation of larger, publicly available annotated datasets for these under-explored event types.
