# SmartCrop: AI-Based Cropping of Sports Videos

Developed by Sayed Mohammad Majidi Dorcheh as part of a Master's thesis in Applied Computer and Information Technology at Oslo Metropolitan University.

## Abstract

Sports multimedia is among the most prominent types of content distributed across social media today, necessitating the retargeting of videos to diverse aspect ratios for appropriate representation on different platforms. SmartCrop is an automated video cropping pipeline designed to curate content tailored to custom aspect ratios suitable for various social media platforms. The system utilizes a Point of Interest (POI) tracking mechanism, with the soccer ball or ice hockey puck serving as the primary POI. Scene detection is achieved through TransNetV2 (a machine learning model) and PySceneDetect (a Python library), while a You Only Look Once (YOLO)v8-medium model, fine-tuned on custom soccer and ice hockey datasets, detects the POIs. Inaccurate detections are filtered through outlier detection methods, and interpolation or smoothing modules are applied when the POI is not visible, specific to either soccer or ice hockey. 

Objective evaluations of each module’s performance within both the SmartCrop-S and SmartCrop-H pipelines have been conducted, validating the proposed architecture in terms of accuracy, efficiency, precision, and error metrics such as RMSE and MAE. These evaluations confirm that the system meets high standards for performance and is effectively adapted to the dynamic requirements of sports video analysis. For the SmartCrop-S pipeline, a crowdsourced subjective user study assessing alternative cropping approaches from 16:9 to 1:1 and 9:16 aspect ratios confirms that the proposed approach significantly enhances the end-user Quality of Experience (QoE). For the SmartCrop-H pipeline, three distinct subjective user studies were conducted: the first to determine the optimal alpha value for the smoothing module, the second to show that the SmartCrop output using the full functionality of the SmartCrop-H pipeline performed better than other alternatives, and the third, designed for competitor analysis, compared SmartCrop-H with professional video editing tools. This last study demonstrated that SmartCrop-H performs on par with, or even surpasses, professional tools in terms of output quality.

## Available Resources

The weights of object detection models for soccer and ice hockey are available at: [SportsVision-YOLO](https://github.com/forzasys-students/SportsVision-YOLO).

## Publications

- **Soccer on Social Media:** [DOI: 10.48550/arXiv.2310.12328](https://doi.org/10.48550/arXiv.2310.12328)
- **SmartCrop: AI-based Cropping of Soccer Videos:** [DOI: 10.1109/ISM59092.2023.00009](https://doi.org/10.1109/ISM59092.2023.00009)
- **AI-Based Sports Highlight Generation for Social Media:** [DOI: 10.1145/3638036.3640799](https://doi.org/10.1145/3638036.3640799)
- **AI-Based Cropping of Sport Videos using SmartCrop:** Accepted for IJSC Journal
- **AI-Based Cropping of Ice Hockey Videos for Different Social Media Representations:** Under review for IEEE Access journal
- **SmartCrop-H: AI-Based Cropping of Ice Hockey Videos:** [DOI: 10.1145/3625468.3652195](https://doi.org/10.1145/3625468.3652195)
- **AI-Based Cropping of Soccer Videos for Different Social Media Representations:** [DOI: 10.1007/978-3-031-53302-0_22](https://doi.org/10.1007/978-3-031-53302-0_22)

## Contributions

The SmartCrop project is developed by Sayed Mohammad Majidi Dorcheh as part of a Master's thesis in Applied Computer and Information Technology at Oslo Metropolitan University. This research is a collaborative effort with SimulaMet, OsloMet, and Forzasys. Contributions are welcome, and you can contribute by opening issues or submitting pull requests on the GitHub repository.

## License

This project is licensed under the AGPL-3.0 License. See the LICENSE file for details.

## SmartCrop: Automated Video Cropping for Social Media

- **Accurate POI Tracking:** Utilizing YOLOv8-medium for detecting soccer balls and ice hockey pucks.
- **Advanced Scene Detection:** Leveraging TransNetV2 and PySceneDetect for precise scene segmentation.
- **Performance Validated:** Objective evaluations with RMSE and MAE metrics.
- **User-Centric:** Enhanced QoE confirmed through extensive user studies.

## Get Started

While the full SmartCrop pipeline is not publicly available, you can access the weights of our object detection models [here](#).

## Collaborative Effort

Developed in collaboration with:

- SimulaMet
- OsloMet
- Forzasys

## Demo

Watch the demo video: (https://www.youtube.com/watch?v=rMmYOCM-k7A&ab_channel=SimulaMet-HOST)

