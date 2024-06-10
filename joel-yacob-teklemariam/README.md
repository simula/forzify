
# Joel Yacob Teklemariam - Automatic Detection of Soccer Events using Game Audio and Large Language Models

## Description

This repository contains the code and resources for Joel Yacob's Master thesis, "Automatic Detection of Soccer Events using Game Audio and Large Language Models (LLM)." The project focuses on developing a system that can automatically detect and classify soccer events (such as Goals, Fouls, and Corners) by analyzing game audio and leveraging large language models. Converting the game audio to Automatic speech recognition (ASR) text and then constructing supervised datasets with the help of metadata. 

![Soccer Events](https://docs.google.com/drawings/d/e/2PACX-1vRFhhda8ZqKKjHTB2Rdxe3YoC1DL-gIYsV499LSIctIobeXlJEFj_WfuBCBko8RD-x0VYUK1PCMMiCr/pub?w=730&h=367)

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
- [Installation](#installation)
- [Results](#results)
- [Contributing](#contributing)
- [Contact Information](#contact-information)

## Introduction

Soccer event detection is a crucial task for automated sports analysis, broadcasting, and highlight generation. Traditional methods rely heavily on video analysis, which can be computationally intensive. This project explores the potential of using game audio and advanced language models to detect key soccer events more efficiently.

## Methodology

The methodology consists of several key steps:
1. **Data Collection**: Constructing supervised text classification datasets with different window sizes from the SoccerNet V2 dataset.
2. **Preprocessing**: Cleaning and preparing the ASR data for analysis.
3. **Model Training**: Training various large language models on the datasets to detect and classify events.
4. **Evaluation**: Assessing the performance of the model using evaluation metrics.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/simula/forzify/tree/main/joel-yacob-teklemariam
    cd joel-yacob-teklemariam
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```


## Results

Detailed results and performance metrics of the model are documented in the [Results](results) section. You can find visualizations, accuracy scores, and comparative analyses there.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch`
5. Create a new Pull Request.


## Contact Information

For any questions or inquiries, please contact me at:

- Email: joelyacob99@gmail.com
- GitHub: [JYT8899](https://github.com/JYT8899)

