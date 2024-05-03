# Real-time Emotion Detection using Facial Recognition and Deep Learning

This project explores the application of real-time emotion detection using a combination of facial recognition and deep learning techniques. It leverages the capabilities of OpenCV for efficient face detection within a live webcam video stream. Subsequently, DeepFace is employed to predict the dominant emotion on the detected face. The predicted emotion, along with its associated confidence score, is then overlaid directly onto the video frame, enabling real-time visualization and analysis of emotional expressions. This system has the potential to be applied in various domains, such as human-computer interaction research or studies investigating emotional responses to stimuli.

## Key Features

- Real-time emotion detection from live webcam video streams.
- Utilization of OpenCV for accurate and efficient face detection.
- Integration of DeepFace for predicting dominant emotions on detected faces.
- Overlay of predicted emotions and confidence scores on video frames for visualization.
- Potential applications in human-computer interaction research and emotional response studies.

## Dependencies

- deepface
- keras
- opencv-python

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/thanhjash/emotion-detection.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Navigate to the project directory:

    ```bash
    cd Emotion-Detection
    ```

2. Run the main script:

    ```bash
    python main.py
    ```

3. Press 'q' to exit the application.

## License



## Acknowledgments

- The developers of OpenCV and DeepFace for their invaluable contributions to computer vision and deep learning technologies.
