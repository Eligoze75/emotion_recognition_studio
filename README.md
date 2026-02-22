# Emotion Recognition Studio

Emotion Recognition Studio is a computer vision project focused on real-time facial emotion recognition using a pretrained YOLO model as a feature extractor.

The system is trained on a small facial emotion dataset and enhanced through aggressive data augmentation techniques. Once trained, the model runs live on webcam input, detecting faces and predicting emotional states in real time.

## Supported Emotions

- anger
- contempt
- disgust
- fear
- happy
- sad
- surprised

## Project Goals

- Use YOLO as a visual backbone for face detection and feature extraction
- Fine-tune a lightweight emotion classifier on top of YOLO features
- Handle small datasets using strong data augmentation
- Perform real-time inference on live webcam input
- Visualize detected faces and top emotion predictions with confidence scores

## Tech Stack

- Python
- PyTorch
- OpenCV
- YOLO (pretrained)

## Status

🚧 Work in progress — MVP under active development
