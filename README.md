# Emotion Recognition Studio

A training pipeline for **facial emotion recognition** that uses a pretrained **YOLOv8** model as a frozen visual backbone and trains a custom **emotion classification head** on top. This repository covers dataset handling, augmentation, model setup, and training only; it does not include webcam inference or real-time visualization.

---

## Project Overview

- **Goal:** Train an emotion classifier on a small facial emotion dataset by reusing YOLOv8’s backbone and neck as a fixed feature extractor and learning only a lightweight head.
- **Scope (MVP):** Repository structure, dataset loading, augmentations, YOLOv8 backbone integration (feature extraction only), training scripts, configuration, environment setup, and documentation. No YOLOv8 training/fine-tuning, no face tracking, no webcam or real-time inference.

---

## Dataset

Raw data lives under `data/raw/archive/`:

- **`emotions.csv`** — Metadata (e.g. person id, gender, age, country).
- **`images/`** — One subfolder per person (`0/`, `1/`, …, `18/`). Inside each folder, images are named by emotion: `Anger.jpg`, `Contempt.jpg`, `Disgust.jpg`, `Fear.jpg`, `Happy.jpg`, `Neutral.jpg`, `Sad.jpg`, `Surprised.jpg`.

**Target emotions (7 classes):** anger, contempt, disgust, fear, happy, sad, surprised.  
**Neutral** is excluded from training. Labels are derived from filenames. The dataset is small (~19 people); strong augmentation is used to improve robustness.

---

## Architecture (YOLOv8-Based)

1. **Backbone + neck (frozen):** A pretrained YOLOv8 model (e.g. `yolov8n.pt`) is loaded via the Ultralytics API. Only the backbone and neck are used; all their parameters are frozen. The detection head is not used; no bounding box prediction or NMS is performed.
2. **Feature extraction:** Input images (e.g. 640×640) are passed through the frozen backbone and neck to obtain a feature map per image.
3. **Emotion head (trainable):** A small classifier (global pooling, fully connected layers, optional dropout) maps the feature map to 7-class logits. Only this head is trained; loss is cross-entropy.

Configuration (model id, classifier hidden dims, dropout, etc.) is read from `configs/default.yaml`.

---

## Training Workflow

1. **Configuration:** All settings (paths, split ratio, batch size, learning rate, epochs, augmentation flags, dropout, seed, output paths) are read from `configs/default.yaml`. No hard-coded paths in code.
2. **Seeding:** A fixed random seed (configurable) is set for reproducibility.
3. **Data:** The dataset traverses `data/raw/archive/images`, builds (image path, label) pairs (excluding Neutral), and splits into train/validation according to the configured ratio. Augmentations (random grayscale, salt-and-pepper, rotation, crop, flip, brightness/contrast) are applied only to training data.
4. **Model:** The YOLOv8 backbone is loaded and frozen; the emotion classifier is initialized to match the backbone’s feature size. Only the classifier is updated.
5. **Training:** Standard cross-entropy training; training and validation loss are logged per epoch.
6. **Checkpoints:** The script saves the classifier weights, the emotion–label mapping, and a snapshot of the config used for the run.

---

## Environment Setup

- **Python 3.12** recommended.
- Create and activate the Conda environment:

```bash
cd emotion_recognition_studio
conda env create -f environment.yml
conda activate emotion_recognition_studio
```

- The `environment.yml` includes Python, PyTorch, torchvision, ultralytics, OpenCV, numpy, pandas, PyYAML, and albumentations. CUDA versions are not pinned; install a CUDA-enabled PyTorch build separately if you need GPU support.

---

## How to Run Training

From the **project root** (`emotion_recognition_studio/`):

```bash
conda activate emotion_recognition_studio
python -m src.training.train
```

Or with a custom config path:

```bash
python -m src.training.train configs/my_config.yaml
```

Outputs (classifier weights, label mapping, config snapshot) are written to the directory specified in the config (default: `outputs/`).

---

## MVP Scope and Limitations

- **In scope:** Repo structure, dataset handling, augmentations, YOLOv8 backbone integration (features only), training pipeline, config, env, and README.
- **Out of scope (not implemented):** Webcam inference, real-time visualization, YOLOv8 training or fine-tuning, and face tracking. The dataset is small (~19 people); expect limited generalization to new identities.

---

## License

MIT License.
