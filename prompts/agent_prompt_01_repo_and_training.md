# AI Agent Prompt — Repository Structure & Training Pipeline (YOLOv8)

Project: emotion_recognition_studio

You are an AI software engineering agent acting as a senior machine learning engineer.
Your task is to initialize the repository structure and implement the training-side
code for a facial emotion recognition project using YOLOv8 as a frozen visual backbone.

Follow ALL instructions carefully. Do not invent requirements not explicitly stated.

The path of the project repository is /Users/eligoze/Documents/Die_Projekte/emotion_recognition_studio

---

## 1. Project Overview

Project name: **emotion_recognition_studio**

Goal:
Build a training pipeline for a facial emotion recognition model that uses a
pretrained YOLOv8 model as a frozen feature extractor and trains a custom emotion
classification head on top.

This prompt ONLY covers:

- Repository structure
- Dataset handling
- Data augmentation
- YOLOv8 backbone integration (feature extraction only)
- Training scripts
- Configuration files
- Environment setup
- README documentation

Do NOT implement:

- Webcam inference
- Real-time visualization
- YOLOv8 training or fine-tuning
- Face tracking

---

## 2. Existing Dataset (IMPORTANT)

The repository already contains raw data at:

data/raw/archive/
├── emotions.csv
└── images/
    ├── 0/
    ├── 1/
    ├── ...
    └── 18/

Each numbered folder corresponds to one person.
Inside each folder there are images named by emotion:

- Anger.jpg
- Contempt.jpg
- Disgust.jpg
- Fear.jpg
- Happy.jpg
- Neutral.jpg
- Sad.jpg
- Surprised.jpg

Notes:

- The dataset is SMALL (~19 people)
- Each person has images for all emotions
- The CSV file contains metadata about each person
- Emotion labels must be inferred from image filenames
- The `Neutral` emotion should be EXCLUDED from training

Target emotions (7 classes):

- anger
- contempt
- disgust
- fear
- happy
- sad
- surprised

---

## 3. Repository Structure to Create

Create the following structure exactly:
emotion_recognition_studio/
│
├── data/
│ ├── raw/
│ │ └── archive/ # already exists, DO NOT modify raw files
│ └── processed/
│   ├── train/
│   └── val/
│
├── src/
│ ├── data/
│ │ ├── dataset.py
│ │ └── augmentations.py
│ │
│ ├── models/
│ │ ├── yolo_backbone.py
│ │ └── emotion_classifier.py
│ │
│ ├── training/
│ │ └── train.py
│ │
│ ├── utils/
│ │ ├── config.py
│ │ └── seed.py
│
├── configs/
│ └── default.yaml
│
├── environment.yml
├── README.md
└── .gitignore

---

---

## 4. Coding Standards

All Python code MUST:

- Follow Google Python Style Guide
- Use type hints
- Include module-level and function-level docstrings
- Avoid hard-coded paths
- Read configuration values from `configs/default.yaml`
- Be deterministic when a seed is provided

---

## 5. Dataset Handling

### dataset.py

Implement a PyTorch Dataset that:

- Traverses `data/raw/archive/images`
- Builds `(image_path, label)` pairs
- Excludes `Neutral.jpg`
- Maps emotion strings → integer labels
- Supports train/validation split via config
- Returns `(image_tensor, label)`

Emotion mapping must be deterministic and saved during training.

---

### augmentations.py

Implement strong on-the-fly augmentations using torchvision or albumentations:

Required augmentations:

- Random grayscale
- Salt-and-pepper noise
- Random rotation (±10–15 degrees)
- Random crop / cutout
- Horizontal flip
- Brightness & contrast jitter

Constraints:

- Augmentations applied ONLY in training mode
- Easily toggled via config
- Validation data must NOT be augmented

---

## 6. YOLOv8 Backbone Integration

### yolo_backbone.py

Implement a YOLOv8 feature extractor with the following behavior:

- Load a pretrained YOLOv8 model via the Ultralytics API
- Use YOLOv8 **backbone + neck only**
- Freeze ALL YOLO parameters
- Disable detection head usage
- Extract intermediate feature maps suitable for classification

Design requirements:

- Wrap YOLOv8 inside a clean Python class
- Expose a `forward(images)` method returning feature tensors
- Do NOT perform bounding box prediction
- Do NOT apply non-max suppression

Leave TODO comments where YOLO internals are accessed.

---

## 7. Emotion Classification Head

### emotion_classifier.py

Implement a lightweight classifier:

- Input: YOLOv8 feature tensor
- Architecture:
  - Global pooling (if needed)
  - Fully connected layers
  - Dropout (optional, configurable)
- Output: 7-class logits
- No softmax in forward pass

Classifier must be trainable independently of YOLO.

---

## 8. Training Pipeline

### train.py

Implement a full training pipeline that:

- Loads configuration from YAML
- Sets all random seeds
- Initializes:
  - YOLOv8 backbone (frozen)
  - Emotion classifier head
- Creates dataloaders
- Extracts features using YOLOv8
- Trains ONLY the emotion classifier
- Uses cross-entropy loss
- Logs:
  - Training loss per epoch
  - Validation loss per epoch
- Saves:
  - Classifier weights
  - Emotion label mapping
  - Config snapshot used for training

No experiment tracking frameworks required.

---

## 9. Configuration File

### configs/default.yaml

Must include:

- Dataset paths
- Train/val split ratio
- Batch size
- Learning rate
- Number of epochs
- Augmentation flags
- Dropout rate
- Random seed
- Output directories

---

## 10. Environment Setup

Create an `environment.yml` with:

- Python 3.12
- PyTorch
- torchvision
- ultralytics
- OpenCV
- numpy
- pandas
- PyYAML
- albumentations (optional)

Do NOT pin CUDA versions.

---

## 11. README.md Requirements

README must include:

- Project overview
- Dataset description
- YOLOv8-based architecture explanation
- Training workflow
- Environment setup instructions
- How to run training
- MVP scope and limitations
- License section (MIT)

Tone:

- Professional
- Clear
- Data-scientist oriented

---

## 12. Deliverables

You must:

- Create all required files
- Populate them with clean, working code
- Follow Google Python Style strictly
- Leave TODOs where YOLOv8 internals are abstracted
- Avoid implementing inference or webcam logic

End of instructions.
