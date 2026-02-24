# Prompt 2 — Offline Evaluation & Inference Sanity Checks

## Context

You are working inside the repository emotion_recognition_studio (The path of the project repository is `/Users/eligoze/Documents/Die_Projekte/emotion_recognition_studio`), which already contains:

- A training pipeline for an emotion classifier built on top of a YOLOv8 backbone
- A trained model checkpoint saved to disk
- A small facial emotion dataset with heavy data augmentation
- A modular project structure under src/

Before moving to live camera inference, the goal of this prompt is to implement offline evaluation and inference sanity checks to validate correctness, robustness, and usability of the trained model.

## Objectives

Implement a minimal but reliable offline evaluation & inference layer that:

1. - Loads a trained model checkpoint correctly
2. Runs inference on:
    - Individual images
    - Small image batches
3. - Verifies output correctness (shape, values, labels)
4. - Produces human-readable visual outputs
5. - Surfaces common failure modes early

This is not a full benchmark or research evaluation — it is a sanity and validation stage.

## Required Deliverables

### 1. Project Structure Additions

Create the following structure if it does not already exist:

```graphql
src/
├── inference/
│   ├── __init__.py
│   ├── predictor.py        # Model loading + forward inference
│   └── visualize.py        # Drawing predictions on images
│
├── evaluation/
│   ├── __init__.py
│   └── offline_eval.py     # Dataset-level sanity evaluation
│
├── utils/
│   └── labels.py           # Emotion label mappings
```

### 2. Emotion Label Mapping

Create a single source of truth for emotion labels.

File: `src/utils/labels.py`

Requirements:

- Define the 7 emotion classes (excluding neutral)
- Provide:
  - IDX_TO_LABEL
  - LABEL_TO_IDX
- Ensure ordering matches the training configuration

### 3. Model Loading & Inference

File: `src/inference/predictor.py`

Implement a EmotionPredictor class that:

- Loads:
  - YOLO backbone
  - Emotion classification head
  - Saved checkpoint
- Switches model to eval() mode
- Supports inference on:
  - Single image tensor
  - Batch of image tensors
- Returns:
  - Raw logits
  - Softmax probabilities
  - Top-K predictions (default K=2)
- Key requirements:
  - No gradients (torch.no_grad)
  - Device-aware (CPU / CUDA)
  - Clear, explicit tensor shape checks

### 4. Visualization Utilities

File: `src/inference/visualize.py`

Implement utilities to:

- Draw bounding boxes for detected faces
- Overlay:
  - Top-2 predicted emotions
  - Corresponding confidence scores
- Use OpenCV (cv2) for drawing
- Support saving images to disk

Visualization requirements:

- Clear font
- Non-overlapping text
- Deterministic colors per emotion (optional but recommended)

### 5. Offline Evaluation Script

File: `src/evaluation/offline_eval.py`

This script should:

- Load a small subset of images `from data/raw/archive/images`
- Run inference on them
- For each image:
  - Print top-2 predictions + scores
  - Save a visualized output image
- Compute simple aggregate metrics:
  - Prediction entropy (to detect overconfidence)
  - Class frequency distribution
- Log warnings if:
  - One class dominates predictions
  - Confidence is always near 1.0
  - Predictions collapse to a single emotion

This is a sanity evaluator, not a research evaluator.

### 6. CLI Usage

The evaluation script must be runnable as:

```bash
python -m src.evaluation.offline_eval \
  --checkpoint path/to/model.pt \
  --output_dir outputs/offline_eval \
  --num_samples 20
```

### 7. Coding Standards

- Follow Google Python Style Guide
- Use:
  - Type hints everywhere
  - Docstrings for all public functions
- No hard-coded paths
- No notebook code
- No training logic duplication

## Non-Goals (Important)

The following are explicitly out of scope for this prompt:

- Live camera input
- Real-time performance optimization
- Model retraining
- Hyperparameter tuning
- Deployment concerns

Those will be handled in later prompts.

## Success Criteria

This prompt is successful when:

- A trained checkpoint can be loaded without errors
- Inference runs deterministically on images
- Outputs are interpretable and visually correct
- Obvious modeling or data issues are detected early
- The system is ready to be extended to live camera input

## Final Note

Prioritize clarity, debuggability, and correctness over performance or abstraction.
Assume this code will be used both by humans and AI agents in later stages.
