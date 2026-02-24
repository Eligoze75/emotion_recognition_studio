# Prompt 3 — Live Webcam Inference (Real-Time Emotion Recognition)

## Context

You are working inside the repository `emotion_recognition_studio` (The path of the project repository is `/Users/eligoze/Documents/Die_Projekte/emotion_recognition_studio`).

At this stage, the project already includes:

- A trained YOLO-based emotion recognition model
- Offline inference and evaluation utilities
- Visualization helpers for bounding boxes and emotion predictions

This prompt focuses on implementing real-time emotion recognition using the computer’s webcam.

## Objectives

Implement a live webcam inference pipeline that:

- Captures video frames from the computer camera
- Runs face detection + emotion classification per frame
- Displays results in real time
- Remains stable, readable, and debuggable

Performance should be good enough for demo purposes, not production-optimized.

## Required Deliverables

### 1. Project Structure Additions

Create the following structure if it does not exist:

```graphql
src/
├── webcam/
│   ├── __init__.py
│   ├── stream.py           # Webcam capture loop
│   └── app.py              # High-level live inference application
```

### 2. Webcam Stream Handler

File: `src/webcam/stream.py`

Implement a `WebcamStream` class that:

- Wraps OpenCV `cv2.VideoCapture`
- Supports:
  - Camera index selection
  - Frame resizing
  - Graceful shutdown
- Exposes:
  - `read()` → returns `(ret, frame)`
  - `release()` → frees camera resources
Requirements:
- Handle camera not found errors
- Ensure deterministic cleanup on exit

### 3. Live Inference Application

File: `src/webcam/app.py`

Implement a `LiveEmotionApp` class that:

- Loads:
  - Trained model checkpoint
  - EmotionPredictor (from offline inference)
- Initializes webcam stream
- Runs an inference loop:
  - Capture frame
  - Preprocess frame
  - Detect faces
  - Extract features
  - Predict emotions
  - Visualize output
- Displays the annotated frame in real time

Exit conditions:

- Pressing q
- Keyboard interrupt

### 4. Frame Processing Requirements

For each frame:

- Detect faces using the YOLO backbone
- For each detected face:
  - Draw bounding box
  - Display top-2 emotion predictions
  - Display confidence scores
- Skip frames if:
  - No faces detected
  - Frame read fails

Visual requirements:

- Bounding box tightly aligned to face
- Emotion text above or below the box
- Font size readable at typical webcam resolutions

### 5. Performance Guardrails

To ensure stability:

- Use torch.no_grad()
- Set model to eval() mode
- Optionally:
  - Run inference every N frames (configurable)
- Log FPS to console (optional)

### 6. CLI Interface

The live app must be runnable as:

```bash
python -m src.webcam.app \
  --checkpoint path/to/model.pt \
  --camera_index 0 \
  --img_size 640 \
  --top_k 2
```

### 7. Error Handling & Logging

Fail fast if:

- Checkpoint not found
- Camera unavailable

Log:

- Model load success
- Camera initialization
- Clean shutdown

Use Python `logging`, not `print`.

### 8. Coding Standards

- Follow Google Python Style Guide
- Full type hints
- Clear docstrings
- No hard-coded paths
- No notebook code

## Non-Goals

The following are explicitly out of scope:

- iPhone or mobile camera input
- Web or UI frameworks
- Model retraining
- Performance benchmarking
- Multi-face tracking across frames

## Success Criteria

This prompt is successful when:

- Running the CLI opens the webcam
- Faces are detected live
- Top-2 emotions + scores are shown in real time
- The app exits cleanly without crashing
- The code is modular and reusable

## Final Note

Prioritize correctness and clarity over FPS.
This is a demo-grade system meant to validate the full ML pipeline end-to-end.
