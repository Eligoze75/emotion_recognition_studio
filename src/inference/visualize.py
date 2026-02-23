"""Drawing predictions on images for offline evaluation and sanity checks."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.utils.labels import EMOTION_NAMES

# Deterministic BGR color per emotion (index 0..6).
EMOTION_COLORS: List[Tuple[int, int, int]] = [
    (0, 0, 255),    # anger - red
    (128, 128, 128), # contempt - gray
    (0, 128, 128),   # disgust - teal
    (128, 0, 128),   # fear - purple
    (0, 255, 0),     # happy - green
    (255, 0, 0),     # sad - blue
    (0, 165, 255),   # surprised - orange
]


def _get_emotion_color(emotion_label: str) -> Tuple[int, int, int]:
    """Return BGR color for an emotion label (deterministic)."""
    emotion_label = emotion_label.lower()
    for i, name in enumerate(EMOTION_NAMES):
        if name == emotion_label:
            return EMOTION_COLORS[i]
    return (200, 200, 200)


def draw_predictions(
    image: np.ndarray,
    top_k_labels_and_scores: Sequence[Tuple[str, float]],
    bbox: Optional[Tuple[int, int, int, int]] = None,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """Draw optional face bbox and overlay top-K predictions with confidence.

    Args:
        image: BGR image (H, W, 3), e.g. from cv2.imread.
        top_k_labels_and_scores: List of (emotion_label, confidence).
        bbox: Optional (x, y, w, h) face rectangle. If None, only text is drawn.
        font_scale: OpenCV putText font scale.
        thickness: Line thickness for bbox and text.

    Returns:
        Image with drawings (copy); original is not modified.
    """
    out = image.copy()
    h_im, w_im = out.shape[:2]

    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox must be (x, y, w, h)")
        x1, y1 = int(bbox[0]), int(bbox[1])
        w, h = int(bbox[2]), int(bbox[3])
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    # Draw top-K predictions: clear font, non-overlapping lines.
    line_height = int(25 * font_scale) + 4
    y_offset = 30
    for i, (label, score) in enumerate(top_k_labels_and_scores[:2]):
        text = f"{label}: {score:.2f}"
        color = _get_emotion_color(label)
        cv2.putText(
            out,
            text,
            (10, y_offset + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return out


def save_visualization(
    image: np.ndarray,
    top_k_labels_and_scores: Sequence[Tuple[str, float]],
    save_path: str | Path,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """Draw predictions on image and save to disk.

    Args:
        image: BGR image (e.g. from cv2.imread).
        top_k_labels_and_scores: List of (emotion_label, confidence).
        save_path: Output file path (e.g. .jpg, .png).
        bbox: Optional face bbox (x, y, w, h).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out = draw_predictions(image, top_k_labels_and_scores, bbox=bbox)
    cv2.imwrite(str(save_path), out)
