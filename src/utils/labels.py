"""Single source of truth for emotion label mappings.

Ordering matches the training configuration and dataset (7 classes, neutral excluded).
"""

from typing import Dict

# 7 emotion classes in deterministic order (matches training).
EMOTION_NAMES: tuple[str, ...] = (
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprised",
)

# Index (0..6) -> emotion label string.
IDX_TO_LABEL: Dict[int, str] = {i: name for i, name in enumerate(EMOTION_NAMES)}

# Emotion label string -> index (0..6).
LABEL_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(EMOTION_NAMES)}

NUM_CLASSES: int = len(EMOTION_NAMES)
