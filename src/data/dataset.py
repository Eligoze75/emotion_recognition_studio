"""PyTorch Dataset for the facial emotion archive.

Traverses data/raw/archive/images, builds (image_path, label) pairs,
excludes Neutral, and maps emotion filenames to integer labels.
Train/validation split is driven by config (val_ratio).
"""

import random
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

# Target emotions (7 classes); Neutral is excluded.
EMOTION_NAMES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprised",
]
FILENAME_TO_EMOTION = {
    "anger": "anger",
    "contempt": "contempt",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprised",
}
# Deterministic mapping: emotion name -> label index (0..6), excluding neutral.
EMOTION_TO_LABEL = {name: i for i, name in enumerate(EMOTION_NAMES)}


def get_emotion_label_mapping() -> dict[str, int]:
    """Return deterministic emotion string -> integer label mapping (7 classes).

    Returns:
        Dict mapping emotion name (e.g. 'anger') to label index 0..6.
        'neutral' is not included.
    """
    return dict(EMOTION_TO_LABEL)


def _collect_samples(
    images_root: Path,
    exclude_neutral: bool = True,
) -> list[tuple[Path, int]]:
    """Collect (image_path, label) pairs from archive/images.

    Each numbered subdir (0, 1, ...) contains images named by emotion,
    e.g. Anger.jpg, Happy.jpg. Neutral.jpg is skipped when exclude_neutral.

    Args:
        images_root: Path to data/raw/archive/images.
        exclude_neutral: If True, skip Neutral.jpg.

    Returns:
        List of (absolute image path, integer label).
    """
    samples: list[tuple[Path, int]] = []
    if not images_root.is_dir():
        return samples

    for person_dir in sorted(images_root.iterdir(), key=lambda p: p.name):
        if not person_dir.is_dir():
            continue
        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = img_path.stem.lower()
            emotion = FILENAME_TO_EMOTION.get(stem)
            if emotion is None:
                continue
            if exclude_neutral and emotion == "neutral":
                continue
            label = EMOTION_TO_LABEL[emotion]
            samples.append((img_path.resolve(), label))
    return samples


def build_train_val_samples(
    config: dict[str, Any],
    project_root: Path,
    seed: Optional[int] = None,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Build train and validation sample lists from config.

    Args:
        config: Full config dict; uses dataset.raw_archive_root, dataset.images_subdir,
            dataset.val_ratio, dataset.exclude_neutral.
        project_root: Project root directory for resolving paths.
        seed: Random seed for split reproducibility; if None, split is non-deterministic.

    Returns:
        (train_samples, val_samples) each a list of (image_path, label).
    """
    raw_root = project_root / config["dataset"]["raw_archive_root"]
    images_dir = raw_root / config["dataset"]["images_subdir"]
    val_ratio = config["dataset"]["val_ratio"]
    exclude_neutral = config["dataset"].get("exclude_neutral", True)

    samples = _collect_samples(images_dir, exclude_neutral=exclude_neutral)
    if not samples:
        return [], []

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(samples)
    else:
        random.shuffle(samples)

    n_val = max(0, int(len(samples) * val_ratio))
    n_train = len(samples) - n_val
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    return train_samples, val_samples


class EmotionDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset of face images with emotion labels (7 classes, no neutral)."""

    def __init__(
        self,
        image_paths_and_labels: list[tuple[Path, int]],
        transform: Optional[callable] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            image_paths_and_labels: List of (image_path, label) as produced
                by _collect_samples.
            transform: Optional callable (image -> tensor); applied in __getitem__.
        """
        self.samples = image_paths_and_labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Minimal default: to tensor (caller may expect tensor)
            import torchvision.transforms as T
            image = T.ToTensor()(image)
        return image, label
