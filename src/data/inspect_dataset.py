"""
Dataset inspection utility for emotion_recognition_studio.

This script performs lightweight sanity checks on the raw facial emotion
dataset before training. It verifies directory structure, image loading,
label consistency, and basic dataset statistics.

Intended to be run once before model training.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd


EMOTIONS = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprised",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def inspect_dataset(data_root: Path) -> None:
    """Run sanity checks on the dataset.

    Args:
        data_root: Path to data/raw/archive directory.
    """
    images_dir = data_root / "images"
    csv_path = data_root / "emotions.csv"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print("✔ Dataset paths found")

    metadata = pd.read_csv(csv_path)
    print(f"✔ Loaded metadata CSV with {len(metadata)} rows")

    emotion_counter: Counter[str] = Counter()
    image_shapes: List[tuple[int, int, int]] = []

    person_dirs = sorted(p for p in images_dir.iterdir() if p.is_dir())
    print(f"✔ Found {len(person_dirs)} person folders")

    for person_dir in person_dirs:
        image_files = [
            p for p in person_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not image_files:
            print(f"⚠ No images found in {person_dir}")
            continue

        for img_path in image_files:
            emotion_name = img_path.stem.lower()

            if emotion_name == "neutral":
                continue  # explicitly excluded

            if emotion_name not in EMOTIONS:
                raise ValueError(
                    f"Unexpected emotion label '{emotion_name}' in {img_path}"
                )

            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")

            image_shapes.append(image.shape)
            emotion_counter[emotion_name] += 1

    print("\n=== Dataset Summary ===")
    print(f"Total usable images: {sum(emotion_counter.values())}")
    print("Class distribution:")
    for emotion in EMOTIONS:
        print(f"  {emotion:10s}: {emotion_counter.get(emotion, 0)}")

    if image_shapes:
        heights = [s[0] for s in image_shapes]
        widths = [s[1] for s in image_shapes]
        channels = {s[2] for s in image_shapes}

        print("\nImage shape stats:")
        print(f"  Height range: {min(heights)}–{max(heights)}")
        print(f"  Width range:  {min(widths)}–{max(widths)}")
        print(f"  Channels:     {sorted(channels)}")

        if len(channels) != 1 or list(channels)[0] != 3:
            print("⚠ Warning: not all images are 3-channel BGR")

    print("\n✔ Dataset inspection completed successfully")


def main() -> None:
    """CLI entrypoint."""
    data_root = Path("data/raw/archive")
    inspect_dataset(data_root)


if __name__ == "__main__":
    main()
