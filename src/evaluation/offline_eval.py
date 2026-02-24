"""Offline evaluation script: run inference on a subset of images and sanity checks.

Run from project root:
  python -m src.evaluation.offline_eval --checkpoint path/to/model.pt --output_dir outputs/offline_eval --num_samples 20
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Reduce chance of segfault on macOS (OpenMP/NumPy/PyTorch/OpenCV conflicts).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

# Ensure project root is on path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.augmentations import get_val_transforms
from src.data.dataset import _collect_samples
from src.inference.predictor import EmotionPredictor
from src.inference.visualize import save_visualization
from src.utils.config import load_config
from src.utils.labels import IDX_TO_LABEL, NUM_CLASSES


def _collect_eval_samples(
    project_root: Path,
    num_samples: int,
    seed: int = 42,
) -> List[Tuple[Path, int]]:
    """Collect up to num_samples (path, label) from data/raw/archive/images."""
    config = load_config(project_root / "configs" / "default.yaml")
    raw_root = project_root / config["dataset"]["raw_archive_root"]
    images_dir = raw_root / config["dataset"]["images_subdir"]
    samples = _collect_samples(images_dir, exclude_neutral=True)
    if not samples:
        return []
    rng = __import__("random").Random(seed)
    rng.shuffle(samples)
    return samples[:num_samples]


def _prediction_entropy(probs: np.ndarray) -> float:
    """Average prediction entropy over the batch (to detect overconfidence)."""
    eps = 1e-12
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    return float(np.mean(ent))


def _class_frequency(predicted_indices: List[int]) -> dict[str, float]:
    """Fraction of predictions per class (by predicted label)."""
    counts: dict[str, int] = {IDX_TO_LABEL[i]: 0 for i in range(NUM_CLASSES)}
    for idx in predicted_indices:
        if 0 <= idx < NUM_CLASSES:
            counts[IDX_TO_LABEL[idx]] += 1
    n = max(1, len(predicted_indices))
    return {k: v / n for k, v in counts.items()}


def run_offline_eval(
    checkpoint_path: Path,
    output_dir: Path,
    num_samples: int = 20,
    project_root: Optional[Path] = None,
) -> None:
    """Load model, run inference on a subset of images, save visualizations and metrics.

    Args:
        checkpoint_path: Path to classifier checkpoint (.pt).
        output_dir: Directory for visualized images and any logs.
        num_samples: Maximum number of images to evaluate.
        project_root: Project root for data paths; if None, uses parent of src/.
    """
    project_root = project_root or _PROJECT_ROOT
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = get_val_transforms()
    samples = _collect_eval_samples(project_root, num_samples)
    if not samples:
        print("No images found under data/raw/archive/images. Exiting.", file=sys.stderr)
        sys.exit(1)

    predictor = EmotionPredictor(checkpoint_path, device=None)
    device = predictor._device

    # Import OpenCV after PyTorch/model load to reduce segfault risk on macOS.
    import cv2

    all_probs: List[np.ndarray] = []
    all_pred_indices: List[int] = []

    print(f"Running inference on {len(samples)} images. Output dir: {output_dir}")
    for i, (img_path, _) in enumerate(samples):
        from PIL import Image
        pil_img = Image.open(img_path).convert("RGB")
        tensor_img = transform(pil_img).unsqueeze(0).to(device)
        _, probs, top_k = predictor.predict_single(tensor_img, top_k=2)
        probs_np = probs.cpu().numpy().squeeze()
        pred_idx = int(np.argmax(probs_np))
        all_probs.append(probs_np)
        all_pred_indices.append(pred_idx)

        top2_str = ", ".join(f"{l}: {s:.2f}" for l, s in top_k)
        print(f"  [{i+1}/{len(samples)}] {img_path.name} -> {top2_str}")

        # Save visualized image (BGR for cv2).
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            img_bgr = np.array(pil_img)[:, :, ::-1].copy()
        out_name = f"{img_path.parent.name}_{img_path.stem}_pred.jpg"
        save_visualization(img_bgr, top_k, output_dir / out_name, bbox=None)

    # Aggregate metrics
    probs_stack = np.stack(all_probs, axis=0)
    mean_entropy = _prediction_entropy(probs_stack)
    class_freq = _class_frequency(all_pred_indices)
    max_conf = float(np.max(probs_stack))
    min_conf = float(np.max(probs_stack, axis=1).min())
    unique_preds = len(set(all_pred_indices))

    print("\n--- Sanity metrics ---")
    print(f"  Mean prediction entropy: {mean_entropy:.4f}")
    print(f"  Class frequency (predicted): {class_freq}")
    print(f"  Max confidence (top-1) range: [{min_conf:.3f}, {max_conf:.3f}]")
    print(f"  Unique predicted classes: {unique_preds}/{NUM_CLASSES}")

    # Warnings
    dominant_class = max(class_freq, key=class_freq.get)
    if class_freq[dominant_class] > 0.7:
        print(f"  [WARNING] One class dominates predictions: {dominant_class} = {class_freq[dominant_class]:.1%}")
    if min_conf > 0.95:
        print("  [WARNING] Confidence is always very high (possible overconfidence).")
    if unique_preds == 1:
        print("  [WARNING] Predictions collapse to a single emotion.")
    print("Done.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline evaluation and sanity checks for the emotion classifier.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained classifier checkpoint (.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/offline_eval"),
        help="Directory for visualized outputs and logs.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Maximum number of images to evaluate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_offline_eval(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )
