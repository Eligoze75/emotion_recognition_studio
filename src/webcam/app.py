"""Live emotion recognition from webcam: face detection + classifier + visualization."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# Project root on path when run as module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.augmentations import get_val_transforms
from src.inference.predictor import EmotionPredictor
from src.inference.visualize import _get_emotion_color
from src.webcam.stream import WebcamStream

logger = logging.getLogger(__name__)

# Default Haar cascade for face detection (bundled with OpenCV).
_DEFAULT_FACE_CASCADE_PATH = (
    Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
)


def _detect_faces(
    frame: np.ndarray,
    cascade_path: Optional[Path] = None,
) -> List[Tuple[int, int, int, int]]:
    """Return list of face bounding boxes (x, y, w, h) in the BGR frame."""
    path = cascade_path or _DEFAULT_FACE_CASCADE_PATH
    if not path.exists():
        logger.warning("Haar cascade not found at %s; no face detection.", path)
        return []
    cascade = cv2.CascadeClassifier(str(path))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]


def _preprocess_face_crops(
    frame_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    transform: torch.nn.Module,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Crop faces from frame, resize/normalize, return batch tensor (B, 3, 640, 640)."""
    if not boxes:
        return None
    crops: List[torch.Tensor] = []
    for (x, y, w, h) in boxes:
        crop_bgr = frame_bgr[y : y + h, x : x + w]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        t = transform(pil).unsqueeze(0)
        crops.append(t)
    batch = torch.cat(crops, dim=0).to(device)
    return batch


def _draw_face_predictions(
    frame: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    top_k_per_face: List[List[Tuple[str, float]]],
    font_scale: float = 0.55,
    thickness: int = 2,
) -> None:
    """Draw bounding boxes and top-2 emotion labels on frame (in-place)."""
    for (x, y, w, h), top_k in zip(boxes, top_k_per_face):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        line_height = int(22 * font_scale) + 2
        # Draw text above the box if room, else below.
        text_y_base = max(y - 8, line_height + 2) if y > line_height + 4 else y + h + line_height
        for i, (label, score) in enumerate(top_k[:2]):
            text = f"{label}: {score:.2f}"
            color = _get_emotion_color(label)
            ty = text_y_base + i * line_height
            cv2.putText(
                frame,
                text,
                (x, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )


class LiveEmotionApp:
    """Runs live webcam inference: face detection + emotion prediction + display."""

    def __init__(
        self,
        checkpoint_path: Path,
        camera_index: int = 0,
        img_size: int = 640,
        top_k: int = 2,
        infer_every_n_frames: int = 1,
        config_path: Optional[Path] = None,
    ) -> None:
        """Load model and prepare for capture.

        Args:
            checkpoint_path: Path to classifier checkpoint (.pt).
            camera_index: Webcam device index.
            img_size: Model input size (height and width).
            top_k: Number of top emotions to show per face.
            infer_every_n_frames: Run model every N frames (1 = every frame).
            config_path: Optional config YAML; default next to checkpoint.
        """
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self._img_size = 640  # Model expects 640x640; do not change without retraining.
        self._top_k = top_k
        self._infer_every_n = max(1, infer_every_n_frames)
        self._frame_count = 0
        self._last_boxes = []
        self._last_top_k: List[List[Tuple[str, float]]] = []

        self._predictor = EmotionPredictor(
            checkpoint_path,
            config_path=config_path,
            device=None,
        )
        logger.info("Model loaded: %s", checkpoint_path)

        self._transform = get_val_transforms()

        # Capture at a reasonable size for face detection and display.
        display_w = img_size if img_size >= 320 else 640
        display_h = int(display_w * 0.75)
        self._stream = WebcamStream(
            camera_index=camera_index,
            target_size=(display_w, display_h),
        )

    def _run_inference(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[List[Tuple[str, float]]]]:
        """Detect faces, run emotion model, return boxes and top-k per face."""
        boxes = _detect_faces(frame)
        if not boxes:
            return [], []
        batch = _preprocess_face_crops(
            frame,
            boxes,
            self._transform,
            self._predictor._device,
        )
        if batch is None:
            return boxes, [[] for _ in boxes]
        with torch.no_grad():
            _, _, top_k_list = self._predictor.predict(batch, top_k=self._top_k)
        return boxes, top_k_list

    def run(self) -> None:
        """Main loop: capture, detect, predict, display. Exit on 'q' or Ctrl+C."""
        logger.info("Starting live loop. Press 'q' to quit.")
        t_start = time.perf_counter()
        n_frames = 0
        try:
            while True:
                ret, frame = self._stream.read()
                if not ret or frame is None:
                    logger.warning("Frame read failed; skipping.")
                    continue
                n_frames += 1
                self._frame_count += 1
                run_model = (self._frame_count % self._infer_every_n) == 0
                if run_model:
                    self._last_boxes, self._last_top_k = self._run_inference(frame)
                if self._last_boxes and self._last_top_k:
                    _draw_face_predictions(
                        frame,
                        self._last_boxes,
                        self._last_top_k,
                    )
                if n_frames > 0 and n_frames % 30 == 0:
                    elapsed = time.perf_counter() - t_start
                    fps = n_frames / elapsed if elapsed > 0 else 0
                    logger.info("FPS: %.1f", fps)
                cv2.imshow("Live Emotion", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._stream.release()
            cv2.destroyAllWindows()
            logger.info("Clean shutdown.")

    def release(self) -> None:
        """Release camera and windows. Called automatically on exit."""
        self._stream.release()
        cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live webcam emotion recognition.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained classifier checkpoint (.pt).",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Webcam device index.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Model input size (height and width).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Number of top emotions to show per face.",
    )
    parser.add_argument(
        "--infer_every_n",
        type=int,
        default=1,
        help="Run model every N frames (1 = every frame).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: next to checkpoint).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    app = LiveEmotionApp(
        checkpoint_path=args.checkpoint,
        camera_index=args.camera_index,
        img_size=args.img_size,
        top_k=args.top_k,
        infer_every_n_frames=args.infer_every_n,
        config_path=args.config,
    )
    app.run()


if __name__ == "__main__":
    main()
