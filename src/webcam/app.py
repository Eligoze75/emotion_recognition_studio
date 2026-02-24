"""Live emotion recognition from webcam: face detection + classifier + visualization."""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Reduce segfault risk on macOS (OpenMP/NumPy/PyTorch/OpenCV) before any heavy libs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from PIL import Image

# OpenCV after PyTorch to avoid library conflicts on macOS.
import cv2

# Project root on path when run as module.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.augmentations import get_val_transforms
from src.inference.predictor import EmotionPredictor
from src.webcam.stream import WebcamStream

logger = logging.getLogger(__name__)

# Default Haar cascade for face detection (bundled with OpenCV).
_DEFAULT_FACE_CASCADE_PATH = (
    Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
)

# Per-emotion color palette (BGR) — vivid, distinguishable.
_EMOTION_COLORS: dict[str, Tuple[int, int, int]] = {
    "anger":     (  0,  30, 230),   # red
    "contempt":  (180,  20, 200),   # violet
    "disgust":   ( 10, 180,  50),   # green
    "fear":      (  0, 140, 220),   # amber
    "happy":     (  0, 220, 255),   # yellow
    "sad":       (210,  80,  20),   # blue
    "surprised": (  0, 155, 255),   # orange
}
_COLOR_DEFAULT = (200, 200, 200)


def _emotion_color(label: str) -> Tuple[int, int, int]:
    return _EMOTION_COLORS.get(label.lower(), _COLOR_DEFAULT)


# ---------------------------------------------------------------------------
# Face detection helpers
# ---------------------------------------------------------------------------

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
        crops.append(transform(pil).unsqueeze(0))
    return torch.cat(crops, dim=0).to(device)


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def _draw_face_predictions(
    frame: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    top_k_per_face: List[List[Tuple[str, float]]],
) -> None:
    """Draw bounding boxes + styled emotion labels on frame (in-place)."""
    h_frame, w_frame = frame.shape[:2]

    for (x, y, w, h), top_k in zip(boxes, top_k_per_face):
        if not top_k:
            continue

        top_label, top_score = top_k[0]
        box_color = _emotion_color(top_label)

        # --- Bounding box (rounded-corner appearance via slightly thick line) ---
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2, cv2.LINE_AA)

        # --- Label panel dimensions ---
        font      = cv2.FONT_HERSHEY_DUPLEX
        n_labels  = min(2, len(top_k))
        row_h     = 24                         # px per emotion row
        bar_h     = 5                          # px for confidence bar
        pad       = 6                          # inner horizontal padding
        panel_h   = n_labels * (row_h + bar_h) + pad * 2
        panel_w   = w

        # Place panel above the box when there is room, otherwise below.
        if y >= panel_h + 4:
            panel_y = y - panel_h - 4
        else:
            panel_y = y + h + 4

        # Clamp to frame bounds
        panel_y = max(0, min(panel_y, h_frame - panel_h))
        panel_x = max(0, min(x, w_frame - panel_w))

        # --- Semi-transparent dark background ---
        roi = frame[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w]
        if roi.size > 0:
            dark = np.zeros_like(roi)
            dark[:] = (15, 15, 15)
            cv2.addWeighted(dark, 0.65, roi, 0.35, 0, roi)
            frame[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w] = roi

        # --- Text + confidence bar per emotion row ---
        for i, (label, score) in enumerate(top_k[:n_labels]):
            color = _emotion_color(label)
            row_y = panel_y + pad + i * (row_h + bar_h)

            # Emotion name + percentage
            font_scale = 0.52
            text = f"{label.capitalize()}  {score:.0%}"
            cv2.putText(
                frame, text,
                (panel_x + pad, row_y + row_h - 6),
                font, font_scale, color, 1, cv2.LINE_AA,
            )

            # Confidence bar
            bar_y = row_y + row_h
            bar_max_w = max(1, panel_w - pad * 2)
            bar_fill  = int(bar_max_w * score)
            cv2.rectangle(
                frame,
                (panel_x + pad, bar_y),
                (panel_x + pad + bar_max_w, bar_y + bar_h),
                (60, 60, 60), -1,
            )
            if bar_fill > 0:
                cv2.rectangle(
                    frame,
                    (panel_x + pad, bar_y),
                    (panel_x + pad + bar_fill, bar_y + bar_h),
                    color, -1,
                )


# ---------------------------------------------------------------------------
# Inference background worker
# ---------------------------------------------------------------------------

class _InferenceWorker:
    """Runs face-detection + emotion model in a background thread.

    The main display loop submits the latest frame via ``submit()``.
    The worker always processes the most-recently submitted frame so it
    never queues up stale work when inference is slower than capture.
    ``get_result()`` is non-blocking and returns the last known result.
    """

    def __init__(
        self,
        predictor: EmotionPredictor,
        transform: torch.nn.Module,
        top_k: int,
    ) -> None:
        self._predictor  = predictor
        self._transform  = transform
        self._top_k      = top_k

        self._pending_frame: Optional[np.ndarray] = None
        self._pending_lock  = threading.Lock()
        self._new_frame     = threading.Event()

        self._result_boxes: List[Tuple[int, int, int, int]] = []
        self._result_top_k: List[List[Tuple[str, float]]]   = []
        self._result_lock   = threading.Lock()

        self._stopped = False
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Hand the latest frame to the worker (replaces any unprocessed frame)."""
        with self._pending_lock:
            self._pending_frame = frame.copy()
        self._new_frame.set()

    def get_result(
        self,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[List[Tuple[str, float]]]]:
        """Return the most recent (boxes, top_k_per_face) — non-blocking."""
        with self._result_lock:
            return list(self._result_boxes), list(self._result_top_k)

    def stop(self) -> None:
        self._stopped = True
        self._new_frame.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stopped:
            triggered = self._new_frame.wait(timeout=0.1)
            if not triggered or self._stopped:
                continue
            self._new_frame.clear()

            with self._pending_lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is None:
                continue

            try:
                boxes = _detect_faces(frame)
                if not boxes:
                    with self._result_lock:
                        self._result_boxes = []
                        self._result_top_k = []
                    continue
                batch = _preprocess_face_crops(
                    frame, boxes, self._transform, self._predictor._device
                )
                if batch is None:
                    with self._result_lock:
                        self._result_boxes = boxes
                        self._result_top_k = [[] for _ in boxes]
                    continue
                with torch.no_grad():
                    _, _, top_k_list = self._predictor.predict(batch, top_k=self._top_k)
                with self._result_lock:
                    self._result_boxes = boxes
                    self._result_top_k = top_k_list
            except Exception:
                logger.exception("Inference worker error; skipping frame.")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

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
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self._top_k        = top_k
        self._infer_every_n = max(1, infer_every_n_frames)
        self._frame_count  = 0

        self._predictor = EmotionPredictor(
            checkpoint_path, config_path=config_path, device=None
        )
        logger.info("Model loaded: %s", checkpoint_path)

        self._transform = get_val_transforms()

        display_w = img_size if img_size >= 320 else 640
        display_h = int(display_w * 0.75)
        self._stream = WebcamStream(
            camera_index=camera_index,
            target_size=(display_w, display_h),
            warmup_frames=25,
        )

        # Inference runs in its own thread so the display loop is never blocked.
        self._worker = _InferenceWorker(self._predictor, self._transform, top_k)

    def run(self) -> None:
        """Display loop: always renders at camera speed; inference is async."""
        logger.info("Starting live loop. Press 'q' to quit.")
        t_start          = time.perf_counter()
        n_frames         = 0
        consecutive_fail = 0

        try:
            while True:
                ret, frame = self._stream.read()

                # --- No frame: show placeholder, try reconnect ---
                if not ret or frame is None:
                    consecutive_fail += 1
                    if consecutive_fail == 1 or consecutive_fail % 30 == 0:
                        logger.warning("Frame read failed (consecutive=%d)", consecutive_fail)
                    if consecutive_fail >= 10 and (consecutive_fail - 10) % 30 == 0:
                        logger.info("Attempting camera reconnect…")
                        if self._stream.reconnect():
                            consecutive_fail = 0

                    w, h   = self._stream._target_size or (640, 480)
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    status = "Reconnecting…" if consecutive_fail >= 10 else "Waiting for camera…"
                    cv2.putText(
                        canvas, status, (w // 6, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 1, cv2.LINE_AA,
                    )
                    cv2.imshow("Live Emotion", canvas)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        break
                    continue

                consecutive_fail = 0
                n_frames        += 1
                self._frame_count += 1

                # Submit to inference worker every N frames (non-blocking).
                if self._frame_count % self._infer_every_n == 0:
                    self._worker.submit(frame)

                # Always overlay the last known result — non-blocking.
                boxes, top_k_per_face = self._worker.get_result()
                if boxes and top_k_per_face:
                    _draw_face_predictions(frame, boxes, top_k_per_face)

                if n_frames % 60 == 0:
                    elapsed = time.perf_counter() - t_start
                    logger.info("FPS: %.1f", n_frames / elapsed if elapsed else 0)

                cv2.imshow("Live Emotion", frame)
                # waitKey(1) — don't cap the display loop; camera fps is the ceiling.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._worker.stop()
            self._stream.release()
            cv2.destroyAllWindows()
            logger.info("Clean shutdown.")

    def release(self) -> None:
        self._worker.stop()
        self._stream.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam emotion recognition.")
    parser.add_argument("--checkpoint",     type=Path, required=True)
    parser.add_argument("--camera_index",   type=int,  default=0)
    parser.add_argument("--img_size",       type=int,  default=640)
    parser.add_argument("--top_k",          type=int,  default=2)
    parser.add_argument("--infer_every_n",  type=int,  default=1)
    parser.add_argument("--config",         type=Path, default=None)
    parser.add_argument(
        "--log_level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
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
