"""Webcam capture loop with OpenCV VideoCapture."""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebcamStream:
    """Wraps cv2.VideoCapture with camera selection, optional resizing, and clean release."""

    def __init__(
        self,
        camera_index: int = 0,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Open the camera and optionally set target frame size.

        Args:
            camera_index: OpenCV camera index (0 = default webcam).
            target_size: If set, (width, height) to resize each frame; None = native size.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self._camera_index = camera_index
        self._target_size = target_size
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Camera unavailable: index={camera_index}")
        logger.info("Camera initialized: index=%s", camera_index)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame.

        Returns:
            (ret, frame): ret is True if a frame was read; frame is BGR (H, W, 3)
                or None on failure. If target_size was set, frame is resized.
        """
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        if self._target_size is not None:
            w, h = self._target_size
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        return True, frame

    def release(self) -> None:
        """Release the camera and free resources. Safe to call multiple times."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released: index=%s", self._camera_index)

    def __enter__(self) -> "WebcamStream":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.release()
