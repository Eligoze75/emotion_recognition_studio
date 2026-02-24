"""Webcam capture loop with OpenCV VideoCapture.

A background thread continuously grabs frames from the camera so the main
inference/display loop always gets the *freshest* frame without ever blocking
on camera I/O.  This keeps the live feed smooth regardless of how long inference
takes.
"""

import logging
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Prefer AVFoundation on macOS for more stable camera behavior.
_CAP_BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY


class WebcamStream:
    """Threaded webcam reader — background thread captures continuously,
    read() always returns the most recent frame instantly."""

    def __init__(
        self,
        camera_index: int = 0,
        target_size: Optional[Tuple[int, int]] = None,
        warmup_frames: int = 20,
    ) -> None:
        """Open the camera, warm up, and start the background capture thread.

        Args:
            camera_index: OpenCV camera index (0 = default webcam).
            target_size: If set, (width, height) to resize each frame.
            warmup_frames: Frames to discard after open for pipeline stabilization.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self._camera_index = camera_index
        self._target_size = target_size
        self._warmup_frames = max(0, warmup_frames)

        self._cap = self._open_cap()

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Camera started (threaded): index=%s (warmup=%d frames)",
            camera_index,
            self._warmup_frames,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_cap(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._camera_index, _CAP_BACKEND)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Camera unavailable: index={self._camera_index}")
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # Discard warm-up frames so the pipeline stabilizes before we start storing.
        for _ in range(self._warmup_frames):
            cap.read()
            time.sleep(0.02)
        return cap

    def _capture_loop(self) -> None:
        """Background loop: grab frames as fast as the camera produces them."""
        while not self._stopped:
            if self._cap is None:
                time.sleep(0.05)
                continue
            ret, frame = self._cap.read()
            if ret and frame is not None:
                if self._target_size is not None:
                    w, h = self._target_size
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.01)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recently captured frame (non-blocking).

        Returns:
            (True, frame_bgr) if a frame is available, (False, None) otherwise.
        """
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return False, None
        return True, frame.copy()

    def reconnect(self) -> bool:
        """Release and re-open the camera pipeline (e.g. after a stall)."""
        logger.info("Reconnecting camera: index=%s", self._camera_index)
        with self._frame_lock:
            self._latest_frame = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        try:
            self._cap = self._open_cap()
            logger.info("Camera reconnected: index=%s", self._camera_index)
            return True
        except RuntimeError:
            logger.warning("Camera reconnect failed: index=%s", self._camera_index)
            return False

    def release(self) -> None:
        """Stop the background thread and release the camera."""
        self._stopped = True
        self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera released: index=%s", self._camera_index)

    def __enter__(self) -> "WebcamStream":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.release()
