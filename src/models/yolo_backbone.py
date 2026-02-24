"""YOLOv8 backbone + neck as frozen feature extractor (manual PyTorch load).

This version **never downloads COCO or triggers Ultralytics training pipelines**.
It loads a YOLOv8 .pt file manually and exposes backbone+neck features for downstream heads.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Project root and default weights dir
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_WEIGHTS_DIR = _PROJECT_ROOT / "weights"

# Last neck layer index: adjust if changing YOLO version
_BACKBONE_NECK_END_INDEX = 22


class YOLOBackbone(nn.Module):
    """Frozen YOLOv8 backbone + neck for feature extraction only."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: Optional[torch.device] = None,
    ) -> None:
        """Load YOLOv8 backbone+neck as a frozen feature extractor.

        Args:
            model_path: Path to a YOLOv8 .pt file (weights_dir/model_path is used if exists).
            device: Device to place the model on; defaults to CUDA if available.
        """
        super().__init__()
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._captured: Optional[torch.Tensor] = None

        # Resolve local weights
        local_path = Path(_DEFAULT_WEIGHTS_DIR) / Path(model_path).name
        if not local_path.exists():
            raise FileNotFoundError(f"YOLO weights not found at {local_path}. Please download manually.")
        state_dict = torch.load(local_path, map_location=self._device)

        # Build model from checkpoint. Ultralytics YOLOv8 .pt files store a full nn.Module
        # under the "model" key; we reuse it directly instead of reconstructing.
        model_obj = state_dict.get("model", None)
        if not isinstance(model_obj, nn.Module):
            raise ValueError(
                "Unexpected YOLO .pt format: 'model' key not found or is not an nn.Module"
            )
        self._model = model_obj.to(self._device)

        # Ensure float32 so input (float32) and weights match; .pt may be saved as half.
        self._model = self._model.float()

        # Freeze parameters
        for p in self._model.parameters():
            p.requires_grad = False

        # Register hook for last neck layer
        self._register_feature_hook()

    def _register_feature_hook(self) -> None:
        """Register a forward hook to capture the last neck layer output."""
        # Ultralytics DetectionModel exposes the backbone/neck layers under `.model`
        # (typically a ModuleList). Fall back to the whole model if not present.
        backbone_container = getattr(self._model, "model", self._model)
        if hasattr(backbone_container, "__getitem__") and hasattr(backbone_container, "__len__"):
            idx = min(_BACKBONE_NECK_END_INDEX - 1, len(backbone_container) - 1)
            layer = backbone_container[idx]
        else:
            layer = backbone_container

        def _hook(_module: Any, _input: Any, output: Any) -> None:
            out = output
            if isinstance(out, (list, tuple)):
                out = out[0]
            self._captured = out

        self._hook_handle = layer.register_forward_hook(_hook)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract feature maps from backbone+neck. Shape [B, C, H', W'].

        Args:
            images: Batched images [B, C, H, W], RGB, normalized (e.g., ImageNet).

        Returns:
            Feature tensor from the last neck layer.
        """
        self._captured = None
        _ = self._model(images)
        if self._captured is None:
            raise RuntimeError("YOLO backbone did not capture features; check hook and layer index.")
        return self._captured

    def __del__(self) -> None:
        if getattr(self, "_hook_handle", None) is not None:
            self._hook_handle.remove()
