"""YOLOv8 backbone + neck as frozen feature extractor.

Loads a pretrained YOLOv8 via Ultralytics API, uses backbone + neck only,
freezes all parameters, and exposes forward(images) returning feature tensors.
No detection head, no NMS. Uses a forward hook to capture neck output.
"""

from typing import Any, Optional

import torch
from torch import nn

# TODO: YOLO internals — layer index for "last neck layer" is model-dependent
# (e.g. 20-22 for yolov8n). Verify by inspecting model.model when upgrading.
_BACKBONE_NECK_END_INDEX = 22


class YOLOBackbone(nn.Module):
    """Frozen YOLOv8 backbone + neck for feature extraction."""

    def __init__(
        self,
        model_id: str = "yolov8n.pt",
        device: Optional[torch.device] = None,
    ) -> None:
        """Load pretrained YOLOv8 and keep backbone + neck only, frozen.

        Args:
            model_id: Ultralytics model id (e.g. 'yolov8n.pt').
            device: Device to place the model on; if None, uses default.
        """
        super().__init__()
        from ultralytics import YOLO

        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._captured: Optional[torch.Tensor] = None

        # TODO: YOLO internals — YOLO() returns a wrapper; .model is the
        # DetectionModel. The actual layer sequence may be in .model.model.
        self._yolo = YOLO(model_id)
        self._detection_model = self._yolo.model
        self._detection_model.to(self._device)

        for p in self._detection_model.parameters():
            p.requires_grad = False

        # Register hook on the layer that produces the last neck features.
        self._register_feature_hook()

    def _register_feature_hook(self) -> None:
        """Register a forward hook to capture the last neck layer output."""
        # TODO: YOLO internals — access the inner Sequential and hook the
        # layer at _BACKBONE_NECK_END_INDEX - 1 (last backbone+neck layer).
        target = getattr(self._detection_model, "model", self._detection_model)
        if hasattr(target, "__getitem__") and hasattr(target, "__len__"):
            idx = min(_BACKBONE_NECK_END_INDEX - 1, len(target) - 1)
            layer = target[idx]
        else:
            layer = self._detection_model

        def _hook(_module: Any, _input: Any, output: Any) -> None:
            out = output
            if isinstance(out, (list, tuple)):
                out = out[0]
            self._captured = out

        self._hook_handle = layer.register_forward_hook(_hook)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract feature maps from backbone + neck.

        Args:
            images: Batched images [B, C, H, W], same format as YOLO expects
                (e.g. RGB, normalized). Input size should match YOLO training
                (e.g. 640x640) for consistent feature resolution.

        Returns:
            Feature tensor from the last neck layer, shape [B, C, H', W'].
            Exact dimensions depend on YOLO internals.
        """
        self._captured = None
        # Run full model forward; hook captures neck output. We ignore head output.
        _ = self._detection_model(images)
        if self._captured is None:
            raise RuntimeError(
                "YOLO backbone did not capture features; check hook and layer index."
            )
        return self._captured

    def __del__(self) -> None:
        if getattr(self, "_hook_handle", None) is not None:
            self._hook_handle.remove()
