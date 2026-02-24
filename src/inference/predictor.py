"""Model loading and forward inference for the emotion classifier."""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import yaml

from src.utils.labels import IDX_TO_LABEL, NUM_CLASSES


class EmotionPredictor:
    """Loads YOLO backbone + emotion classifier and runs inference.

    Supports single image or batch; returns logits, softmax probabilities,
    and top-K predictions. No gradients; device-aware (CPU/CUDA).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load backbone, classifier, and checkpoint.

        Args:
            checkpoint_path: Path to saved classifier state dict (.pt).
            config_path: Path to config YAML (e.g. config_snapshot.yaml).
                If None, uses same directory as checkpoint for config_snapshot.yaml.
            device: Device for inference; if None, uses CUDA when available.
        """
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if config_path is None:
            config_path = checkpoint_path.parent / "config_snapshot.yaml"
        else:
            config_path = Path(config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")

        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._checkpoint_path = checkpoint_path

        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_cfg = config.get("model", {})
        yolo_id = model_cfg.get("yolo_model", "yolov8n.pt")
        hidden_dims = model_cfg.get("classifier_hidden_dims", [512, 256])
        dropout = model_cfg.get("dropout", 0.0)
        num_classes = model_cfg.get("num_classes", NUM_CLASSES)

        from src.models.yolo_backbone import YOLOBackbone
        from src.models.emotion_classifier import EmotionClassifier

        self._backbone = YOLOBackbone(model_path=yolo_id, device=self._device)
        self._backbone.eval()

        # Infer feature channels with a dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640, device=self._device)
            feat = self._backbone(dummy)
        feature_channels = int(feat.shape[1])

        self._classifier = EmotionClassifier(
            feature_channels=feature_channels,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self._device)
        state = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        self._classifier.load_state_dict(state, strict=True)
        self._classifier.eval()

        self._num_classes = num_classes

    def _validate_input(self, x: torch.Tensor, batch: bool) -> None:
        """Ensure input tensor has correct shape (B, 3, 640, 640)."""
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (B, C, H, W), got dim={x.dim()}"
            )
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x.shape[1]}")
        if x.shape[2] != 640 or x.shape[3] != 640:
            raise ValueError(
                f"Expected spatial size 640x640, got {x.shape[2]}x{x.shape[3]}"
            )

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        top_k: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[str, float]]]]:
        """Run inference on a batch of images.

        Args:
            images: Tensor of shape (B, 3, 640, 640), same normalization as
                validation (ImageNet mean/std). On correct device.
            top_k: Number of top predictions to return per sample.

        Returns:
            logits: (B, num_classes).
            probs: (B, num_classes) softmax probabilities.
            top_k_list: For each sample, list of (label_str, score) of length
                min(top_k, num_classes).
        """
        self._validate_input(images, batch=True)
        images = images.to(self._device, non_blocking=True)
        features = self._backbone(images)
        logits = self._classifier(features)

        if logits.dim() != 2 or logits.shape[1] != self._num_classes:
            raise ValueError(
                f"Classifier output shape {logits.shape} does not match "
                f"num_classes={self._num_classes}"
            )
        probs = torch.softmax(logits, dim=1)
        top_k_list: List[List[Tuple[str, float]]] = []
        for i in range(logits.shape[0]):
            p = probs[i]
            k = min(top_k, self._num_classes)
            scores, idxs = torch.topk(p, k)
            top_k_list.append([
                (IDX_TO_LABEL[int(idx)], float(score))
                for idx, score in zip(idxs.tolist(), scores.tolist())
            ])
        return logits, probs, top_k_list

    @torch.no_grad()
    def predict_single(
        self,
        image: torch.Tensor,
        top_k: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, float]]]:
        """Run inference on a single image (1, 3, 640, 640).

        Returns:
            logits: (1, num_classes).
            probs: (1, num_classes).
            top_k_list: List of (label_str, score) for this image.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        self._validate_input(image, batch=False)
        logits, probs, top_k_list = self.predict(image, top_k=top_k)
        return logits, probs, top_k_list[0]
