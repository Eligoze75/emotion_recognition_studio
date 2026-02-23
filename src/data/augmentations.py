"""On-the-fly augmentations for training; validation uses no augmentation."""

from typing import Any, Optional

import torch
from torchvision import transforms


def get_train_transforms(
    config: Optional[dict[str, Any]] = None,
    enabled: bool = True,
) -> transforms.Compose:
    """Build training transform pipeline.

    When enabled, applies: random grayscale, salt-and-pepper, rotation,
    random crop/scale, horizontal flip, brightness/contrast jitter.
    When disabled or config has augmentation.enabled=false, returns only
    Resize + ToTensor + Normalize (minimal pipeline for training).

    Args:
        config: Full config dict; if None, augmentation is disabled.
        enabled: Override; if False, no augmentation is applied.

    Returns:
        torchvision Compose transform for training.
    """
    if config is None:
        aug_config = {}
        enabled = False
    else:
        aug_config = config.get("augmentation", {}) or {}
        enabled = enabled and aug_config.get("enabled", True)

    # Common: resize to fixed size for YOLO/backbone, then to tensor.
    # YOLOv8 typically uses 640; we use a reasonable input size.
    size = 640
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if not enabled:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalizer,
        ])

    prob_grayscale = aug_config.get("random_grayscale_prob", 0.2)
    prob_salt_pepper = aug_config.get("salt_pepper_prob", 0.05)
    amount_sp = aug_config.get("salt_pepper_amount", 0.02)
    rot_deg = aug_config.get("rotation_degrees", 15)
    crop_scale = aug_config.get("random_crop_scale", [0.85, 1.0])
    prob_hflip = aug_config.get("horizontal_flip_prob", 0.5)
    brightness = aug_config.get("brightness_jitter", 0.3)
    contrast = aug_config.get("contrast_jitter", 0.3)

    return transforms.Compose([
        transforms.Resize((int(size * 1.1), int(size * 1.1))),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(p=prob_hflip),
        transforms.RandomRotation(rot_deg),
        transforms.RandomGrayscale(p=prob_grayscale),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        transforms.ToTensor(),
        _SaltPepper(prob=prob_salt_pepper, amount=amount_sp),
        normalizer,
    ])


def get_val_transforms() -> transforms.Compose:
    """Transform for validation: no augmentation, deterministic resize + normalize."""
    size = 640
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class _SaltPepper(torch.nn.Module):
    """Salt-and-pepper noise: random pixels set to 0 or 1 (on [0,1] tensor)."""

    def __init__(self, prob: float = 0.05, amount: float = 0.02) -> None:
        super().__init__()
        self.prob = prob
        self.amount = amount

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prob <= 0 or self.amount <= 0 or not self.training:
            return x
        if torch.rand(1).item() > self.prob:
            return x
        c, h, w = x.shape
        n = int(h * w * self.amount)
        flat = x.view(c, -1)
        idx = torch.randperm(flat.numel(), device=x.device)[:n]
        salt = torch.randint(0, 2, (n,), device=x.device, dtype=x.dtype)
        flat.view(-1)[idx] = salt
        return x
