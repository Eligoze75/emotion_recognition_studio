"""Full training pipeline: frozen YOLOv8 backbone + trainable emotion classifier."""

import json
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on path when running as script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader


def _project_root() -> Path:
    """Return project root (directory containing src/)."""
    return _PROJECT_ROOT


def _load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load YAML config; default to configs/default.yaml under project root."""
    if config_path is None:
        config_path = _project_root() / "configs" / "default.yaml"
    from src.utils.config import load_config
    return load_config(config_path)


def _set_seed(seed: Optional[int]) -> None:
    """Set all random seeds from config."""
    from src.utils.seed import set_seed
    set_seed(seed)


def _get_dataloaders(config: dict[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build train and validation DataLoaders from config."""
    from src.data import (
        build_train_val_samples,
        EmotionDataset,
        get_train_transforms,
        get_val_transforms,
    )

    root = _project_root()
    train_samples, val_samples = build_train_val_samples(
        config, root, seed=config.get("seed")
    )
    dl_cfg = config.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size", 16)
    num_workers = dl_cfg.get("num_workers", 0)
    pin_memory = dl_cfg.get("pin_memory", False)

    train_ds = EmotionDataset(
        train_samples,
        transform=get_train_transforms(config),
    )
    val_ds = EmotionDataset(
        val_samples,
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def _infer_backbone_feature_channels(backbone: Any, device: torch.device) -> int:
    """Run one dummy forward to get feature map channel size."""
    dummy = torch.zeros(1, 3, 640, 640, device=device)
    with torch.no_grad():
        out = backbone(dummy)
    # out: [B, C, H, W]
    return int(out.shape[1])


def main(config_path: Optional[Path] = None) -> None:
    """Run the full training pipeline.

    Loads config, sets seeds, builds backbone + classifier and dataloaders,
    trains only the classifier with cross-entropy, logs epoch train/val loss,
    and saves classifier weights, label mapping, and config snapshot.
    """
    config = _load_config(config_path)
    seed = config.get("seed")
    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config.get("model", {})
    yolo_id = model_cfg.get("yolo_model", "yolov8n.pt")
    hidden_dims = model_cfg.get("classifier_hidden_dims", [512, 256])
    dropout = model_cfg.get("dropout", 0.3)
    num_classes = model_cfg.get("num_classes", 7)

    from src.models.yolo_backbone import YOLOBackbone
    from src.models.emotion_classifier import EmotionClassifier

    backbone = YOLOBackbone(model_id=yolo_id, device=device)
    backbone.eval()
    feature_channels = _infer_backbone_feature_channels(backbone, device)
    classifier = EmotionClassifier(
        feature_channels=feature_channels,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    train_loader, val_loader = _get_dataloaders(config)
    if len(train_loader.dataset) == 0:
        print("No training samples found. Check data/raw/archive/images.", file=sys.stderr)
        sys.exit(1)

    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 50)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    output_dir = _project_root() / config.get("output_dir", "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_classifier = config.get("save_classifier_name", "emotion_classifier.pt")
    save_mapping = config.get("save_label_mapping_name", "label_mapping.json")
    save_config_snap = config.get("save_config_snapshot_name", "config_snapshot.yaml")

    from src.data.dataset import get_emotion_label_mapping
    label_mapping = get_emotion_label_mapping()
    with open(output_dir / save_mapping, "w") as f:
        json.dump(label_mapping, f, indent=2)

    with open(output_dir / save_config_snap, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    for epoch in range(epochs):
        classifier.train()
        train_loss = 0.0
        n_train = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.no_grad():
                features = backbone(images)
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            n_train += images.size(0)

        train_loss /= max(n_train, 1)
        classifier.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                features = backbone(images)
                logits = classifier(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                n_val += images.size(0)
        val_loss /= max(n_val, 1)
        print(
            f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

    torch.save(classifier.state_dict(), output_dir / save_classifier)
    print(f"Saved classifier to {output_dir / save_classifier}")
    print(f"Saved label mapping to {output_dir / save_mapping}")
    print(f"Saved config snapshot to {output_dir / save_config_snap}")


if __name__ == "__main__":
    path = None
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    main(path)
