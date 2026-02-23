"""Lightweight emotion classification head on top of YOLOv8 features."""

from typing import List, Sequence, Union

import torch
from torch import nn


class EmotionClassifier(nn.Module):
    """Classifier: global pooling + FC layers -> 7-class logits. No softmax in forward."""

    def __init__(
        self,
        feature_channels: int,
        hidden_dims: Union[Sequence[int], List[int]] = (512, 256),
        num_classes: int = 7,
        dropout: float = 0.0, # Explicitly OFF by default
    ) -> None:
        """Build classifier head.

        Args:
            feature_channels: Number of channels from backbone/neck feature map.
            hidden_dims: Hidden layer sizes; last layer projects to num_classes.
            num_classes: Number of output classes (7 emotions).
            dropout: Dropout probability after pooling and between FC layers.
        """
        super().__init__()
        self.feature_channels = feature_channels
        self.num_classes = num_classes

        layers: List[nn.Module] = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        dims = [feature_channels] + list(hidden_dims) + [num_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
        
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute class logits from feature maps.

        Args:
            features: [B, C, H, W] from YOLO backbone.

        Returns:
            Logits [B, num_classes]; no softmax applied.
        """
        return self.head(features)
