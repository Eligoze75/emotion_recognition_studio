"""Model components: YOLOv8 backbone and emotion classification head."""

from src.models.yolo_backbone import YOLOBackbone
from src.models.emotion_classifier import EmotionClassifier

__all__ = ["YOLOBackbone", "EmotionClassifier"]
