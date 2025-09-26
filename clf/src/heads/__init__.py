"""Neural network heads for task-specific outputs.

This module provides head architectures that convert encoder features into
task-specific predictions:
- BaseHead: Abstract base class with registry mechanism for all heads
- ClfHead: Classification head with global average pooling and linear classifier

Heads are the final components of neural networks that process multi-scale
features from encoders to produce outputs like class logits, bounding boxes,
or segmentation masks.
"""

from src.heads.base import BaseHead
from src.heads.clf import ClfHead

__all__ = ["BaseHead", "ClfHead"]
