"""Encoder architectures for feature extraction.

This module provides various encoder/backbone architectures for computer vision tasks:
- BaseEncoder: Abstract base class with registry mechanism
- ConvNextEncoder: ConvNeXt architecture implementation
- DLAEncoder: Deep Layer Aggregation encoder
- GhostNetEncoder: Efficient GhostNet architecture
- MiTEncoder: Mix Transformer encoder
- MobileNetEncoder: MobileNet family encoders
- PreProcessing: Input preprocessing utilities

All encoders follow a common interface and support multi-scale feature extraction,
quantization-aware training, and efficient inference.
"""

from src.encoders.base import BaseEncoder
from src.encoders.convnext import ConvNextEncoder
from src.encoders.dla import DLAEncoder
from src.encoders.ghostnet import GhostNetEncoder
from src.encoders.mit import MiTEncoder
from src.encoders.mobilenet import MobileNetEncoder
from src.encoders.preprocessing import PreProcessing

__all__ = [
    "BaseEncoder",
    "ConvNextEncoder",
    "DLAEncoder",
    "GhostNetEncoder",
    "MiTEncoder",
    "MobileNetEncoder",
    "PreProcessing",
]
