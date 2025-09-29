"""Classification model implementation.

This module provides a complete classification model that combines:
- Preprocessing: Input normalization and preparation
- Encoder: Feature extraction backbone (e.g., ConvNext, MobileNet, etc.)
- Head: Classification head for converting features to class logits
- Loss computation: Support for both CrossEntropy and Focal Loss

The model supports various training configurations including class weighting,
label smoothing, and quantization for efficient deployment.
"""

from typing import Callable

import torch
from torch import nn

from src.encoders import BaseEncoder, PreProcessing
from src.heads import BaseHead
from src.losses import ArcMarginProduct, FocalLoss


class Clf(nn.Module):
    """
    Complete classification model with preprocessing, encoder, head, and quantization support.

    This model provides a full pipeline for image classification:
    1. Preprocessing: Normalizes input images
    2. Encoder: Extracts multi-scale features from images
    3. Head: Converts features to class predictions
    4. Loss computation: Supports CrossEntropy and Focal Loss
    5. Quantization: Full QAT support for efficient deployment

    The model can be configured with different encoders, heads, loss functions,
    and supports advanced features like class weighting and label smoothing.
    """

    def __init__(
        self,
        encoder: BaseEncoder | Callable[..., BaseEncoder],
        head: BaseHead | Callable[..., BaseHead],
        focal: bool = False,
        label_smoothing: float = 0.0,
        adaptive_smoothing: bool = False,
        **kwargs: any,
    ):
        """
        Initialize the classification model.

        Args:
            encoder (BaseEncoder | Callable): Encoder instance or callable that returns
                an encoder. Handles feature extraction from input images.
            head (BaseHead | Callable): Head instance or callable that returns a head.
                Converts encoder features to class logits.
            focal (bool, optional): Whether to use Focal Loss instead of CrossEntropy.
                Focal Loss is better for imbalanced datasets. Defaults to False.
            label_smoothing (float, optional): Label smoothing factor for CrossEntropy loss.
                Helps prevent overfitting. Range: [0.0, 1.0]. Defaults to 0.0.
            adaptive_smoothing (bool): If True, apply stronger smoothing to
                confident predictions. Defaults to False.
            **kwargs (any): Additional keyword arguments (unused).
        """
        super().__init__()

        # Initialize model components
        self.preprocessing = PreProcessing()
        self.encoder = encoder if isinstance(encoder, BaseEncoder) else encoder()
        self.head = head if isinstance(head, BaseHead) else head(in_channels=self.encoder.out_channelss)
        self.adacos = ArcMarginProduct(
            self.encoder.out_channels,
            self.head.num_classes,
            scale=30,
            margin=0.5,
            easy_margin=False,
            ls_eps=0.1,
        )

        # Configure loss function
        if focal:
            self.clsloss_crit = FocalLoss(
                alpha=1.0,
                gamma=2.0,
                use_softmax=True,
                label_smoothing=label_smoothing,
                adaptive_smoothing=adaptive_smoothing,
            )
        else:
            self.clsloss_crit = nn.CrossEntropyLoss(
                reduction="mean",
                label_smoothing=label_smoothing,
            )

        # Loss multiplier (can be adjusted for multi-task learning)
        self.clsloss_mult = 0.2
        self.cosloss_mult = 0.8

    def count_losses(
        self, logits: torch.Tensor, cos_logits: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Compute classification losses.

        Args:
            logits (torch.Tensor): Model predictions with shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth class indices with shape (batch_size,).

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - "loss": Total weighted loss for backpropagation
                - "clsloss": Raw classification loss (CrossEntropy or Focal)
        """
        clsloss = self.clsloss_crit(logits, targets)
        cosloss = self.clsloss_crit(cos_logits, targets)
        loss = self.clsloss_mult * clsloss + self.cosloss_mult * cosloss
        return {
            "loss": loss,
            "clsloss": clsloss,
            "cosloss": cosloss,
        }

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass of the classification model.

        Args:
            x (torch.Tensor): Input images with shape (batch_size, channels, height, width).
            targets (torch.Tensor | None, optional): Ground truth class indices for training.
                If provided, returns loss dictionary. If None, returns only logits. Defaults to None.

        Returns:
            torch.Tensor | dict[str, torch.Tensor]:
                - If targets is None: Raw logits with shape (batch_size, num_classes)
                - If targets provided: Dictionary with "logit", "loss", and "clsloss"
        """
        # Preprocessing: normalize and prepare inputs
        preprocessed = self.preprocessing(x)
        # Feature extraction through encoder
        encoded = self.encoder(preprocessed)
        # Classification head: convert features to logits
        logits, features = self.head(encoded)
        # Training mode: compute losses
        if targets is not None:
            # AdaCos
            cos_logits = self.adacos(features, targets)
            res = self.count_losses(logits, cos_logits, targets)
            res["logit"] = logits
            res["cos_logit"] = cos_logits
            res["feature"] = features
            return res
        # Inference mode: return only logits
        return logits, features
