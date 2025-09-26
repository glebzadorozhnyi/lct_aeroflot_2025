"""Classification head implementation.

This module provides a classification head that converts multi-scale feature maps
from an encoder/backbone into class prediction logits. The head uses global average
pooling and a linear classifier with optional dropout for regularization.
"""

import torch
from torch import nn

from src.heads.base import BaseHead


class ClfHead(BaseHead, registry_name="clf"):
    """
    Classification head for image classification tasks.

    This head takes multi-scale feature maps from an encoder and produces
    class prediction logits. It uses the highest-level feature map (typically P7)
    and applies global average pooling followed by a linear classifier.

    The head is automatically registered in the BaseHead registry with the name "clf".

    Architecture:
        1. Global Average Pooling (AdaptiveAvgPool2d) - reduces spatial dimensions to 1x1
        2. Flatten - converts to 1D feature vector
        3. Dropout (optional) - for regularization
        4. Linear layer - maps features to class logits
    """

    def __init__(self, in_channels: list[int], num_classes: int, dropout: float = 0.0):
        """
        Initialize the classification head.

        Args:
            in_channels (list[int]): List of input channel dimensions for each feature level.
                The last element (in_channels[-1]) is used as the input dimension for
                the linear classifier, corresponding to the highest-level features.
            num_classes (int): Number of output classes for classification.
            dropout (float, optional): Dropout probability for regularization.
                Set to 0.0 to disable dropout. Defaults to 0.0.
        """
        super().__init__()
        # Neck: converts 2D feature maps to 1D feature vectors
        self.neck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(1),  # Flatten: (B, C, 1, 1) -> (B, C)
        )
        # Head: classification layer with optional dropout
        self.num_classes = num_classes
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_channels[-1], num_classes))
        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize the weights of the linear classifier.

        Uses Xavier uniform initialization for the weight matrix and
        constant initialization (0) for the bias term. This helps with
        training stability and convergence.
        """
        nn.init.xavier_uniform_(self.head[1].weight)
        nn.init.constant_(self.head[1].bias, 0)

    def forward(
        self,
        inputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """
        Forward pass of the classification head.

        Takes multi-scale feature maps and produces class prediction logits.
        Only the highest-level feature map (last element) is used for classification.

        Args:
            inputs (tuple[torch.Tensor, ...]): Tuple of 6 feature tensors from different
                scales/levels of the encoder, typically:
                - inputs[0]: P2 features (lowest level, highest resolution)
                - inputs[1]: P3 features
                - inputs[2]: P4 features
                - inputs[3]: P5 features
                - inputs[4]: P6 features
                - inputs[5]: P7 features (highest level, lowest resolution)

                Each tensor has shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Classification logits with shape (batch_size, num_classes).
                These are raw scores (not probabilities) that can be passed to
                a loss function like CrossEntropyLoss or converted to probabilities
                using softmax.
        """
        # Use the highest-level features (P7) for classification
        c = inputs[-1]  # Shape: (B, C, H, W)

        # Apply global average pooling and flatten
        features = self.neck(c)  # Shape: (B, C)

        # Apply dropout and linear classification layer
        logits = self.head(features)  # Shape: (B, num_classes)

        return logits, features
