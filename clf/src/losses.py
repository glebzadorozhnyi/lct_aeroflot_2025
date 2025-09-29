"""Classification loss functions.

This module provides various loss functions for classification tasks:
- FocalLoss: Focal Loss for addressing class imbalance by down-weighting
  easy examples and focusing on hard examples.

Focal Loss is particularly useful for datasets with severe class imbalance,
as it reduces the contribution of well-classified examples and focuses
training on hard, misclassified examples.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.

    Focal Loss modifies the standard cross-entropy loss by adding a modulating
    factor (1 - p_t)^gamma that reduces the loss contribution from easy examples
    and focuses learning on hard examples. This is particularly effective for
    datasets with severe class imbalance.

    The loss is defined as:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
    - p_t is the model's estimated probability for the true class
    - α_t is a weighting factor for class t
    - γ (gamma) is the focusing parameter that adjusts the rate at which
      easy examples are down-weighted

    Reference:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal loss for dense object detection. ICCV, 2017.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        use_softmax: bool = True,
        ignore_index: int | None = None,
        label_smoothing: float = 0.0,
        adaptive_smoothing: bool = False,
    ):
        """
        Initialize Focal Loss with label smoothing.

        Args:
            alpha (float, optional): Weighting factor for rare class. Defaults to 1.0.
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            use_softmax (bool, optional): Whether to apply softmax to inputs. Defaults to True.
            ignore_index (int | None, optional): Class index to ignore. Defaults to None.
            label_smoothing (float, optional): Label smoothing factor [0.0, 1.0).
                0.0 means no smoothing. Defaults to 0.0.
            adaptive_smoothing (bool): If True, apply stronger smoothing to
                confident predictions. Defaults to False.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_softmax = use_softmax
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.adaptive_smoothing = adaptive_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss with label smoothing.

        Args:
            inputs (torch.Tensor): Model predictions (batch_size, num_classes).
            targets (torch.Tensor): Ground truth class indices (batch_size,).

        Returns:
            torch.Tensor: Scalar focal loss value.
        """
        num_classes = inputs.shape[1]

        # Convert inputs to probabilities
        inputs_prob = F.softmax(inputs, dim=1) if self.use_softmax else inputs

        # Create one-hot encoding of targets
        one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()

        # Apply label smoothing
        if self.label_smoothing > 0.0:
            if self.adaptive_smoothing:
                # Adaptive smoothing based on prediction confidence
                max_prob = torch.max(inputs_prob, dim=1)[0]
                adaptive_eps = self.label_smoothing * max_prob.unsqueeze(1)
                smooth_targets = (1.0 - adaptive_eps) * one_hot_targets + adaptive_eps / num_classes
            else:
                # Standard uniform smoothing
                smooth_targets = (1.0 - self.label_smoothing) * one_hot_targets + self.label_smoothing / num_classes
        else:
            smooth_targets = one_hot_targets

        # Compute log probabilities
        inputs_log_prob = F.log_softmax(inputs, dim=1) if self.use_softmax else torch.log(inputs)

        # Compute the modulating factor (1 - p_t)^gamma for true class only
        # For label smoothing, we use the original one-hot targets for modulation
        true_class_prob = torch.sum(inputs_prob * one_hot_targets, dim=1)
        weight = torch.pow(1.0 - true_class_prob, self.gamma)

        # Compute focal loss: -α * (1 - p_t)^γ * log(p_t)
        # Apply to smoothed targets
        focal = -self.alpha * inputs_log_prob
        focal_loss = torch.sum(focal * smooth_targets, dim=1)

        # Apply modulation weight
        focal_loss = weight * focal_loss

        # Optionally ignore specific class index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            focal_loss = focal_loss * mask.float()
            focal_loss = focal_loss.sum() / mask.float().sum().clamp(min=1.0)
        else:
            focal_loss = torch.mean(focal_loss)

        return focal_loss


class AdaCos(nn.Module):
    """
    AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations.

    An alternative to ArcFace that adaptively scales the cosine similarities
    without requiring manual tuning of the scale parameter.

    Reference:
        Zhang, X., Zhao, R., Qiao, Y., Wang, X., & Li, H. (2019).
        AdaCos: Adaptively scaling cosine logits for effectively learning deep face representations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        margin: float = 0.50,
        theta_zero: float = math.pi / 4,
    ):
        """
        Initialize AdaCos layer.

        Args:
            in_features (int): Size of input features.
            out_features (int): Size of output features.
            margin (float): Fixed angular margin. Defaults to 0.50.
            theta_zero (float): Initial value for adaptive scaling. Defaults to π/4.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = math.sqrt(2) * math.log(10)
        self.margin = margin
        self.theta_zero = theta_zero

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of AdaCos."""
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Calculate sine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Calculate cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Create one-hot encoding
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        # Apply margin to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Adaptive scaling
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.scale * cosine), torch.zeros_like(cosine))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)))
            self.scale = torch.log(B_avg) / (
                torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med)) + 1e-7
            )

        output *= self.scale

        return output


class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.

    This module implements the ArcFace loss which adds an angular margin penalty
    to enhance the discriminative power of face recognition models by maximizing
    the decision boundary in angular space.

    Reference:
        Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019).
        ArcFace: Additive angular margin loss for deep face recognition.
        In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 30.0,
        margin: float = 0.50,
        easy_margin: bool = False,
        ls_eps: float = 0.0,
    ):
        """
        Initialize ArcMarginProduct layer.

        Args:
            in_features (int): Size of input features (embedding dimension).
            out_features (int): Size of output features (number of classes).
            scale (float): Feature scale parameter (s). Controls the radius of the
                hypersphere. Typical values: 30-64. Defaults to 30.0.
            margin (float): Angular margin penalty (m) in radians. Controls the
                margin between classes. Typical values: 0.3-0.7. Defaults to 0.50.
            easy_margin (bool): Whether to use easy margin. If True, uses
                cos(theta + m), otherwise uses more complex formulation. Defaults to False.
            ls_eps (float): Label smoothing epsilon. If > 0, applies label smoothing.
                Defaults to 0.0.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps

        # Weight matrix (W): learnable parameters for each class
        # Shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cosine and sine of margin for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # threshold
        self.mm = math.sin(math.pi - margin) * margin  # margin offset

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ArcMarginProduct.

        Args:
            input (torch.Tensor): Input features with shape (batch_size, in_features).
                These should be L2-normalized embeddings.
            label (torch.Tensor): Ground truth labels with shape (batch_size,).

        Returns:
            torch.Tensor: Logits with angular margin applied, shape (batch_size, out_features).
        """
        # Normalize input features and weights
        # input: (batch_size, in_features)
        # weight: (out_features, in_features)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # Clamp cosine to avoid numerical issues
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Calculate sine from cosine: sin^2 + cos^2 = 1
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # Calculate cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # Easy margin: simply use cos(theta + m) when cos(theta) > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Hard margin: use cos(theta + m) when theta + m < pi
            # This prevents the margin from being applied when theta is already large
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding for target labels
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Apply label smoothing if specified
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        # Apply margin only to the target class
        # For target class: use phi (cos(theta + m))
        # For non-target classes: use original cosine (cos(theta))
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the output
        output *= self.scale

        return output
