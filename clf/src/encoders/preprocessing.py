"""PreProcessing"""

import torch
from torch import nn

__all__ = ["PreProcessing"]


class PreProcessing(nn.Module):
    """PreProcessing"""

    def __init__(self):
        super().__init__()
        self.register_buffer("norm_k", torch.tensor(255, dtype=torch.float32))
        self.norm = nn.BatchNorm2d(3)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Call

        Args:
            image (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        image = image.float()  # uint8 -> float32
        image = image.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
        image = image / self.norm_k  # [0, 255] -> [0, 1]
        image = self.norm(image)  # normalization
        return image

    def fuse_modules(self, qat: bool = False):
        """fuse_modules"""
