"""MobileNetEncoder"""

from functools import partial

import torch
from timm import create_model
from timm.models._efficientnet_blocks import SqueezeExcite
from timm.models.mobilenetv3 import MobileNetV3Features

from src.encoders.base import BaseEncoder


class MobileNetEncoder(BaseEncoder, registry_name="mobilenet"):
    """MobileNetEncoder"""

    IN_SIZE = {
        "mobilenetv3_large_100": 224,
        "mobilenetv3_small_050": 224,
        "mobilenetv3_small_075": 224,
        "mobilenetv3_small_100": 224,
        "mobilenetv3_rw": 224,
        "fbnetv3_b": 224,
        "fbnetv3_d": 224,
        "fbnetv3_g": 224,
        "lcnet_050": 224,
        "lcnet_075": 224,
        "lcnet_100": 224,
    }
    OUT_CHANNELS = {
        "mobilenetv3_large_100": [24, 40, 112, 960],
        "mobilenetv3_small_050": [8, 16, 24, 288],
        "mobilenetv3_small_075": [16, 24, 40, 432],
        "mobilenetv3_small_100": [16, 24, 48, 576],
        "mobilenetv3_rw": [24, 40, 112, 960],
        "fbnetv3_b": [24, 40, 120, 1344],
        "fbnetv3_d": [24, 40, 128, 1440],
        "fbnetv3_g": [40, 56, 160, 1728],
        "lcnet_050": [32, 64, 128, 256],
        "lcnet_075": [48, 96, 192, 384],
        "lcnet_100": [64, 128, 256, 512],
    }
    INSTANCE = {
        "mobilenetv3_large_100": partial(
            create_model,
            model_name="mobilenetv3_large_100",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mobilenetv3_small_050": partial(
            create_model,
            model_name="mobilenetv3_small_050",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mobilenetv3_small_075": partial(
            create_model,
            model_name="mobilenetv3_small_075",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mobilenetv3_small_100": partial(
            create_model,
            model_name="mobilenetv3_small_100",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mobilenetv3_rw": partial(
            create_model,
            model_name="mobilenetv3_rw",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "fbnetv3_b": partial(
            create_model,
            model_name="fbnetv3_b",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "fbnetv3_d": partial(
            create_model,
            model_name="fbnetv3_d",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "fbnetv3_g": partial(
            create_model,
            model_name="fbnetv3_g",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "lcnet_050": partial(
            create_model,
            model_name="lcnet_050",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "lcnet_075": partial(
            create_model,
            model_name="lcnet_075",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "lcnet_100": partial(
            create_model,
            model_name="lcnet_100",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
    }

    def __init__(
        self,
        name: str,
        *args: torch.any,
        pretrained: bool = True,
        **kwargs: torch.any,
    ):
        super().__init__(name, *args, pretrained=pretrained, **kwargs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward

        Args:
            x (torch.Tensor): input

        Returns:
            tuple[torch.Tensor]: features: c2, c3, c4, c5
        """
        return tuple(self.backbone(x)[1:])

    def fuse_modules(self, qat: bool = False):
        """fuse_modules"""
        fuse_modules = torch.ao.quantization.fuse_modules_qat if qat else torch.ao.quantization.fuse_modules
        for m in self.modules():
            if isinstance(m, MobileNetV3Features):
                fuse_modules(m, ["conv_stem", "bn1"], inplace=True)
            elif isinstance(m, SqueezeExcite):
                fuse_modules(m, ["conv_reduce", "act1"], inplace=True)


if __name__ == "__main__":
    model = MobileNetEncoder("mobilenetv3_small_100")
    model.eval()
    model.fuse_modules()
    print(model)
