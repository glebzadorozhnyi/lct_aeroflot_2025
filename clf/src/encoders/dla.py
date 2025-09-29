"""DLAEncoder"""

from functools import partial

import torch
from timm import create_model

from src.encoders.base import BaseEncoder


class DLAEncoder(BaseEncoder, registry_name="dls"):
    """DLAEncoder"""

    IN_SIZE = {
        "dla34": 224,
        "dla46_c": 224,
        "dla46x_c": 224,
        "dla60": 224,
        "dla60x": 224,
        "dla60x_c": 224,
        "dla60_res2next": 224,
        "dla60_res2net": 224,
        "dla102": 224,
        "dla102x": 224,
        "dla102x2": 224,
        "dla169": 224,
    }
    OUT_CHANNELS = {
        "dla34": [64, 128, 256, 512],
        "dla46_c": [64, 64, 128, 256],
        "dla46x_c": [64, 64, 128, 256],
        "dla60": [128, 256, 512, 1024],
        "dla60x": [128, 256, 512, 1024],
        "dla60x_c": [64, 64, 128, 256],
        "dla60_res2next": [128, 256, 512, 1024],
        "dla60_res2net": [128, 256, 512, 1024],
        "dla102": [128, 256, 512, 1024],
        "dla102x": [128, 256, 512, 1024],
        "dla102x2": [128, 256, 512, 1024],
        "dla169": [128, 256, 512, 1024],
    }
    INSTANCE = {
        "dla34": partial(
            create_model,
            model_name="dla34",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "dla46_c": partial(
            create_model,
            model_name="dla46_c",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "dla46x_c": partial(
            create_model,
            model_name="dla46x_c",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "dla60": partial(
            create_model,
            model_name="dla60",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "dla60x": partial(
            create_model,
            model_name="dla60x",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "dla60x_c": partial(
            create_model,
            model_name="dla60x_c",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "dla60_res2next": partial(
            create_model,
            model_name="dla60_res2next",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "dla60_res2net": partial(
            create_model,
            model_name="dla60_res2net",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "dla102": partial(
            create_model,
            model_name="dla102",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "dla102x": partial(
            create_model,
            model_name="dla102x",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "dla102x2": partial(
            create_model,
            model_name="dla102x2",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "dla169": partial(
            create_model,
            model_name="dla169",
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
