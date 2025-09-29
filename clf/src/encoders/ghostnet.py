"""GhostNetEncoder"""

import os
from functools import partial
from warnings import warn

import torch
from timm import create_model
from timm.models._registry import generate_default_cfgs, register_model
from timm.models.ghostnet import (
    GhostBottleneck,
    GhostModule,
    GhostNet,
    _cfg,
    _create_ghostnet,
    default_cfgs,
)

from src.encoders.base import BaseEncoder

default_cfgs.update(
    generate_default_cfgs(
        {
            "ghostnet_010.untrained": _cfg(),
            "ghostnet_020.untrained": _cfg(),
            "ghostnet_050.pretrained": _cfg(),
        }
    )
)

GHOSTNET_URLS = {
    "ghostnet_010": "/home/akobylin/logs/imagenet/encstud/ghostnet_010/version_0/checkpoints/best.pt",
    "ghostnet_020": "/home/akobylin/logs/imagenet/encstud/ghostnet_020/version_0/checkpoints/best.pt",
    "ghostnet_050": "/home/akobylin/logs/imagenet/encstud/ghostnet_050/version_0/checkpoints/best.pt",
}


@register_model
def ghostnet_010(pretrained=False, **kwargs) -> GhostNet:
    """GhostNet-0.1x"""
    model = _create_ghostnet("ghostnet_010", width=0.1, pretrained=False, **kwargs)
    if pretrained:
        name = "ghostnet_010"
        weights_path = GHOSTNET_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            warn(f"There is not pretrained weights for model '{name}'")
    return model


@register_model
def ghostnet_020(pretrained=False, **kwargs) -> GhostNet:
    """GhostNet-0.2x"""
    model = _create_ghostnet("ghostnet_020", width=0.2, pretrained=False, **kwargs)
    if pretrained:
        name = "ghostnet_020"
        weights_path = GHOSTNET_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            warn(f"There is not pretrained weights for model '{name}'")
    return model


@register_model
def ghostnet_050_pretrained(pretrained=False, **kwargs) -> GhostNet:
    """GhostNet-0.5x-pretrained"""
    model = _create_ghostnet("ghostnet_050_pretrained", width=0.5, pretrained=False, **kwargs)
    if pretrained:
        name = "ghostnet_050"
        weights_path = GHOSTNET_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            warn(f"There is not pretrained weights for model '{name}'")
    return model


class GhostNetEncoder(BaseEncoder, registry_name="ghostnet"):
    """GhostNetEncoder"""

    IN_SIZE = {
        "ghostnet_010": 224,
        "ghostnet_020": 224,
        "ghostnet_050": 224,
        "ghostnet_100": 224,
        "ghostnet_130": 224,
        "ghostnetv2_100": 224,
        "ghostnetv2_130": 224,
        "ghostnetv2_160": 224,
    }
    OUT_CHANNELS = {
        "ghostnet_010": [4, 4, 8, 16],
        "ghostnet_020": [8, 8, 16, 32],
        "ghostnet_050": [12, 20, 40, 80],
        "ghostnet_100": [24, 40, 80, 160],
        "ghostnet_130": [32, 52, 104, 208],
        "ghostnetv2_100": [24, 40, 80, 160],
        "ghostnetv2_130": [32, 52, 104, 208],
        "ghostnetv2_160": [40, 64, 128, 256],
    }
    INSTANCE = {
        "ghostnet_010": partial(
            create_model,
            model_name="ghostnet_010",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "ghostnet_020": partial(
            create_model,
            model_name="ghostnet_020",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "ghostnet_050": partial(
            create_model,
            model_name="ghostnet_050_pretrained",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "ghostnet_100": partial(
            create_model,
            model_name="ghostnet_100",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "ghostnet_130": partial(
            create_model,
            model_name="ghostnet_130",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "ghostnetv2_100": partial(
            create_model,
            model_name="ghostnetv2_100",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "ghostnetv2_130": partial(
            create_model,
            model_name="ghostnetv2_130",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "ghostnetv2_160": partial(
            create_model,
            model_name="ghostnetv2_160",
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
            if isinstance(m, GhostNet):
                fuse_modules(m, ["conv_stem", "bn1"], inplace=True)
            elif isinstance(m, GhostModule):
                if isinstance(m.primary_conv[2], torch.nn.ReLU):
                    fuse_modules(m.primary_conv, ["0", "1", "2"], inplace=True)
                else:
                    fuse_modules(m.primary_conv, ["0", "1"], inplace=True)
                if isinstance(m.cheap_operation[2], torch.nn.ReLU):
                    fuse_modules(m.cheap_operation, ["0", "1", "2"], inplace=True)
                else:
                    fuse_modules(m.cheap_operation, ["0", "1"], inplace=True)
            elif isinstance(m, GhostBottleneck):
                if m.se:
                    fuse_modules(m.se, ["conv_reduce", "act1"], inplace=True)
                if m.conv_dw and m.bn_dw:
                    fuse_modules(m, ["conv_dw", "bn_dw"], inplace=True)
                if m.shortcut and len(m.shortcut):
                    fuse_modules(m.shortcut, ["0", "1"], inplace=True)
                    fuse_modules(m.shortcut, ["2", "3"], inplace=True)
