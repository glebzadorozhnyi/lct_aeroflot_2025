"""ConvNextBackbone"""

import os
from functools import partial
from warnings import warn

import torch
from timm import create_model
from timm.models.convnext import (
    ConvNeXt,
    ConvNeXtBlock,
    Mlp,
    _cfgv2,
    _create_convnext,
    default_cfgs,
    register_model,
)

from src.encoders.base import BaseEncoder

default_cfgs.update(
    {
        "convnextv2_atto_05": _cfgv2(url="", hf_hub_id="timm/", test_input_size=(3, 288, 288), test_crop_pct=0.95),
        "convnextv2_atto_02": _cfgv2(url="", hf_hub_id="timm/", test_input_size=(3, 288, 288), test_crop_pct=0.95),
        "convnextv2_atto_01": _cfgv2(url="", hf_hub_id="timm/", test_input_size=(3, 288, 288), test_crop_pct=0.95),
        "convnextv2_atto_s": _cfgv2(url="", hf_hub_id="timm/", test_input_size=(3, 288, 288), test_crop_pct=0.95),
        "convnextv2_atto_s_05": _cfgv2(url="", hf_hub_id="timm/", test_input_size=(3, 288, 288), test_crop_pct=0.95),
        "convnextv2_atto_s_02": _cfgv2(url="", hf_hub_id="timm/", test_input_size=(3, 288, 288), test_crop_pct=0.95),
        "convnextv2_atto_s_01": _cfgv2(url="", hf_hub_id="timm/", test_input_size=(3, 288, 288), test_crop_pct=0.95),
    }
)

CONVNEXT_URLS = {
    "convnextv2_atto_05": "/home/akobylin/logs/imagenet/encstud/convnextv2_atto_05/version_0/checkpoints/best.pt",
    "convnextv2_atto_02": "/home/akobylin/logs/imagenet/encstud/convnextv2_atto_02/version_0/checkpoints/best.pt",
    "convnextv2_atto_01": "/home/akobylin/logs/imagenet/encstud/convnextv2_atto_01/version_0/checkpoints/best.pt",
    "convnextv2_atto_s": "/home/akobylin/logs/imagenet/encstud/convnextv2_atto_s/version_0/checkpoints/best.pt",
    "convnextv2_atto_s_05": "/home/akobylin/logs/imagenet/encstud/convnextv2_atto_s_05/version_0/checkpoints/best.pt",
    "convnextv2_atto_s_02": "/home/akobylin/logs/imagenet/encstud/convnextv2_atto_s_02/version_0/checkpoints/best.pt",
    "convnextv2_atto_s_01": "/home/akobylin/logs/imagenet/encstud/convnextv2_atto_s_01/version_0/checkpoints/best.pt",
}


@register_model
def convnextv2_atto_05(pretrained=False, **kwargs) -> ConvNeXt:
    """convnextv2_atto_05"""
    model_args = {
        "depths": (2, 2, 6, 2),
        "dims": (24, 40, 80, 160),
        "use_grn": True,
        "ls_init_value": None,
        "conv_mlp": True,
    }
    model = _create_convnext("convnextv2_atto_05", pretrained=False, **dict(model_args, **kwargs))
    if pretrained:
        name = "convnextv2_atto_05"
        weights_path = CONVNEXT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            warn(f"There is not pretrained weights for model '{name}'")
    return model


@register_model
def convnextv2_atto_02(pretrained=False, **kwargs) -> ConvNeXt:
    """convnextv2_atto_02"""
    model_args = {
        "depths": (2, 2, 6, 2),
        "dims": (8, 16, 32, 64),
        "use_grn": True,
        "ls_init_value": None,
        "conv_mlp": True,
    }
    model = _create_convnext("convnextv2_atto_02", pretrained=False, **dict(model_args, **kwargs))
    if pretrained:
        name = "convnextv2_atto_02"
        weights_path = CONVNEXT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            warn(f"There is not pretrained weights for model '{name}'")
    return model


@register_model
def convnextv2_atto_01(pretrained=False, **kwargs) -> ConvNeXt:
    """convnextv2_atto_01"""
    model_args = {
        "depths": (2, 2, 6, 2),
        "dims": (8, 8, 16, 32),
        "use_grn": True,
        "ls_init_value": None,
        "conv_mlp": True,
    }
    model = _create_convnext("convnextv2_atto_01", pretrained=False, **dict(model_args, **kwargs))
    if pretrained:
        name = "convnextv2_atto_01"
        weights_path = CONVNEXT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            warn(f"There is not pretrained weights for model '{name}'")
    return model


@register_model
def convnextv2_atto_s(pretrained=False, **kwargs) -> ConvNeXt:
    """convnextv2_atto_s"""
    model_args = {
        "depths": (2, 2, 6, 2),
        "dims": (40, 80, 160, 320),
        "ls_init_value": None,
        "conv_mlp": True,
        "conv_bias": True,
        "use_grn": False,
        "act_layer": "relu",
        "norm_layer": torch.nn.BatchNorm2d,
    }
    model = _create_convnext("convnextv2_atto", pretrained=False, **dict(model_args, **kwargs))
    if pretrained:
        name = "convnextv2_atto_s"
        weights_path = CONVNEXT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            warn(f"There is not pretrained weights for model '{name}'")
    return model


@register_model
def convnextv2_atto_s_05(pretrained=False, **kwargs) -> ConvNeXt:
    """convnextv2_atto_s_05"""
    model_args = {
        "depths": (2, 2, 6, 2),
        "dims": (24, 40, 80, 160),
        "ls_init_value": None,
        "conv_mlp": True,
        "conv_bias": True,
        "use_grn": False,
        "act_layer": "relu",
        "norm_layer": torch.nn.BatchNorm2d,
    }
    model = _create_convnext("convnextv2_atto_s_05", pretrained=False, **dict(model_args, **kwargs))
    if pretrained:
        name = "convnextv2_atto_s_05"
        weights_path = CONVNEXT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            warn(f"There is not pretrained weights for model '{name}'")
    return model


@register_model
def convnextv2_atto_s_02(pretrained=False, **kwargs) -> ConvNeXt:
    """convnextv2_atto_s_02"""
    model_args = {
        "depths": (2, 2, 6, 2),
        "dims": (8, 16, 32, 64),
        "ls_init_value": None,
        "conv_mlp": True,
        "conv_bias": True,
        "use_grn": False,
        "act_layer": "relu",
        "norm_layer": torch.nn.BatchNorm2d,
    }
    model = _create_convnext("convnextv2_atto_s_02", pretrained=False, **dict(model_args, **kwargs))
    if pretrained:
        name = "convnextv2_atto_s_02"
        weights_path = CONVNEXT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            warn(f"There is not pretrained weights for model '{name}'")
    return model


@register_model
def convnextv2_atto_s_01(pretrained=False, **kwargs) -> ConvNeXt:
    """convnextv2_atto_s_01"""
    model_args = {
        "depths": (2, 2, 6, 2),
        "dims": (8, 8, 16, 32),
        "ls_init_value": None,
        "conv_mlp": True,
        "conv_bias": True,
        "use_grn": False,
        "act_layer": "relu",
        "norm_layer": torch.nn.BatchNorm2d,
    }
    model = _create_convnext("convnextv2_atto_s_01", pretrained=False, **dict(model_args, **kwargs))
    if pretrained:
        name = "convnextv2_atto_s_01"
        weights_path = CONVNEXT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            warn(f"There is not pretrained weights for model '{name}'")
    return model


class ConvNextEncoder(BaseEncoder, registry_name="convnext"):
    """ConvNextEncoder"""

    IN_SIZE = {
        "convnext_atto": 224,
        "convnext_atto_ols": 224,
        "convnext_femto": 224,
        "convnext_femto_ols": 224,
        "convnext_pico": 224,
        "convnext_pico_ols": 224,
        "convnext_nano": 224,
        "convnext_nano_ols": 224,
        "convnext_tiny": 224,
        "convnext_tiny_hnf": 224,
        "convnext_small": 224,
        "convnext_base": 224,
        "convnext_large": 224,
        "convnext_large_mlp": 320,
        "convnext_xlarge": 224,
        "convnext_xxlarge": 256,
        "convnextv2_atto_01": 224,
        "convnextv2_atto_02": 224,
        "convnextv2_atto_05": 224,
        "convnextv2_atto": 224,
        "convnextv2_atto_s_01": 224,
        "convnextv2_atto_s_02": 224,
        "convnextv2_atto_s_05": 224,
        "convnextv2_atto_s": 224,
        "convnextv2_femto": 224,
        "convnextv2_pico": 224,
        "convnextv2_nano": 224,
        "convnextv2_tiny": 224,
        "convnextv2_base": 224,
        "convnextv2_large": 224,
        "convnextv2_huge": 384,
    }
    OUT_CHANNELS = {
        "convnext_atto": [40, 80, 160, 320],
        "convnext_atto_ols": [40, 80, 160, 320],
        "convnext_femto": [48, 96, 192, 384],
        "convnext_femto_ols": [48, 96, 192, 384],
        "convnext_pico": [64, 128, 256, 512],
        "convnext_pico_ols": [64, 128, 256, 512],
        "convnext_nano": [80, 160, 320, 640],
        "convnext_nano_ols": [80, 160, 320, 640],
        "convnext_tiny": [96, 192, 384, 768],
        "convnext_tiny_hnf": [96, 192, 384, 768],
        "convnext_small": [96, 192, 384, 768],
        "convnext_base": [128, 256, 512, 1024],
        "convnext_large": [192, 384, 768, 1536],
        "convnext_large_mlp": [192, 384, 768, 1536],
        "convnext_xlarge": [256, 512, 1024, 2048],
        "convnext_xxlarge": [384, 768, 1536, 3072],
        "convnextv2_atto_01": [8, 8, 16, 32],
        "convnextv2_atto_02": [8, 16, 32, 64],
        "convnextv2_atto_05": [24, 40, 80, 160],
        "convnextv2_atto": [40, 80, 160, 320],
        "convnextv2_atto_s_01": [8, 8, 16, 32],
        "convnextv2_atto_s_02": [8, 16, 32, 64],
        "convnextv2_atto_s_05": [24, 40, 80, 160],
        "convnextv2_atto_s": [40, 80, 160, 320],
        "convnextv2_femto": [48, 96, 192, 384],
        "convnextv2_pico": [64, 128, 256, 512],
        "convnextv2_nano": [80, 160, 320, 640],
        "convnextv2_tiny": [96, 192, 384, 768],
        "convnextv2_base": [128, 256, 512, 1024],
        "convnextv2_large": [192, 384, 768, 1536],
        "convnextv2_huge": [352, 704, 1408, 2816],
    }
    INSTANCE = {
        "convnext_atto": partial(
            create_model,
            model_name="convnext_atto",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_atto_ols": partial(
            create_model,
            model_name="convnext_atto_ols",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_femto": partial(
            create_model,
            model_name="convnext_femto",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_femto_ols": partial(
            create_model,
            model_name="convnext_femto_ols",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_pico": partial(
            create_model,
            model_name="convnext_pico",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_pico_ols": partial(
            create_model,
            model_name="convnext_pico_ols",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_nano": partial(
            create_model,
            model_name="convnext_nano",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_nano_ols": partial(
            create_model,
            model_name="convnext_nano_ols",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_tiny": partial(
            create_model,
            model_name="convnext_tiny",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_tiny_hnf": partial(
            create_model,
            model_name="convnext_tiny_hnf",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_small": partial(
            create_model,
            model_name="convnext_small",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_base": partial(
            create_model,
            model_name="convnext_base",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_large": partial(
            create_model,
            model_name="convnext_large",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_large_mlp": partial(
            create_model,
            model_name="convnext_large_mlp",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_xlarge": partial(
            create_model,
            model_name="convnext_xlarge",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnext_xxlarge": partial(
            create_model,
            model_name="convnext_xxlarge",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_atto_01": partial(
            create_model,
            model_name="convnextv2_atto_01",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_atto_02": partial(
            create_model,
            model_name="convnextv2_atto_02",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_atto_05": partial(
            create_model,
            model_name="convnextv2_atto_05",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_atto": partial(
            create_model,
            model_name="convnextv2_atto",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_atto_s_01": partial(
            create_model,
            model_name="convnextv2_atto_s_01",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_atto_s_02": partial(
            create_model,
            model_name="convnextv2_atto_s_02",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_atto_s_05": partial(
            create_model,
            model_name="convnextv2_atto_s_05",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_atto_s": partial(
            create_model,
            model_name="convnextv2_atto_s",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_femto": partial(
            create_model,
            model_name="convnextv2_femto",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_pico": partial(
            create_model,
            model_name="convnextv2_pico",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_nano": partial(
            create_model,
            model_name="convnextv2_nano",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_tiny": partial(
            create_model,
            model_name="convnextv2_tiny",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_base": partial(
            create_model,
            model_name="convnextv2_base",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_large": partial(
            create_model,
            model_name="convnextv2_large",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "convnextv2_huge": partial(
            create_model,
            model_name="convnextv2_huge",
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
        return tuple(self.backbone(x))

    def fuse_modules(self, qat: bool = False):
        """fuse_modules"""
        fuse_modules = torch.ao.quantization.fuse_modules_qat if qat else torch.ao.quantization.fuse_modules
        for m in self.modules():
            if isinstance(m, ConvNeXtBlock):
                fuse_modules(m, ["conv_dw", "norm"], inplace=True)
            elif isinstance(m, Mlp):
                fuse_modules(m, ["fc1", "act"], inplace=True)
