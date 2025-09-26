"""MiTBackbone"""

import math
import os
from functools import partial
from typing import Callable

import torch
from timm import create_model
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models._registry import generate_default_cfgs, register_model
from torch import nn

from src.encoders.base import BaseEncoder


class DWConv(nn.Module):
    """DWConv"""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): input
            h (int): height
            w (int): width

        Returns:
            torch.Tensor: output
        """
        b, _, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    """MLP"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable = nn.GELU,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """init_weights

        Args:
            m (nn.Module): module
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): input
            h (int): height
            w (int): width

        Returns:
            torch.Tensor: output
        """
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """init_weights

        Args:
            m (nn.Module): module
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): input
            h (int): height
            w (int): width

        Returns:
            torch.Tensor: output
        """
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """Block"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            dropout=dropout,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """init_weights

        Args:
            m (nn.Module): module
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): input
            h (int): height
            w (int): width

        Returns:
            torch.Tensor: output
        """
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        img_size_t = to_2tuple(img_size)
        patch_size_t = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.h, self.w = (
            img_size_t[0] // patch_size_t[0],
            img_size_t[1] // patch_size_t[1],
        )
        self.num_patches = self.h * self.w
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size_t,
            stride=stride,
            padding=(patch_size_t[0] // 2, patch_size_t[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """init_weights

        Args:
            m (nn.Module): module
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """forward

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
            int: h - height
            int: w - width
        """
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, h, w


class MixTransformer(nn.Module):
    """MixTransformer"""

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: list[int] | None = None,
        num_heads: list[int] | None = None,
        mlp_ratios: list[int] | None = None,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Callable = nn.LayerNorm,
        depths: list[int] | None = None,
        sr_ratios: list[int] | None = None,
        **kwargs: any,
    ):
        super().__init__()
        embed_dims = embed_dims or [64, 128, 256, 512]
        num_heads = num_heads or [1, 2, 4, 8]
        mlp_ratios = mlp_ratios or [4, 4, 4, 4]
        depths = depths or [3, 4, 6, 3]
        sr_ratios = sr_ratios or [8, 4, 2, 1]
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        self.neck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.head = nn.Linear(embed_dims[3], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """init_weights

        Args:
            m (nn.Module): module
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """forward_features

        Args:
            x (torch.Tensor): input

        Returns:
            list[torch.Tensor]: outputs
        """
        b = x.shape[0]
        outs = []

        # stage 1
        x, h, w = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h, w)
        x = self.norm1(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, h, w = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, h, w)
        x = self.norm2(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, h, w = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, h, w)
        x = self.norm3(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, h, w = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, h, w)
        x = self.norm4(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return tuple(outs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward_features

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        features = self.forward_features(x)[-1]
        logits = self.head(self.neck(features))
        return logits


MIT_URLS = {
    "mit-b01": "",
    "mit-b02": "",
    "mit-b04": "",
    "mit-b0": "/home/akobylin/models/mit/mit_b0.pth",
    "mit-b1": "/home/akobylin/models/mit/mit_b1.pth",
    "mit-b2": "/home/akobylin/models/mit/mit_b2.pth",
    "mit-b3": "/home/akobylin/models/mit/mit_b3.pth",
    "mit-b4": "/home/akobylin/models/mit/mit_b4.pth",
    "mit-b5": "/home/akobylin/models/mit/mit_b5.pth",
}


@register_model
def mit_b01(in_size: int = 224, pretrained: bool = True, **kwargs: any) -> nn.Module:
    """MiT-B0.1"""
    model = MixTransformer(
        img_size=in_size,
        embed_dims=[4, 8, 20, 32],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        dropout=0.0,
        drop_path=0.1,
        **kwargs,
    )
    if pretrained:
        name = "mit-b01"
        weights_path = MIT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
    return model


@register_model
def mit_b02(in_size: int = 224, pretrained: bool = True, **kwargs: any) -> nn.Module:
    """MiT-B0.2"""
    model = MixTransformer(
        img_size=in_size,
        embed_dims=[8, 16, 40, 64],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        dropout=0.0,
        drop_path=0.1,
        **kwargs,
    )
    if pretrained:
        name = "mit-b02"
        weights_path = MIT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
    return model


@register_model
def mit_b05(in_size: int = 224, pretrained: bool = True, **kwargs: any) -> nn.Module:
    """MiT-B0.5"""
    model = MixTransformer(
        img_size=in_size,
        embed_dims=[16, 32, 80, 128],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        dropout=0.0,
        drop_path=0.1,
        **kwargs,
    )
    if pretrained:
        name = "mit-b05"
        weights_path = MIT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
    return model


@register_model
def mit_b0(in_size: int = 224, pretrained: bool = True, **kwargs: any) -> nn.Module:
    """MiT-B0"""
    model = MixTransformer(
        img_size=in_size,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        dropout=0.0,
        drop_path=0.1,
        **kwargs,
    )
    if pretrained:
        name = "mit-b0"
        weights_path = MIT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
    return model


@register_model
def mit_b1(in_size: int = 224, pretrained: bool = True, **kwargs: any) -> nn.Module:
    """MiT-B1"""
    model = MixTransformer(
        img_size=in_size,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        dropout=0.0,
        drop_path=0.1,
        **kwargs,
    )
    if pretrained:
        name = "mit-b1"
        weights_path = MIT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
    return model


@register_model
def mit_b2(in_size: int = 224, pretrained: bool = True, **kwargs: any) -> nn.Module:
    """MiT-B2"""
    model = MixTransformer(
        img_size=in_size,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        dropout=0.0,
        drop_path=0.1,
        **kwargs,
    )
    if pretrained:
        name = "mit-b2"
        weights_path = MIT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
    return model


@register_model
def mit_b3(in_size: int = 224, pretrained: bool = True, **kwargs: any) -> nn.Module:
    """MiT-B3"""
    model = MixTransformer(
        img_size=in_size,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        dropout=0.0,
        drop_path=0.1,
        **kwargs,
    )
    if pretrained:
        name = "mit-b3"
        weights_path = MIT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
    return model


@register_model
def mit_b4(in_size: int = 224, pretrained: bool = True, **kwargs: any) -> nn.Module:
    """MiT-B4"""
    model = MixTransformer(
        img_size=in_size,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 8, 27, 3],
        sr_ratios=[8, 4, 2, 1],
        dropout=0.0,
        drop_path=0.1,
        **kwargs,
    )
    if pretrained:
        name = "mit-b4"
        weights_path = MIT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
    return model


@register_model
def mit_b5(in_size: int = 224, pretrained: bool = True, **kwargs: any) -> nn.Module:
    """MiT-B5"""
    model = MixTransformer(
        img_size=in_size,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 6, 40, 3],
        sr_ratios=[8, 4, 2, 1],
        dropout=0.0,
        drop_path=0.1,
        **kwargs,
    )
    if pretrained:
        name = "mit-b5"
        weights_path = MIT_URLS.get(name, "")
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
    return model


def _cfg(**kwargs):
    return {**kwargs}


default_cfgs = generate_default_cfgs(
    {
        "mit-b01.untrained": _cfg(),
        "mit-b02.untrained": _cfg(),
        "mit-b05.untrained": _cfg(),
        "mit-b0": _cfg(),
        "mit-b1": _cfg(),
        "mit-b2": _cfg(),
        "mit-b3": _cfg(),
        "mit-b4": _cfg(),
        "mit-b5": _cfg(),
    }
)


class MiTEncoder(BaseEncoder, registry_name="mit"):
    """MiTEncoder"""

    IN_SIZE = {
        "mit-b01": 224,
        "mit-b02": 224,
        "mit-b05": 224,
        "mit-b0": 224,
        "mit-b1": 224,
        "mit-b2": 224,
        "mit-b3": 224,
        "mit-b4": 224,
        "mit-b5": 224,
    }
    OUT_CHANNELS = {
        "mit-b01": [4, 8, 20, 32],
        "mit-b02": [8, 16, 40, 64],
        "mit-b05": [16, 32, 80, 128],
        "mit-b0": [32, 64, 160, 256],
        "mit-b1": [64, 128, 320, 512],
        "mit-b2": [64, 128, 320, 512],
        "mit-b3": [64, 128, 320, 512],
        "mit-b4": [64, 128, 320, 512],
        "mit-b5": [64, 128, 320, 512],
    }
    INSTANCE = {
        "mit-b01": partial(
            create_model,
            model_name="mit_b01",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mit-b02": partial(
            create_model,
            model_name="mit_b02",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mit-b05": partial(
            create_model,
            model_name="mit_b05",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mit-b0": partial(
            create_model,
            model_name="mit_b0",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mit-b1": partial(
            create_model,
            model_name="mit_b1",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mit-b2": partial(
            create_model,
            model_name="mit_b2",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mit-b3": partial(
            create_model,
            model_name="mit_b3",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mit-b4": partial(
            create_model,
            model_name="mit_b4",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
        "mit-b5": partial(
            create_model,
            model_name="mit_b5",
            scriptable=True,
            exportable=True,
            features_only=True,
        ),
    }

    def __init__(self, name: str, *args: any, pretrained: bool = True, **kwargs: any):
        in_size = kwargs.get("in_size", self.IN_SIZE[name])
        for _name in self.INSTANCE:
            self.INSTANCE[_name] = partial(self.INSTANCE[_name], in_size=in_size)
        super().__init__(name, *args, pretrained=pretrained, **kwargs)
        del self.backbone.neck, self.backbone.head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward

        Args:
            x (torch.Tensor): input

        Returns:
            tuple[torch.Tensor]: features: c2, c3, c4, c5
        """
        return self.backbone.forward_features(x)
