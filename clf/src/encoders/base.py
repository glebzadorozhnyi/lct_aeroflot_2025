"""BaseEncoder."""

from abc import ABC, abstractmethod
from typing import Callable, Type, TypeVar

import torch
from torch import nn

__all__ = ["BaseEncoder"]

T = TypeVar("T", bound="BaseEncoder")


class BaseEncoder(nn.Module, ABC):
    """BaseEncoder"""

    _registry: dict[str, Type[T]] = {}

    @classmethod
    def __init_subclass__(cls: Type[T], registry_name: str, **kwargs: any):
        super().__init_subclass__(**kwargs)
        cls._registry[registry_name] = cls

    @classmethod
    def get(cls: Type[T], registry_name: str, **kwargs: any) -> T:
        """Get instance by name

        Args:
            cls (Type[T]): instance class
            registry_name (str): class name

        Returns:
            T: instance
        """
        return cls._registry[registry_name](**kwargs)

    IN_SIZE: dict[str, int] = {}
    OUT_CHANNELSS: dict[str, list[int]] = {}
    OUT_STRIDES: list[int] = [4, 8, 16, 32]
    INSTANCE: dict[str, Callable] = {}

    def __init__(
        self,
        name: str,
        *args: any,
        in_size: int | None = None,
        **kwargs: any,
    ):
        super().__init__()
        self._name = name
        self.backbone = self.INSTANCE[name](*args, **kwargs)
        self._in_size = self.IN_SIZE[name] if in_size is None else in_size
        self._out_channelss = self.OUT_CHANNELS[name]
        self._out_channels = self.out_channelss[-1]
        self._out_strides = self.OUT_STRIDES
        self._out_stride = self.out_strides[-1]
        self._out_sizes = [self.in_size // out_stride for out_stride in self.out_strides]
        self._out_size = self.out_sizes[-1]

    @property
    def trainable_parameters(self) -> int:
        """trainable_parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def all_parameters(self) -> int:
        """all_parameters"""
        return sum(p.numel() for p in self.parameters())

    @property
    def name(self) -> str:
        """name"""
        return self._name

    @property
    def in_size(self) -> int:
        """in_size"""
        return self._in_size

    @property
    def out_channelss(self) -> list[int]:
        """out_channelss"""
        return self._out_channelss

    @property
    def out_channels(self) -> int:
        """out_channels"""
        return self._out_channels

    @property
    def out_strides(self) -> list[int]:
        """out_strides"""
        return self._out_strides

    @property
    def out_stride(self) -> int:
        """out_stride"""
        return self._out_stride

    @property
    def out_sizes(self) -> list[int]:
        """out_sizes"""
        return self._out_sizes

    @property
    def out_size(self) -> int:
        """out_size"""
        return self._out_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward

        Args:
            x (torch.Tensor): input

        Returns:
            tuple[torch.Tensor]: features: c2, c3, c4, c5
        """

    def fuse_modules(self, qat: bool = False):
        """fuse_modules"""
