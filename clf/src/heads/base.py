"""BaseHead abstract class for neural network heads.

This module defines the foundational abstract class `BaseHead` which serves as
a common interface for all head modules in the model architecture. It includes
a class-based registry mechanism for easy instantiation of different head types.
"""

from abc import ABC, abstractmethod
from typing import Type, TypeVar

import torch
from torch import nn

__all__ = ["BaseHead"]

T = TypeVar("T", bound="BaseHead")


class BaseHead(nn.Module, ABC):
    """
    Abstract base class for all neural network heads.

    This class provides a common interface for different head architectures
    and implements a registry mechanism to dynamically create instances of
    registered head classes by name.

    Heads are typically the final layers of a neural network that convert
    features extracted by a backbone/encoder into task-specific outputs
    (e.g., classification scores, bounding box regressions).
    """

    _registry: dict[str, Type[T]] = {}

    @classmethod
    def __init_subclass__(cls: Type[T], registry_name: str, **kwargs: any):
        """
        Register subclasses into the _registry dictionary.

        This method is automatically called when a class inherits from BaseHead.
        It registers the subclass with a unique `registry_name`.

        Args:
            cls (Type[T]): The subclass being created.
            registry_name (str): The name under which the subclass will be registered.
            **kwargs (any): Additional keyword arguments passed to the super's __init_subclass__.
        """
        super().__init_subclass__(**kwargs)
        cls._registry[registry_name] = cls

    @classmethod
    def get(cls: Type[T], registry_name: str, **kwargs: any) -> T:
        """
        Retrieve and instantiate a registered head class by its name.

        Args:
            cls (Type[T]): The BaseHead class itself.
            registry_name (str): The registration name of the desired head class.
            **kwargs (any): Keyword arguments to pass to the constructor of the
                instantiated head class.

        Returns:
            T: An instance of the registered head class.

        Raises:
            KeyError: If `registry_name` is not found in the registry.
        """
        if registry_name not in cls._registry:
            raise KeyError(f"Head '{registry_name}' not found in registry. \nAvailable: {list(cls._registry.keys())}")
        return cls._registry[registry_name](**kwargs)

    def __init__(
        self,
        *args: any,
        **kwargs: any,
    ):
        """
        Initialize the BaseHead.

        Note: Subclasses should call `super().__init__(*args, **kwargs)`
        and implement their specific layer definitions.
        """
        super().__init__()

    @abstractmethod
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
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Abstract forward pass definition for the head.

        This method must be implemented by all concrete subclasses to define
        how input features are processed to produce the head's output.

        Args:
            inputs (tuple[torch.Tensor, ...]): A tuple of input tensors,
                typically feature maps from different stages of an encoder.
                Expected to contain 6 tensors (e.g., P2, P3, P4, P5, P6, P7).

        Returns:
            torch.Tensor | list[torch.Tensor]: The output of the head, which can be
                a single tensor (e.g., classification logits) or a list of tensors
                (e.g., for multi-scale outputs).
        """
        pass  # Added 'pass' explicitly for abstract method, though not strictly required.

    def fuse_modules(self, qat: bool = False) -> None:
        """
        Fuse modules for model optimization, e.g., for Quantization Aware Training (QAT).

        This method can be overridden by subclasses to perform module fusion (e.g., Conv+BN+ReLU)
        which is a common optimization technique, especially useful before quantization.

        Args:
            qat (bool, optional): If True, fusion might be QAT-specific. Defaults to False.
        """
