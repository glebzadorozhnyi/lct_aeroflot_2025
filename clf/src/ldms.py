"""LightningDataModule for PyTorch Lightning training.

This module provides a LightningDataModule implementation for the  dataset,
handling data loading, transforms, sampling strategies, and batch creation
for both training and validation phases.
"""

from lightning.pytorch import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from src.datasets import Dataset, collate_fn


class ClfLDM(LightningDataModule):
    """
    PyTorch Lightning DataModule for  classification dataset.

    This DataModule handles all data-related operations including:
    - Dataset creation with configurable transforms and sampling ratios
    - DataLoader setup with balanced sampling options
    - Class weight computation for handling imbalanced datasets
    - Efficient data loading with prefetching and memory pinning

    The module supports various augmentation strategies, balanced sampling,
    and optional data preloading for improved training performance.
    """

    def __init__(
        self,
        root: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        seed: int = 42,
    ):
        """
        Initialize the  LightningDataModule.

        Args:
            root (str): Path to the dataset root directory containing annotations.csv.
            image_size (int): Target image size for resizing (assumes square images).
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of worker processes for data loading.
        """
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Create datasets during initialization for class weight computation
        self._train_dataset = self.train_dataset()
        self._valid_dataset = self.valid_dataset()

    def train_dataset(self) -> Dataset:
        """
        Create and return the training dataset.

        The training dataset includes data augmentations and can be subsampled
        based on the train_ratio parameter.

        Returns:
            Dataset: Configured training dataset with augmentations.
        """
        return Dataset(
            root=self.root,
            split="train",
            image_size=self.image_size,
        )

    def valid_dataset(self) -> Dataset:
        """
        Create and return the validation dataset.

        The validation dataset uses only basic transforms (resize, normalize)
        without augmentations for consistent evaluation.

        Returns:
            Dataset: Configured validation dataset without augmentations.
        """
        return Dataset(
            root=self.root,
            split="valid",
            image_size=self.image_size,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create and return the training DataLoader.

        Configures the DataLoader with balanced sampling (if enabled), shuffling,
        and optimizations like memory pinning and prefetching.

        Returns:
            DataLoader: Configured training DataLoader with balanced sampling and optimizations.
        """
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            batch_sampler=None,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,  # Faster GPU transfer
            drop_last=True,  # Consistent batch sizes for training
            prefetch_factor=4,  # Prefetch batches for better GPU utilization
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation DataLoader.

        Configures the DataLoader for deterministic validation without shuffling
        or balanced sampling.

        Returns:
            DataLoader: Configured validation DataLoader for deterministic evaluation.
        """
        return DataLoader(
            dataset=self._valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Deterministic validation
            sampler=None,  # No custom sampling for validation
            batch_sampler=None,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,  # Faster GPU transfer
            drop_last=False,  # Keep all validation samples
            prefetch_factor=4,
        )
