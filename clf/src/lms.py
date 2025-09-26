"""Classification LightningModule for PyTorch Lightning training.

This module provides a LightningModule implementation for classification tasks,
handling training/validation loops, metrics computation, and model optimization.
It supports multi-class classification with detailed per-class and size-based metrics.
"""

from typing import Mapping

import hydra
import numpy as np
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.functional import softmax
from torchmetrics import AUROC, Accuracy

from src.datasets import Batch


class ClfLM(LightningModule):
    """
    PyTorch Lightning Module for classification tasks.

    This module handles the complete training and validation workflow for
    multi-class classification, including:
    - Optimizer and learning rate scheduler configuration
    - Training and validation step implementations
    - Comprehensive metrics computation (accuracy, AUROC) per class and object size
    - Automatic logging of metrics and losses

    The module supports detailed evaluation with size-based metrics to analyze
    model performance across different object scales.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the Classification LightningModule.

        Args:
            cfg (DictConfig): Hydra configuration containing model, optimizer,
                scheduler, and training parameters.
        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_classes = cfg.num_classes

        # Instantiate model
        self.model = hydra.utils.instantiate(cfg.model)

        # Initialize metrics for validation
        self.acc_macro = Accuracy(
            task="multiclass",
            threshold=0.5,
            num_classes=cfg.num_classes,
            average="macro",
            multidim_average="global",
            top_k=1,
        )
        self.auroc = AUROC(task="multiclass", num_classes=cfg.num_classes, average="macro")

        # Storage for detailed validation metrics
        self.targets: list[np.ndarray] = []
        self.predictions: list[np.ndarray] = []

        # Note: Uncomment if you want to save hyperparameters
        self.save_hyperparameters(cfg)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            OptimizerLRScheduler: Dictionary containing optimizer and lr_scheduler
                configured from the Hydra config.
        """
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.model.parameters())
        lr_scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, image: torch.Tensor, **kwargs: any) -> torch.Tensor:
        """
        Forward pass for inference.

        Args:
            image (torch.Tensor): Input image tensor with shape (B, C, H, W).
            **kwargs (any): Additional keyword arguments.

        Returns:
            torch.Tensor: Softmax probabilities with shape (B, num_classes).
        """
        logit, feature = self.model(image)
        p = softmax(logit, dim=-1)
        return p, feature

    def training_step(self, batch: Batch, *args: any, **kwargs: any) -> torch.Tensor | Mapping[str, any] | None:
        """
        Training step for one batch.

        Args:
            batch (Batch): Batch containing images and targets.
            *args (any): Additional positional arguments.
            **kwargs (any): Additional keyword arguments.

        Returns:
            torch.Tensor: Training loss for backpropagation.
        """
        res = self.model(batch.image, batch.target)

        # Log training metrics
        self.log_dict(
            {
                "loss/train": res["loss"],
                "clsloss/train": res["clsloss"],
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return res["loss"]

    def on_validation_epoch_start(self) -> None:
        """
        Reset validation metrics storage at the start of each validation epoch.
        """
        self.targets: list[np.ndarray] = []
        self.predictions: list[np.ndarray] = []

    def on_validation_epoch_end(self) -> None:
        """
        Compute and log detailed validation metrics at the end of each validation epoch.

        Computes:
        - Per-class accuracy and AUROC
        - Size-based macro accuracy for different object size ranges
        """
        # Concatenate all validation predictions and targets
        t = np.concatenate(self.targets, axis=0)
        p = np.concatenate(self.predictions, axis=0)

        metrics = {}

        # Compute per-class metrics (accuracy and AUROC)
        for target in range(self.num_classes):
            t_ = (t == target).astype(np.int32)  # Binary target for current class
            p_ = p[..., target]  # Probabilities for current class
            pr_ = (p_ > 0.5).astype(np.int32)  # Binary predictions

            metrics.update(
                {
                    f"acc/{target}/valid": accuracy_score(t_, pr_),
                    f"auroc/{target}/valid": 0 if len(np.unique(t_)) < 2 else roc_auc_score(t_, p_),
                }
            )

        # Log all computed metrics
        self.log_dict(
            metrics,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def validation_step(self, batch: Batch, *args: any, **kwargs: any) -> torch.Tensor | Mapping[str, any] | None:
        """
        Validation step for one batch.

        Args:
            batch (Batch): Batch containing images, targets, and sizes.
            *args (any): Additional positional arguments.
            **kwargs (any): Additional keyword arguments.

        Returns:
            torch.Tensor: Validation loss.
        """
        res = self.model(batch.image, batch.target)
        p = softmax(res["logit"], dim=-1)

        # Store predictions and targets for detailed metrics computation
        self.targets.append(batch.target.clone().detach().cpu().numpy())
        self.predictions.append(p.clone().detach().cpu().numpy())

        # Update torchmetrics
        self.acc_macro(p, batch.target)
        self.auroc(p, batch.target)

        # Log validation metrics
        self.log_dict(
            {
                "loss/valid": res["loss"],
                "clsloss/valid": res["clsloss"],
                "acc/macro/valid": self.acc_macro,
                "auroc/macro/valid": self.auroc,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return res["loss"]

    def transfer_batch_to_device(self, batch: Batch, device: torch.device, dataloader_idx: int) -> any:
        """
        Transfer batch tensors to the specified device (GPU/CPU).

        Args:
            batch (Batch): Batch to transfer.
            device (torch.device): Target device.
            dataloader_idx (int): Index of the dataloader (unused).

        Returns:
            Batch: Batch with tensors moved to the target device.
        """
        batch.target = batch.target.to(device)
        batch.image = batch.image.to(device)
        return batch
