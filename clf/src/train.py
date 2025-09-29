"""Training pipeline for PyTorch Lightning models.

This module provides the complete training workflow including:
- Model and data module instantiation from Hydra configs
- Training with PyTorch Lightning Trainer
- FLOPS computation for efficiency analysis
- Integration with Optuna for hyperparameter optimization
- Automatic model export after training

The training pipeline supports various features like checkpointing,
early stopping, and pruning for hyperparameter optimization.
"""

import warnings
from dataclasses import dataclass

import hydra
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf

from src.export import export

warnings.filterwarnings("ignore")


@dataclass
class TrainResult:
    """
    Results from a training run.

    Attributes:
        best_model_path (str): Path to the best model checkpoint saved during training.
        best_model_score (float): Best validation score achieved during training.
    """

    best_model_path: str = ""
    best_model_score: float = 0.0


def train(cfg: DictConfig) -> TrainResult:
    """
    Execute the complete training pipeline.

    This function handles the entire training workflow:
    1. Sets up reproducible training environment
    2. Instantiates data module and lightning module from config
    3. Creates trainer with specified callbacks and settings
    4. Runs training with optional Optuna pruning
    5. Computes model FLOPS for efficiency analysis
    6. Returns training results for further processing

    Args:
        cfg (DictConfig): Complete Hydra configuration containing all training
            parameters, model architecture, data settings, and trainer configuration.

    Returns:
        TrainResult: Training results containing best model path, score, and FLOPS.
    """
    # Set up reproducible training environment
    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")  # Optimize for A100/H100 GPUs

    # Instantiate components from Hydra config
    ldm: LightningDataModule = hydra.utils.instantiate(cfg.ldm)
    lm: LightningModule = hydra.utils.instantiate(cfg.lm, cfg)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # Run training
    trainer.fit(model=lm, datamodule=ldm, ckpt_path=cfg.ckpt_path)

    # Compute model FLOPS for efficiency analysis
    lm.eval()

    return TrainResult(
        best_model_path=trainer.checkpoint_callback.best_model_path,
        best_model_score=trainer.checkpoint_callback.best_model_score.item(),
    )


@hydra.main(config_path="../configs", config_name="cosine-112.yaml", version_base=None)
def main(cfg: DictConfig | None = None):
    """
    Main training function with automatic model export.

    Runs the complete training pipeline followed by model export to various
    deployment formats. The function:
    1. Executes training with the provided configuration
    2. Updates config with the best model checkpoint path
    3. Exports the trained model to multiple formats (ONNX, TensorRT, etc.)

    Args:
        cfg (DictConfig | None, optional): Hydra configuration loaded from YAML files.
            Contains all training, model, and export settings. Defaults to None.
    """
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Run training
    train_result = train(cfg)

    # Update config with best model path for export
    cfg.ckpt_path = train_result.best_model_path

    # Export trained model to deployment formats
    export(cfg)


if __name__ == "__main__":
    main()
