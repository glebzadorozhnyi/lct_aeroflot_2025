"""Cosine Annealing learning rate scheduler with warmup and restarts.

This module implements a sophisticated learning rate scheduler that combines:
- Linear warmup: Gradual increase from min_lr to max_lr
- Cosine annealing: Smooth decrease following cosine curve
- Restarts: Periodic resets to max_lr for continued exploration
- Cycle multiplication: Progressive increase in cycle lengths
- Gamma decay: Gradual reduction of max_lr across cycles

This scheduler is particularly effective for training deep networks,
helping to escape local minima and achieve better convergence.
"""

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["CosineAnnealingWarmupRestarts"]


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine Annealing learning rate scheduler with linear warmup and restarts.

    This scheduler implements the SGDR (Stochastic Gradient Descent with Warm Restarts)
    approach with additional linear warmup. The learning rate follows this pattern:

    1. Linear warmup: lr increases linearly from min_lr to max_lr over warmup_steps
    2. Cosine annealing: lr decreases following cosine curve from max_lr to min_lr
    3. Restart: lr jumps back to max_lr (possibly reduced by gamma) for next cycle

    The cycle length can be multiplied by cycle_mult after each restart, and
    the maximum learning rate can be reduced by gamma after each cycle.

    Reference:
        Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic gradient descent
        with warm restarts. arXiv preprint arXiv:1608.03983.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        first_cycle_steps (int): Number of steps in the first cycle.
        cycle_mult (float, optional): Multiplier for cycle length after each restart.
            If 1.0, all cycles have the same length. Defaults to 1.0.
        max_lr (float, optional): Maximum learning rate in the first cycle. Defaults to 0.1.
        min_lr (float, optional): Minimum learning rate (baseline). Defaults to 0.001.
        warmup_steps (int, optional): Number of linear warmup steps at the beginning
            of each cycle. Must be < first_cycle_steps. Defaults to 0.
        gamma (float, optional): Decay factor for max_lr after each cycle.
            max_lr *= gamma after each restart. Defaults to 1.0 (no decay).
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps, "Warmup steps must be less than first cycle steps"

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # current cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        # Initialize learning rate to min_lr
        self.init_lr()

    def init_lr(self) -> None:
        """
        Initialize learning rates to minimum value.

        Sets all parameter groups to min_lr and stores as base learning rates.
        """
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self) -> list[float]:
        """
        Calculate learning rates for all parameter groups.

        Returns:
            list[float]: Learning rates for each parameter group.
        """
        if self.step_in_cycle == -1:
            return self.base_lrs

        # Linear warmup phase
        if self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs
            ]

        # Cosine annealing phase
        return [
            base_lr
            + (self.max_lr - base_lr)
            * (
                1
                + math.cos(
                    math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
                )
            )
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: int | None = None) -> None:
        """
        Update learning rates for the next epoch/step.

        Args:
            epoch (int | None, optional): Current epoch number. If None,
                increments from last_epoch. Defaults to None.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1

            # Check if current cycle is complete
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                # Calculate next cycle length
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
                )
        else:
            # Calculate cycle and step from absolute epoch number
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    # Simple case: all cycles have same length
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    # Complex case: cycles grow by cycle_mult
                    n = int(
                        math.log(
                            (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult**n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                # Still in first cycle
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        # Update max learning rate with gamma decay
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)

        # Apply new learning rates to optimizer
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
