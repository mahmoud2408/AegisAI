"""Shared PyTorch helpers for anomaly-detection experiments."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn


@dataclass(frozen=True)
class TrainingSummary:
    """Training history and efficiency metadata."""

    train_loss: list[float]
    validation_loss: list[float]
    epochs_ran: int
    best_validation_loss: float | None
    training_seconds: float
    parameter_count: int
    peak_rss_mb: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def count_torch_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""

    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )


def process_rss_mb() -> float | None:
    """Return current process resident memory in MiB when available."""

    try:
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024 * 1024))
    except psutil.Error:
        return None


def resolve_device(requested: str) -> torch.device:
    """Resolve a configured torch device."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
