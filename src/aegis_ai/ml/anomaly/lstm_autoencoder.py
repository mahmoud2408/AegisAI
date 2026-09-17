"""LSTM autoencoder detector for temporal anomaly detection."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aegis_ai.ml.anomaly.torch_training import (
    TrainingSummary,
    count_torch_parameters,
    process_rss_mb,
    resolve_device,
    set_reproducible_seed,
)


@dataclass(frozen=True)
class LSTMAutoencoderConfig:
    """Configuration for the LSTM autoencoder."""

    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 4
    patience: int = 2
    min_delta: float = 1e-5
    random_seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1 or self.epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, epochs, and patience must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LSTMAutoencoderConfig:
        return cls(**payload)


@dataclass
class LSTMAutoencoderDetector:
    """Window-level LSTM autoencoder anomaly detector."""

    config: LSTMAutoencoderConfig = field(default_factory=LSTMAutoencoderConfig)
    input_size: int | None = None
    sequence_length: int | None = None
    training_summary: TrainingSummary | None = None

    def fit(
        self,
        windows: np.ndarray,
        *,
        validation_windows: np.ndarray | None = None,
        checkpoint_path: Path | None = None,
    ) -> TrainingSummary:
        """Fit the LSTM autoencoder to normal training windows."""

        train_array = _as_window_tensor(windows)
        validation_array = (
            _as_window_tensor(validation_windows) if validation_windows is not None else None
        )
        if train_array.shape[0] == 0:
            raise ValueError("cannot train LSTM autoencoder on zero windows")

        set_reproducible_seed(self.config.random_seed)
        self.sequence_length = int(train_array.shape[1])
        self.input_size = int(train_array.shape[2])
        model = _LSTMAutoencoderNetwork(
            input_size=self.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            sequence_length=self.sequence_length,
        )
        device = resolve_device(self.config.device)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        generator = torch.Generator()
        generator.manual_seed(self.config.random_seed)
        loader = _make_loader(
            train_array, self.config.batch_size, shuffle=True, generator=generator
        )

        train_losses: list[float] = []
        validation_losses: list[float] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_metric = float("inf")
        epochs_without_improvement = 0
        started = time.perf_counter()

        for _epoch in range(self.config.epochs):
            model.train()
            batch_losses: list[float] = []
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(batch_losses))
            train_losses.append(train_loss)
            validation_loss = (
                _mean_reconstruction_loss(model, validation_array, self.config.batch_size, device)
                if validation_array is not None and validation_array.shape[0] > 0
                else train_loss
            )
            validation_losses.append(validation_loss)
            if validation_loss < best_metric - self.config.min_delta:
                best_metric = validation_loss
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
                if checkpoint_path is not None:
                    self._save_state(checkpoint_path, model, best_metric)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        summary = TrainingSummary(
            train_loss=train_losses,
            validation_loss=validation_losses,
            epochs_ran=len(train_losses),
            best_validation_loss=best_metric if np.isfinite(best_metric) else None,
            training_seconds=float(time.perf_counter() - started),
            parameter_count=count_torch_parameters(model),
            peak_rss_mb=process_rss_mb(),
        )
        self.training_summary = summary
        return summary

    def reconstruction_error(self, windows: np.ndarray) -> np.ndarray:
        """Return mean squared reconstruction error per window."""

        model = self._fitted_model()
        array = _as_window_tensor(windows)
        if array.shape[0] == 0:
            return np.empty(0, dtype=float)
        device = resolve_device(self.config.device)
        model.to(device)
        model.eval()
        errors: list[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in _make_loader(array, self.config.batch_size, shuffle=False):
                batch = batch.to(device)
                reconstruction = model(batch)
                error = torch.mean((reconstruction - batch) ** 2, dim=(1, 2))
                errors.append(error.detach().cpu().numpy())
        return np.concatenate(errors).astype(float)

    def reconstruction_error_by_feature(self, windows: np.ndarray) -> np.ndarray:
        """Return reconstruction error aggregated per original feature."""

        model = self._fitted_model()
        array = _as_window_tensor(windows)
        if array.shape[0] == 0:
            return np.empty((0, int(array.shape[2])), dtype=float)
        device = resolve_device(self.config.device)
        model.to(device)
        model.eval()
        errors: list[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in _make_loader(array, self.config.batch_size, shuffle=False):
                batch = batch.to(device)
                reconstruction = model(batch)
                error = torch.mean((reconstruction - batch) ** 2, dim=1)
                errors.append(error.detach().cpu().numpy())
        return np.concatenate(errors).astype(float)

    def predict(self, windows: np.ndarray, threshold: float) -> np.ndarray:
        """Predict anomalous windows using reconstruction-error threshold."""

        return (self.reconstruction_error(windows) >= threshold).astype(int)

    def parameter_count(self) -> int:
        return count_torch_parameters(self._fitted_model())

    def save(self, path: Path) -> None:
        """Serialize the fitted model."""

        model = self._fitted_model()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "input_size": self.input_size,
                "sequence_length": self.sequence_length,
                "state_dict": model.state_dict(),
                "training_summary": (
                    self.training_summary.to_dict() if self.training_summary is not None else None
                ),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a fitted LSTM autoencoder."""

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        detector = cls(config=LSTMAutoencoderConfig.from_dict(checkpoint["config"]))
        detector.input_size = int(checkpoint["input_size"])
        detector.sequence_length = int(checkpoint["sequence_length"])
        model = _LSTMAutoencoderNetwork(
            input_size=detector.input_size,
            hidden_size=detector.config.hidden_size,
            num_layers=detector.config.num_layers,
            dropout=detector.config.dropout,
            sequence_length=detector.sequence_length,
        )
        model.load_state_dict(checkpoint["state_dict"])
        detector._model = model
        return detector

    def _fitted_model(self) -> nn.Module:
        model = getattr(self, "_model", None)
        if model is None:
            raise ValueError("model must be fitted before inference")
        return cast(nn.Module, model)

    def _save_state(self, path: Path, model: nn.Module, validation_loss: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "input_size": self.input_size,
                "sequence_length": self.sequence_length,
                "validation_loss": validation_loss,
                "state_dict": model.state_dict(),
            },
            path,
        )


class _LSTMAutoencoderNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        sequence_length: int,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.sequence_length = sequence_length
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(batch)
        context = hidden[-1].unsqueeze(1).repeat(1, self.sequence_length, 1)
        decoded, _ = self.decoder(context)
        return cast(torch.Tensor, self.output(decoded))


def _as_window_tensor(windows: np.ndarray | None) -> np.ndarray:
    if windows is None:
        raise ValueError("windows must not be None")
    array = np.asarray(windows, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("expected windows shaped [windows, sequence_length, features]")
    return array


def _make_loader(
    array: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[Any]:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _mean_reconstruction_loss(
    model: nn.Module,
    array: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    loss_fn = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_items = 0
    model.eval()
    with torch.no_grad():
        for (batch,) in _make_loader(array, batch_size, shuffle=False):
            batch = batch.to(device)
            reconstruction = model(batch)
            total_loss += float(loss_fn(reconstruction, batch).detach().cpu().item())
            total_items += int(batch.numel())
    return float(total_loss / total_items) if total_items else float("nan")
