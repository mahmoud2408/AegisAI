"""Dense autoencoder detector for windowed time-series anomaly detection."""

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
class DenseAutoencoderConfig:
    """Configuration for the dense autoencoder."""

    hidden_dims: tuple[int, ...] = (32, 16)
    latent_dim: int = 8
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 4
    patience: int = 2
    min_delta: float = 1e-5
    random_seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        if any(dim < 1 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must be positive")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1 or self.epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, epochs, and patience must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DenseAutoencoderConfig:
        values = dict(payload)
        if "hidden_dims" in values:
            values["hidden_dims"] = tuple(int(dim) for dim in values["hidden_dims"])
        return cls(**values)


@dataclass
class DenseAutoencoderDetector:
    """Window-level dense autoencoder anomaly detector."""

    config: DenseAutoencoderConfig = field(default_factory=DenseAutoencoderConfig)
    input_dim: int | None = None
    sequence_shape: tuple[int, int] | None = None
    training_summary: TrainingSummary | None = None

    def fit(
        self,
        windows: np.ndarray,
        *,
        validation_windows: np.ndarray | None = None,
        checkpoint_path: Path | None = None,
    ) -> TrainingSummary:
        """Fit the model to normal training windows."""

        train_array = _flatten_windows(windows)
        validation_array = (
            _flatten_windows(validation_windows) if validation_windows is not None else None
        )
        if train_array.shape[0] == 0:
            raise ValueError("cannot train dense autoencoder on zero windows")

        set_reproducible_seed(self.config.random_seed)
        self.sequence_shape = (int(windows.shape[1]), int(windows.shape[2]))
        self.input_dim = int(train_array.shape[1])
        model = _DenseAutoencoderNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.config.hidden_dims,
            latent_dim=self.config.latent_dim,
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
        array = _flatten_windows(windows)
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
                error = torch.mean((reconstruction - batch) ** 2, dim=1)
                errors.append(error.detach().cpu().numpy())
        return np.concatenate(errors).astype(float)

    def reconstruction_error_by_feature(self, windows: np.ndarray) -> np.ndarray:
        """Return reconstruction error aggregated per original feature."""

        model = self._fitted_model()
        window_array = np.asarray(windows, dtype=np.float32)
        array = _flatten_windows(window_array)
        if array.shape[0] == 0:
            return np.empty((0, int(window_array.shape[2])), dtype=float)
        sequence_length = int(window_array.shape[1])
        feature_count = int(window_array.shape[2])
        device = resolve_device(self.config.device)
        model.to(device)
        model.eval()
        errors: list[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in _make_loader(array, self.config.batch_size, shuffle=False):
                batch = batch.to(device)
                reconstruction = model(batch)
                raw_error = (reconstruction - batch) ** 2
                error = raw_error.reshape(-1, sequence_length, feature_count).mean(dim=1)
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
                "input_dim": self.input_dim,
                "sequence_shape": self.sequence_shape,
                "state_dict": model.state_dict(),
                "training_summary": (
                    self.training_summary.to_dict() if self.training_summary is not None else None
                ),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a fitted model."""

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        detector = cls(config=DenseAutoencoderConfig.from_dict(checkpoint["config"]))
        detector.input_dim = int(checkpoint["input_dim"])
        detector.sequence_shape = tuple(checkpoint["sequence_shape"])
        model = _DenseAutoencoderNetwork(
            input_dim=detector.input_dim,
            hidden_dims=detector.config.hidden_dims,
            latent_dim=detector.config.latent_dim,
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
                "input_dim": self.input_dim,
                "sequence_shape": self.sequence_shape,
                "validation_loss": validation_loss,
                "state_dict": model.state_dict(),
            },
            path,
        )


class _DenseAutoencoderNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        latent_dim: int,
    ) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        previous = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(previous, hidden_dim), nn.ReLU()])
            previous = hidden_dim
        encoder_layers.append(nn.Linear(previous, latent_dim))

        decoder_layers: list[nn.Module] = []
        previous = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(previous, hidden_dim), nn.ReLU()])
            previous = hidden_dim
        decoder_layers.append(nn.Linear(previous, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(self.encoder(batch)))


def _flatten_windows(windows: np.ndarray | None) -> np.ndarray:
    if windows is None:
        raise ValueError("windows must not be None")
    array = np.asarray(windows, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("expected windows shaped [windows, sequence_length, features]")
    return array.reshape(array.shape[0], -1)


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
