"""
Abstract base class for all generative models used in the comparison.
Every model must implement: train_model, sample, load, save.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class GenerativeModel(ABC):
    """
    Unified interface for generative models (DDPM, Latent DDPM, VAE, …).

    Subclasses must set ``self.model`` to the underlying ``nn.Module`` and
    implement the four abstract methods below.
    """

    name: str = "base"  # human-readable name for reports / plots

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model: nn.Module | None = None
        self._is_trained = False

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    @abstractmethod
    def train_model(self, train_loader, epochs: int, lr: float, **kwargs):
        """Train the model from scratch."""
        ...

    @abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Return ``n_samples`` generated images as a tensor of shape
        ``(n_samples, 1, 28, 28)`` with values in [0, 1].
        """
        ...

    @abstractmethod
    def save(self, path: str | Path):
        """Persist model weights to *path*."""
        ...

    @abstractmethod
    def load(self, path: str | Path):
        """Restore model weights from *path*."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def to(self, device: str):
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self

    def eval(self):
        if self.model is not None:
            self.model.eval()

    def timed_sample(self, n_samples: int):
        """
        Sample *n_samples* images and return ``(samples, elapsed_seconds)``.
        ``samples`` has shape ``(n_samples, 1, 28, 28)`` in [0, 1].
        """
        self.eval()
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            samples = self.sample(n_samples)
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        elapsed = time.perf_counter() - start
        return samples, elapsed

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name!r} device={self.device}>"
