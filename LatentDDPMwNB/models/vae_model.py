"""
PLACEHOLDER — Standalone VAE for the model-comparison pipeline.

This file registers the model but is **not yet implemented**.
A teammate should fill in the TODO sections below.

Can wrap either:
  - The Gaussian β-VAE from ``project.models.beta_vae`` (standard MNIST), or
  - The Bernoulli VAE from week 1 (binarised MNIST).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .base import GenerativeModel
from .registry import ModelRegistry


@ModelRegistry.register("vae")
class VAEModel(GenerativeModel):
    """
    Standalone VAE for comparison (PLACEHOLDER).

    TODO — to implement this model:
      1. Choose a VAE variant (Gaussian or Bernoulli decoder).
      2. Build the model in ``__init__``  (see ``beta_vae.py`` for helpers).
      3. Implement ``train_model``, ``sample``, ``save``, ``load``.
    See ``LatentDDPMModel`` for a working example of the interface.
    """

    def __init__(self, device: str = "cpu", beta: float = 1.0,
                 latent_dim: int = 32, hidden: int = 512):
        super().__init__(device)
        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.model = None  # TODO: build BetaVAE or week-1 VAE here

    # ---- GenerativeModel API (stubs) ----

    def train_model(self, train_loader, epochs: int = 50, lr: float = 1e-3, **kwargs):
        # TODO: implement training loop
        raise NotImplementedError("VAE training not yet implemented — placeholder model.")

    def sample(self, n_samples: int) -> torch.Tensor:
        # TODO: sample and return (n_samples, 1, 28, 28) in [0,1]
        raise NotImplementedError("VAE sampling not yet implemented — placeholder model.")

    def save(self, path: str | Path):
        # TODO: persist model weights
        raise NotImplementedError("VAE save not yet implemented — placeholder model.")

    def load(self, path: str | Path):
        # TODO: restore model weights
        raise NotImplementedError("VAE load not yet implemented — placeholder model.")
