"""
Pixel-space DDPM with a U-Net backbone on MNIST.

Ported from
-----------
``DDPM/ddpm.py``  — DDPM core and training loop
    (DTU course 02460 code by Jes Frellsen, 2024)
``DDPM/unet.py``  — U-Net noise-prediction network
    (based on code by Muhammad Firmansyah Kasim, MIT License, 2022)

Both files are located in ``Advanced-Machine-learning/DDPM/`` and are used
here unchanged through the shared ``models/ddpm.py`` re-export.  This module
provides only the ``GenerativeModel`` wrapper so the pixel-space DDPM fits
seamlessly into the comparison pipeline.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .ddpm import DDPM, Unet, train as ddpm_train
from .base import GenerativeModel
from .registry import ModelRegistry


@ModelRegistry.register("ddpm_unet")
class DDPMUNetModel(GenerativeModel):
    """
    Pixel-space DDPM with a U-Net backbone.

    Uses the ``Unet`` architecture for noise prediction and the standard
    DDPM training / sampling algorithms.  Data is expected as flat tensors
    in [-1, 1] (use ``project.data.get_ddpm_mnist``).
    """

    def __init__(self, device: str = "cpu", T: int = 1000,
                 beta_1: float = 1e-4, beta_T: float = 2e-2):
        super().__init__(device)
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T

        network = Unet()
        self.ddpm = DDPM(network, beta_1=beta_1, beta_T=beta_T, T=T).to(self.device)
        self.model = self.ddpm  # base-class helpers reference self.model

    # ------------------------------------------------------------------
    # GenerativeModel API
    # ------------------------------------------------------------------

    def train_model(self, train_loader, epochs: int = 100, lr: float = 2e-4, **kwargs):
        """Train the pixel-space DDPM with the U-Net backbone."""
        optimizer = torch.optim.Adam(self.ddpm.parameters(), lr=lr)
        ddpm_train(self.ddpm, optimizer, train_loader, epochs, self.device)
        self._is_trained = True

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from the trained DDPM.

        Returns images in [0, 1] of shape ``(n_samples, 1, 28, 28)``.
        """
        self.ddpm.eval()
        D = 28 * 28
        with torch.no_grad():
            samples = self.ddpm.sample((n_samples, D)).cpu()
        # Map from [-1, 1] → [0, 1]
        samples = samples / 2 + 0.5
        samples = samples.clamp(0.0, 1.0)
        return samples.view(n_samples, 1, 28, 28)

    def save(self, path: str | Path):
        """Persist DDPM + U-Net weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "ddpm_state": self.ddpm.state_dict(),
            "config": {
                "T": self.T,
                "beta_1": self.beta_1,
                "beta_T": self.beta_T,
            },
        }, path)

    def load(self, path: str | Path):
        """Restore DDPM + U-Net weights."""
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt["config"]
        # Rebuild if architecture config differs
        if cfg["T"] != self.T:
            self.__init__(device=str(self.device), **cfg)  # type: ignore[misc]
        self.ddpm.load_state_dict(ckpt["ddpm_state"])
        self._is_trained = True
