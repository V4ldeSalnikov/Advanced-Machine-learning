"""
Latent DDPM — a DDPM that operates in the latent space of a pre-trained β-VAE.

Pipeline
--------
1.  Train a β-VAE with Gaussian likelihood on standard MNIST.
2.  Encode the training set → obtain latent codes z ~ q(z|x).
3.  Train a DDPM on those latent codes.
4.  To generate an image:  z ~ DDPM  →  x = decoder_mean(z).

This module provides:
    * ``LatentDDPMModel`` — full pipeline wrapped in the ``GenerativeModel``
      interface for seamless comparison with the pixel-space DDPM and the VAE.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from .ddpm import DDPM, FcNetwork, train as ddpm_train
from .base import GenerativeModel
from .registry import ModelRegistry
from .beta_vae import (
    BetaVAE,
    build_beta_vae,
    train_beta_vae,
)


# ---------------------------------------------------------------
# Deeper FC network for the latent DDPM (latent dim is small, no
# need for U-Net)
# ---------------------------------------------------------------

class LatentFcNetwork(nn.Module):
    """
    FC noise-prediction network for use inside the latent-space DDPM.
    Takes (z_t, t) and predicts the noise ε added to z_t.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(latent_dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# ---------------------------------------------------------------
# Latent-encoding helper
# ---------------------------------------------------------------

def encode_dataset(vae: BetaVAE, loader, device: str) -> torch.utils.data.DataLoader:
    """
    Encode an entire DataLoader through the VAE encoder and return a new
    DataLoader of latent codes (using the posterior **mean**, not samples,
    for a clean training signal).
    """
    vae.eval()
    all_z = []
    with torch.no_grad():
        for x in tqdm(loader, desc="Encoding dataset"):
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            q = vae.encode(x)
            all_z.append(q.mean.cpu())
    all_z = torch.cat(all_z, dim=0)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(all_z),
        batch_size=loader.batch_size,
        shuffle=True,
    )


# ---------------------------------------------------------------
# Main model
# ---------------------------------------------------------------

@ModelRegistry.register("latent_ddpm")
class LatentDDPMModel(GenerativeModel):
    """
    Full latent-DDPM pipeline:  β-VAE encoder → DDPM in z-space → VAE decoder.

    Parameters
    ----------
    beta : float
        β weight for the VAE's KL term.
    latent_dim : int
        Dimensionality of the VAE latent space (= input dim of the DDPM).
    vae_hidden : int
        Hidden width for  VAE encoder/decoder.
    ddpm_hidden : int
        Hidden width for the latent DDPM FC network.
    ddpm_layers : int
        Number of layers in the latent DDPM FC network.
    T : int
        Number of diffusion steps.
    beta_1, beta_T : float
        DDPM noise schedule endpoints.
    """

    def __init__(
        self,
        device: str = "cpu",
        beta: float = 1e-6,
        latent_dim: int = 32,
        vae_hidden: int = 512,
        ddpm_hidden: int = 256,
        ddpm_layers: int = 4,
        T: int = 1000,
        beta_1: float = 1e-4,
        beta_T: float = 2e-2,
    ):
        super().__init__(device)
        self.beta = beta
        self.latent_dim = latent_dim
        self.vae_hidden = vae_hidden
        self.ddpm_hidden = ddpm_hidden
        self.ddpm_layers = ddpm_layers
        self.T = T
        self.beta_1 = beta_1
        self.beta_T = beta_T

        # Build sub-models
        self.vae: BetaVAE = build_beta_vae(
            latent_dim=latent_dim, beta=beta,
            hidden=vae_hidden, device=device,
        )
        net = LatentFcNetwork(latent_dim, ddpm_hidden, ddpm_layers)
        self.ddpm: DDPM = DDPM(net, beta_1=beta_1, beta_T=beta_T, T=T).to(self.device)

        # ``self.model`` is set to the DDPM for the base-class helpers, but
        # we also carry the VAE separately.
        self.model = self.ddpm

    # ------------------------------------------------------------------
    # Training (two-stage)
    # ------------------------------------------------------------------

    def train_vae(self, train_loader, epochs: int = 50, lr: float = 1e-3, beta_warmup_epochs: int = None):
        """
        Stage 1: train the β-VAE.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        beta_warmup_epochs : int
            Number of epochs to warmup β from 0 to target (default: half of epochs).
            This helps prevent posterior collapse.
        """
        if beta_warmup_epochs is None:
            beta_warmup_epochs = epochs // 2
        train_beta_vae(self.vae, train_loader, epochs, lr=lr, device=str(self.device),
                       beta_warmup_epochs=beta_warmup_epochs)

    def train_ddpm(self, train_loader, epochs: int = 100, lr: float = 2e-4):
        """
        Stage 2: encode the dataset and train the latent DDPM.

        *train_loader* should supply the **original** images (not latent
        codes) — encoding is performed inside this method.
        """
        latent_loader = encode_dataset(self.vae, train_loader, str(self.device))
        optimizer = torch.optim.Adam(self.ddpm.parameters(), lr=lr)
        ddpm_train(self.ddpm, optimizer, latent_loader, epochs, self.device)

    def train_model(self, train_loader, epochs: int = 50, lr: float = 1e-3,
                    vae_epochs: int | None = None, ddpm_epochs: int | None = None,
                    vae_lr: float = 1e-3, ddpm_lr: float = 2e-4, **kwargs):
        """
        Convenience method: run both stages sequentially.

        Parameters
        ----------
        vae_epochs / ddpm_epochs :
            If given, override *epochs* for the respective stage.
        """
        ve = vae_epochs if vae_epochs is not None else epochs
        de = ddpm_epochs if ddpm_epochs is not None else epochs
        self.train_vae(train_loader, epochs=ve, lr=vae_lr)
        self.train_ddpm(train_loader, epochs=de, lr=ddpm_lr)
        self._is_trained = True

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        z ~ DDPM  →  x = decoder_mean(z).
        Returns images in [0, 1] of shape (n_samples, 1, 28, 28).
        """
        self.ddpm.eval()
        self.vae.eval()
        with torch.no_grad():
            z = self.ddpm.sample((n_samples, self.latent_dim))
            flat = self.vae.decode_mean(z).cpu()
        imgs = flat.clamp(0.0, 1.0)
        return imgs.view(n_samples, 1, 28, 28)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "vae_state": self.vae.state_dict(),
            "ddpm_state": self.ddpm.state_dict(),
            "config": {
                "beta": self.beta,
                "latent_dim": self.latent_dim,
                "vae_hidden": self.vae_hidden,
                "ddpm_hidden": self.ddpm_hidden,
                "ddpm_layers": self.ddpm_layers,
                "T": self.T,
                "beta_1": self.beta_1,
                "beta_T": self.beta_T,
            },
        }, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt["config"]
        # Rebuild if architecture differs from __init__ defaults
        if cfg["latent_dim"] != self.latent_dim or cfg["ddpm_hidden"] != self.ddpm_hidden:
            self.__init__(device=str(self.device), **cfg)  # type: ignore[misc]
        self.vae.load_state_dict(ckpt["vae_state"])
        self.ddpm.load_state_dict(ckpt["ddpm_state"])
        self._is_trained = True

    # ------------------------------------------------------------------
    # Analysis helpers  (for plotting prior vs aggregate posterior, etc.)
    # ------------------------------------------------------------------

    def get_aggregate_posterior_samples(self, data_loader, max_samples: int = 10000):
        """
        Encode real data through the VAE encoder and collect z samples
        (aggregate posterior).
        """
        self.vae.eval()
        zs = []
        n = 0
        with torch.no_grad():
            for x in data_loader:
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = x.to(self.device)
                q = self.vae.encode(x)
                zs.append(q.mean.cpu())
                n += x.shape[0]
                if n >= max_samples:
                    break
        return torch.cat(zs, dim=0)[:max_samples]

    def get_ddpm_prior_samples(self, n_samples: int = 10000):
        """Sample latent codes from the trained DDPM (learned prior)."""
        self.ddpm.eval()
        with torch.no_grad():
            return self.ddpm.sample((n_samples, self.latent_dim)).cpu()

    def get_vae_prior_samples(self, n_samples: int = 10000):
        """Sample from the standard Gaussian VAE prior p(z) = N(0, I)."""
        return torch.randn(n_samples, self.latent_dim)
