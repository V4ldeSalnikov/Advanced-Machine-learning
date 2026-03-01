"""
β-VAE with a **Gaussian** decoder p(x|z) for standard (non-binarised) MNIST.

This module provides:
    - ``GaussianDecoder``  – learns mean *and* log-variance of p(x|z)
    - ``BetaVAE``          – VAE with tuneable β for the KL term

The encoder / decoder networks are intentionally simple FC nets; swap them
out for CNNs if desired.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm


# ---------------------------------------------------------------
# Components
# ---------------------------------------------------------------

class GaussianPrior(nn.Module):
    """Standard N(0, I) prior."""

    def __init__(self, M: int):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(M), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(self.mean, self.std), 1)


class GaussianEncoder(nn.Module):
    """q(z|x): diagonal Gaussian parameterised by an encoder network."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        mean, log_std = torch.chunk(self.net(x), 2, dim=-1)
        return td.Independent(td.Normal(mean, torch.exp(log_std)), 1)


class GaussianDecoder(nn.Module):
    """
    p(x|z): diagonal Gaussian whose mean comes from a decoder network.
    A learnable per-pixel log-variance is shared across the dataset.
    """

    def __init__(self, net: nn.Module, output_dim: int):
        super().__init__()
        self.net = net
        self.log_var = nn.Parameter(torch.zeros(output_dim))

    def forward(self, z):
        mean = self.net(z)
        std = torch.exp(0.5 * self.log_var).expand_as(mean)
        return td.Independent(td.Normal(mean, std), 1)


class BetaVAE(nn.Module):
    """
    VAE with a tuneable β multiplier on the KL term.

    ELBO(x) = E_q[ log p(x|z) ] − β · KL[ q(z|x) ‖ p(z) ]
    """

    def __init__(self, prior, encoder, decoder, beta: float = 1.0):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        recon = self.decoder(z).log_prob(x)
        kl = td.kl_divergence(q, self.prior())
        return (recon - self.beta * kl).mean()

    def forward(self, x):
        """Return negative ELBO (loss)."""
        return -self.elbo(x)

    def sample(self, n_samples: int = 1):
        z = self.prior().sample((n_samples,))
        return self.decoder(z).mean  # use decoder mean (less noisy)

    def encode(self, x):
        """Return posterior q(z|x)."""
        return self.encoder(x)

    def decode_mean(self, z):
        """Deterministic decoder mean."""
        return self.decoder(z).mean


# ---------------------------------------------------------------
# Default network builders
# ---------------------------------------------------------------

def make_encoder_net(input_dim: int, latent_dim: int, hidden: int = 512) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, latent_dim * 2),  # mean + log_std
    )


def make_decoder_net(latent_dim: int, output_dim: int, hidden: int = 512) -> nn.Module:
    return nn.Sequential(
        nn.Linear(latent_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
    )


def build_beta_vae(latent_dim: int = 32, beta: float = 1.0,
                   hidden: int = 512, input_dim: int = 784,
                   device: str = "cpu") -> BetaVAE:
    """Convenience constructor for a fully-connected β-VAE on MNIST."""
    prior = GaussianPrior(latent_dim)
    enc_net = make_encoder_net(input_dim, latent_dim, hidden)
    dec_net = make_decoder_net(latent_dim, input_dim, hidden)
    encoder = GaussianEncoder(enc_net)
    decoder = GaussianDecoder(dec_net, output_dim=input_dim)
    model = BetaVAE(prior, encoder, decoder, beta=beta).to(device)
    return model


# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------

def train_beta_vae(model: BetaVAE, train_loader, epochs: int,
                   lr: float = 1e-3, device: str = "cpu"):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total = len(train_loader) * epochs
    pbar = tqdm(range(total), desc="β-VAE training")
    for epoch in range(epochs):
        for x in train_loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            pbar.update()
