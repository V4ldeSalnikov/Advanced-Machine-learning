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

    The observation std is **fixed** at 1.0 (log_var=0, requires_grad=False).
    This is critical: a learnable small variance (e.g. std=0.1) amplifies the
    reconstruction gradient by 1/std^2 = 100x, making the KL term negligible
    regardless of the beta value and causing prior misalignment.
    With fixed std=1, reconstruction and KL gradients are balanced, so beta
    behaves as expected.
    """

    def __init__(self, net: nn.Module, output_dim: int):
        super().__init__()
        self.net = net
        # Fixed std=1 (log_var=0). NOT learnable — a small learned variance
        # would amplify reconstruction gradients and crush the KL term.
        self.log_var = nn.Parameter(torch.zeros(output_dim), requires_grad=False)

    def forward(self, z):
        mean = self.net(z)
        std = torch.exp(0.5 * self.log_var).expand_as(mean)  # std = 1.0
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
# Default network builders (no BatchNorm — see comments in each fn)
# ---------------------------------------------------------------

def make_encoder_net(input_dim: int, latent_dim: int, hidden: int = 512) -> nn.Module:
    # No BatchNorm: with a large hidden dim (e.g. 2048) many neurons can be dead
    # after the first ReLU, giving near-zero batch variance in the second BN layer
    # → normalizes by ~316× → NaN in the encoder output.
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, latent_dim * 2),  # mean + log_std
    )


def make_decoder_net(latent_dim: int, output_dim: int, hidden: int = 512) -> nn.Module:
    # No BatchNorm in the decoder: with a small latent_dim (e.g. 16) feeding into
    # a large hidden layer (e.g. 2048), the first BN can see near-zero batch variance
    # → divides by √eps → amplifies by ~300× → final linear outputs ±inf → Sigmoid = NaN.
    return nn.Sequential(
        nn.Linear(latent_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
        nn.Sigmoid(),  # Constrain decoder mean to [0, 1] for MNIST
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
                   lr: float = 1e-3, device: str = "cpu",
                   beta_warmup_epochs: int = 0, gradient_clip: float = 1.0):
    """
    Train β-VAE with extended options for better convergence.
    
    Parameters
    ----------
    model : BetaVAE
        VAE model to train.
    train_loader : DataLoader
        Training data.
    epochs : int
        Total training epochs.
    lr : float
        Initial learning rate (default: 1e-3).
    device : str
        Device to train on.
    beta_warmup_epochs : int
        Number of epochs to warmup β from 0 to its final value (default: 0).
        Setting this to ~half of epochs helps prevent posterior collapse.
    gradient_clip : float
        Gradient clipping threshold (default: 1.0).
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    total = len(train_loader) * epochs
    pbar = tqdm(range(total), desc="β-VAE training")
    initial_beta = model.beta
    
    for epoch in range(epochs):
        # Beta warmup: linearly increase β from 0 to its target value
        if beta_warmup_epochs > 0:
            warmup_progress = min(epoch / beta_warmup_epochs, 1.0)
            model.beta = initial_beta * warmup_progress
        
        epoch_loss = 0.0
        for x in train_loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            epoch_loss = loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():12.4f}",
                beta=f"{model.beta:.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                epoch=f"{epoch+1}/{epochs}"
            )
            pbar.update()
        
        scheduler.step()
    
    # Restore final beta value
    model.beta = initial_beta
    pbar.close()
