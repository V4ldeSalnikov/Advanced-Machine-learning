"""
Self-contained DDPM implementation for the project.

Ported from
-----------
``DDPM/ddpm.py``
    Source: ``Advanced-Machine-learning/DDPM/ddpm.py``
    DTU course 02460 code by Jes Frellsen (2024).
    Contains the ``DDPM`` class and the ``train`` function.

``DDPM/unet.py``
    Source: ``Advanced-Machine-learning/DDPM/unet.py``
    Based on code by Muhammad Firmansyah Kasim (MIT License, 2022).
    Contains the ``Unet`` noise-prediction network for 28×28 MNIST.

This file re-exports both, plus adds ``FcNetwork`` (a lightweight FC
alternative to U-Net used by the latent-space DDPM).

Contents:
    - ``DDPM``          — Denoising Diffusion Probabilistic Model
    - ``train``         — generic DDPM training loop
    - ``FcNetwork``     — simple fully-connected noise-prediction network
    - ``Unet``          — U-Net noise-prediction network for 28×28 MNIST
"""

import torch
import torch.nn as nn
from tqdm import tqdm


# ===================================================================
# DDPM core
# ===================================================================

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters
        ----------
        network : nn.Module
            The network to use for the diffusion process.
        beta_1 : float
            The noise at the first step of the diffusion process.
        beta_T : float
            The noise at the last step of the diffusion process.
        T : int
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)

    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters
        ----------
        x : torch.Tensor
            A batch of data (x) of dimension ``(batch_size, *)``.

        Returns
        -------
        torch.Tensor
            The negative ELBO of the batch of dimension ``(batch_size,)``.
        """
        batch_size = x.shape[0]
        device = x.device

        # Sample a time step uniformly for each data point
        t = torch.randint(1, self.T + 1, (batch_size,), device=device)

        # Sample Gaussian noise
        epsilon = torch.randn_like(x)
        alpha_bar_t = self.alpha_cumprod[t - 1]
        alpha_bar_t = alpha_bar_t.reshape(batch_size, *([1] * (x.dim() - 1)))
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1.0 - alpha_bar_t) * epsilon

        # Predict noise using normalized time input
        t_input = (t.float() / self.T).unsqueeze(-1)
        epsilon_theta = self.network(x_t, t_input)

        # Per-sample MSE loss
        neg_elbo = (epsilon_theta - epsilon).square()
        neg_elbo = neg_elbo.flatten(start_dim=1).sum(dim=1)

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters
        ----------
        shape : tuple
            The shape of the samples to generate.

        Returns
        -------
        torch.Tensor
            The generated samples.
        """
        x_t = torch.randn(shape).to(self.alpha.device)
        batch_size = shape[0]

        for t in range(self.T - 1, -1, -1):
            alpha_t = self.alpha[t]
            beta_t = self.beta[t]
            alpha_bar_t = self.alpha_cumprod[t]

            t_input = torch.full(
                (batch_size, 1),
                (t + 1) / self.T,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            epsilon_theta = self.network(x_t, t_input)

            z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

            x_t = (
                (x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * epsilon_theta)
                / torch.sqrt(alpha_t)
                + torch.sqrt(beta_t) * z
            )

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters
        ----------
        x : torch.Tensor
            A batch of data (x) of dimension ``(batch_size, *)``.

        Returns
        -------
        torch.Tensor
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


# ===================================================================
# Training loop
# ===================================================================

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a DDPM model.

    Parameters
    ----------
    model : DDPM
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    data_loader : torch.utils.data.DataLoader
        The data loader to use for training.
    epochs : int
        Number of epochs to train for.
    device : torch.device
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(
                loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}"
            )
            progress_bar.update()


# ===================================================================
# FC noise-prediction network
# ===================================================================

class FcNetwork(nn.Module):
    """Simple fully-connected noise-prediction network for the DDPM."""

    def __init__(self, input_dim, num_hidden):
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, input_dim),
        )

    def forward(self, x, t):
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


# ===================================================================
# U-Net noise-prediction network for 28×28 MNIST
# ===================================================================
# Based on code by Muhammad Firmansyah Kasim (MIT License, 2022)
# https://github.com/mfkasim1/score-based-tutorial/blob/main/03-SGM-with-SDE-MNIST.ipynb

class Unet(torch.nn.Module):
    """
    A simple U-Net architecture for MNIST that takes an input image and time.
    """

    def __init__(self):
        super().__init__()
        chs = [32, 64, 128, 256, 256]
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(2, chs[0], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
            ),
        ])
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
                torch.nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1)
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))
        tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)
        x2t = torch.cat((x2, tt), dim=-3)
        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))
        return signal
