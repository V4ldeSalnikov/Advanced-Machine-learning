"""
Fréchet Inception Distance (FID) for MNIST.

Since MNIST images are 28×28 greyscale (too small for the standard Inception-v3
network used in FID), we use a **LeNet-5–like** classifier trained on MNIST as
the feature extractor, following the approach described in the course.

If a file ``fid.py`` is later provided by the course, you can swap
``compute_fid`` in the comparison pipeline to use that instead.

Public API
----------
    compute_fid(real_images, generated_images, device="cpu") -> float
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm
from torchvision import datasets, transforms


# ------------------------------------------------------------------
# Feature extractor (simple LeNet-style CNN trained on MNIST)
# ------------------------------------------------------------------

class _MNISTFeatureNet(nn.Module):
    """Small CNN whose penultimate layer serves as the feature space."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        """Return 128-d feature vector."""
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        return self.fc2(self.features(x))


def _train_feature_net(device: str = "cpu", epochs: int = 3) -> _MNISTFeatureNet:
    """Train the feature extractor on MNIST (quick, ~3 epochs)."""
    transform = transforms.Compose([transforms.ToTensor()])
    data = datasets.MNIST("data/", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=512, shuffle=True, num_workers=2)

    net = _MNISTFeatureNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    for _ in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            F.cross_entropy(net(imgs), labels).backward()
            opt.step()
    return net


_CACHED_NET: _MNISTFeatureNet | None = None


def _get_feature_net(device: str = "cpu") -> _MNISTFeatureNet:
    global _CACHED_NET
    if _CACHED_NET is None:
        _CACHED_NET = _train_feature_net(device)
    _CACHED_NET.to(device).eval()
    return _CACHED_NET


# ------------------------------------------------------------------
# FID computation
# ------------------------------------------------------------------

def _get_features(images: torch.Tensor, net: _MNISTFeatureNet,
                  device: str, batch_size: int = 256) -> np.ndarray:
    """Extract features for a batch of (N, 1, 28, 28) images in [0,1]."""
    feats = []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            if batch.dim() == 3:
                batch = batch.unsqueeze(1)
            feats.append(net.features(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = "cpu",
) -> float:
    """
    Compute FID between *real_images* and *generated_images*.

    Both tensors should be of shape ``(N, 1, 28, 28)`` with values in [0, 1].

    Returns
    -------
    fid : float
    """
    net = _get_feature_net(device)

    feats_real = _get_features(real_images, net, device)
    feats_gen = _get_features(generated_images, net, device)

    mu_r, sigma_r = feats_real.mean(0), np.cov(feats_real, rowvar=False)
    mu_g, sigma_g = feats_gen.mean(0), np.cov(feats_gen, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)
