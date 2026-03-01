"""
MNIST data-loading helpers shared across all models.

All loaders return images as flat tensors in [0, 1] by default (standard MNIST,
**not** binarised), which is what the Gaussian-likelihood β-VAE expects.

For the DDPM the images are shifted to [-1, 1]; a separate loader is provided.
"""

from __future__ import annotations

import torch
import torch.utils.data
from torchvision import datasets, transforms


def get_standard_mnist(batch_size: int = 64, root: str = "data/", flatten: bool = True):
    """
    Standard MNIST in [0, 1] with optional dequantisation noise.

    Returns (train_loader, test_loader).
    """
    tfm = [transforms.ToTensor()]
    if flatten:
        tfm.append(transforms.Lambda(lambda x: x.flatten()))
    transform = transforms.Compose(tfm)

    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_ddpm_mnist(batch_size: int = 64, root: str = "data/"):
    """
    MNIST rescaled to [-1, 1] with dequantisation noise (as the week-3 code
    uses).  Returns flat tensors of shape (784,).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255.0),
        transforms.Lambda(lambda x: (x - 0.5) * 2.0),
        transforms.Lambda(lambda x: x.flatten()),
    ])
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_real_images_for_fid(n: int = 10000, root: str = "data/") -> torch.Tensor:
    """
    Return *n* real MNIST test images as a (n, 1, 28, 28) tensor in [0, 1].
    Useful as the reference set for FID computation.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    data = datasets.MNIST(root, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=n, shuffle=False)
    imgs, _ = next(iter(loader))
    return imgs[:n]
