"""
Comparison utilities: sampling, timing, FID, and plotting for all registered
generative models.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from .fid import compute_fid
from .data import get_real_images_for_fid
from .models.base import GenerativeModel


# ------------------------------------------------------------------
# 1. Sample & show representative images
# ------------------------------------------------------------------

def show_samples(model: GenerativeModel, n: int = 4, title: str | None = None,
                 save_path: str | Path | None = None):
    """
    Display *n* samples from *model* in a single row.
    """
    samples = model.sample(n).cpu()  # (n, 1, 28, 28)
    grid = make_grid(samples, nrow=n, padding=2, normalize=False)
    fig, ax = plt.subplots(figsize=(n * 2, 2.5))
    ax.imshow(grid.permute(1, 2, 0).squeeze(-1), cmap="gray")
    ax.set_title(title or model.name)
    ax.axis("off")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    return samples


# ------------------------------------------------------------------
# 2. Timed sampling
# ------------------------------------------------------------------

def measure_sampling_speed(model: GenerativeModel, n_samples: int = 100,
                           warmup: int = 1) -> dict:
    """
    Return ``{samples_per_sec, total_time, n_samples}`` for *model*.
    """
    # warm-up
    for _ in range(warmup):
        _ = model.sample(2)

    samples, elapsed = model.timed_sample(n_samples)
    return {
        "model": model.name,
        "n_samples": n_samples,
        "total_time_s": elapsed,
        "samples_per_sec": n_samples / elapsed,
    }


# ------------------------------------------------------------------
# 3. FID evaluation
# ------------------------------------------------------------------

def evaluate_fid(model: GenerativeModel, n_gen: int = 10000,
                 device: str = "cpu", n_real: int = 10000) -> float:
    """
    Generate *n_gen* samples and compute FID vs. MNIST test images.
    """
    real = get_real_images_for_fid(n_real)
    gen = model.sample(n_gen).cpu()
    return compute_fid(real, gen, device=device)


# ------------------------------------------------------------------
# 4. Full comparison table
# ------------------------------------------------------------------

def compare_models(
    models: List[GenerativeModel],
    n_fid: int = 10000,
    n_speed: int = 100,
    device: str = "cpu",
    save_dir: str | Path | None = None,
) -> Dict[str, dict]:
    """
    Run FID, timing, and sample-visualisation for every model in *models*.
    Returns a dict mapping model name to its results.
    """
    results: Dict[str, dict] = {}
    save_dir = Path(save_dir) if save_dir else None

    real = get_real_images_for_fid(n_fid)

    for m in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {m.name}")
        print(f"{'='*60}")

        # Samples
        sp = (save_dir / f"samples_{m.name}.png") if save_dir else None
        show_samples(m, n=4, save_path=sp)

        # FID
        gen = m.sample(n_fid).cpu()
        fid = compute_fid(real, gen, device=device)
        print(f"  FID = {fid:.2f}")

        # Speed
        speed = measure_sampling_speed(m, n_speed)
        print(f"  {speed['samples_per_sec']:.1f} samples/s  "
              f"({speed['total_time_s']:.2f}s for {n_speed} samples)")

        results[m.name] = {"fid": fid, "speed": speed}

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'FID':>10} {'Samples/s':>12}")
    print(f"{'-'*42}")
    for name, r in results.items():
        print(f"{name:<20} {r['fid']:>10.2f} {r['speed']['samples_per_sec']:>12.1f}")
    print(f"{'='*60}")

    return results


# ------------------------------------------------------------------
# 5. Latent-space visualisation
# ------------------------------------------------------------------

def plot_latent_distributions(
    latent_ddpm_model,  # LatentDDPMModel
    data_loader,
    dims: tuple = (0, 1),
    n_samples: int = 5000,
    save_path: str | Path | None = None,
):
    """
    Plot three distributions in the first two latent dimensions:
    1. VAE prior  p(z) = N(0, I)
    2. Aggregate posterior  q(z) = 1/N Σ q(z|x_i)
    3. Learned DDPM prior (samples from the latent DDPM)

    This is the plot required by the project description for comparing
    the β-VAE prior and the learned latent DDPM distribution against
    the aggregate posterior.
    """
    d0, d1 = dims

    z_prior = latent_ddpm_model.get_vae_prior_samples(n_samples)
    z_agg = latent_ddpm_model.get_aggregate_posterior_samples(data_loader, n_samples)
    z_ddpm = latent_ddpm_model.get_ddpm_prior_samples(n_samples)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ["VAE prior N(0,I)", "Aggregate posterior", "Learned DDPM prior"]
    data = [z_prior, z_agg, z_ddpm]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for ax, z, title, c in zip(axes, data, titles, colors):
        ax.scatter(z[:, d0].numpy(), z[:, d1].numpy(), s=1, alpha=0.3, c=c)
        ax.set_title(title)
        ax.set_xlabel(f"z[{d0}]")
        ax.set_ylabel(f"z[{d1}]")
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    return fig


def fid_vs_beta(
    betas: List[float],
    train_loader,
    device: str = "cpu",
    latent_dim: int = 32,
    vae_epochs: int = 50,
    ddpm_epochs: int = 50,
    n_fid: int = 10000,
    save_path: str | Path | None = None,
) -> Dict[float, float]:
    """
    Train a latent DDPM for each β value and report FID.

    Returns a dict ``{beta: fid_score}``.
    """
    from .models.latent_ddpm import LatentDDPMModel  # local import to avoid circular

    results: Dict[float, float] = {}
    for beta in betas:
        print(f"\n--- β = {beta} ---")
        m = LatentDDPMModel(device=device, beta=beta, latent_dim=latent_dim)
        m.train_model(train_loader, vae_epochs=vae_epochs, ddpm_epochs=ddpm_epochs)
        fid = evaluate_fid(m, n_gen=n_fid, device=device)
        results[beta] = fid
        print(f"  β={beta:.1e}  →  FID={fid:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = list(results.keys())
    ys = list(results.values())
    ax.plot(xs, ys, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("β")
    ax.set_ylabel("FID")
    ax.set_title("FID vs β for Latent DDPM")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

    return results
