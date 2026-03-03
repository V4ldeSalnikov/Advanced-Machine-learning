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
from torchvision import datasets, transforms

from .fid import Classifier, compute_fid
from .data import get_real_images_for_fid
from .models.base import GenerativeModel

# Path to the provided classifier checkpoint (required by project spec).
# Resolve relative to _this_ file so it works regardless of cwd.
_CLASSIFIER_CKPT = str(Path(__file__).parent / "mnist_classifier.pth")


def _ensure_classifier_ckpt(device: str = "cpu") -> str:
    """Train and save the MNIST classifier if the checkpoint does not exist."""
    ckpt = Path(_CLASSIFIER_CKPT)
    if ckpt.exists():
        return str(ckpt)

    print("[FID] Training MNIST classifier (one-time) ...")
    transform = transforms.Compose([transforms.ToTensor()])
    data = datasets.MNIST("data/", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=512, shuffle=True)

    clf = Classifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    clf.train()
    for epoch in range(3):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            torch.nn.functional.cross_entropy(clf(imgs), labels).backward()
            opt.step()

    torch.save(clf.state_dict(), str(ckpt))
    print(f"[FID] Classifier saved to {ckpt}")
    return str(ckpt)


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

def _to_minus1_plus1(x: torch.Tensor) -> torch.Tensor:
    """Rescale tensor from [0, 1] to [-1, 1] as required by fid.py."""
    return x * 2.0 - 1.0


def evaluate_fid(model: GenerativeModel, n_gen: int = 10000,
                 device: str = "cpu", n_real: int = 10000) -> float:
    """
    Generate *n_gen* samples and compute FID vs. MNIST test images.

    Images are rescaled from [0, 1] to [-1, 1] to match the range
    expected by the provided ``fid.py`` / ``mnist_classifier.pth``.
    """
    ckpt = _ensure_classifier_ckpt(device)
    real = _to_minus1_plus1(get_real_images_for_fid(n_real))
    gen = _to_minus1_plus1(model.sample(n_gen).cpu())
    return compute_fid(real, gen, device=device, classifier_ckpt=ckpt)


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

    ckpt = _ensure_classifier_ckpt(device)
    real = _to_minus1_plus1(get_real_images_for_fid(n_fid))

    for m in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {m.name}")
        print(f"{'='*60}")

        # Samples
        sp = (save_dir / f"samples_{m.name}.png") if save_dir else None
        show_samples(m, n=4, save_path=sp)

        # FID  – rescale [0,1] → [-1,1] for fid.py
        gen = _to_minus1_plus1(m.sample(n_fid).cpu())
        fid = compute_fid(real, gen, device=device, classifier_ckpt=ckpt)
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

def _kde_contour(ax, pts, color, label, levels=14, fill_alpha=0.18, line_alpha=0.85):
    """
    Draw filled + outline KDE contours for a 2-D point cloud.

    Parameters
    ----------
    pts : np.ndarray, shape (N, 2)
    color : matplotlib color string
    """
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(pts.T, bw_method="scott")

    # Build evaluation grid from the data range with a small margin
    margin = 0.10
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    dx = (x_max - x_min) * margin
    dy = (y_max - y_min) * margin
    xx, yy = np.meshgrid(
        np.linspace(x_min - dx, x_max + dx, 200),
        np.linspace(y_min - dy, y_max + dy, 200),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=levels, colors=[color], alpha=fill_alpha)
    cs = ax.contour(xx, yy, zz, levels=levels, colors=[color], alpha=line_alpha,
                    linewidths=0.8)
    # invisible proxy for the legend
    ax.plot([], [], color=color, linewidth=1.5, label=label)


def plot_latent_distributions(
    latent_ddpm_model,  # LatentDDPMModel
    data_loader,
    dims: tuple = (0, 1),
    n_samples: int = 5000,
    save_path: str | Path | None = None,
):
    """
    Overlay KDE contour plots (filled + outline) in PCA space.

    Two panels:
      Left  — Aggregate posterior q(z) vs VAE prior p(z) = N(0,I)
      Right — Aggregate posterior q(z) vs Learned DDPM prior

    PCA is fit on the aggregate posterior so the axes reflect the
    directions of maximum variance in the encoder output.  Axis labels
    include the fraction of variance explained by each PC.
    """
    from sklearn.decomposition import PCA

    # ---- collect samples ---------------------------------------------------
    z_agg  = latent_ddpm_model.get_aggregate_posterior_samples(data_loader, n_samples).numpy()
    z_prior = latent_ddpm_model.get_vae_prior_samples(n_samples).numpy()
    z_ddpm  = latent_ddpm_model.get_ddpm_prior_samples(n_samples).numpy()

    # ---- fit PCA on aggregate posterior ------------------------------------
    pca = PCA(n_components=2)
    pca.fit(z_agg)
    var = pca.explained_variance_ratio_ * 100  # percent

    agg_2d   = pca.transform(z_agg)
    prior_2d = pca.transform(z_prior)
    ddpm_2d  = pca.transform(z_ddpm)

    xlabel = f"PC1 ({var[0]:.1f}%)"
    ylabel = f"PC2 ({var[1]:.1f}%)"

    # ---- styling -----------------------------------------------------------
    BLUE = "#3A6BC9"   # aggregate posterior
    RED  = "#D95F4B"   # VAE prior / DDPM prior

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")

    panel_cfg = [
        (axes[0], prior_2d, RED,  "Prior p(z)",
         agg_2d,  BLUE, "Aggregate Posterior q(z)",
         "VAE Prior  vs  Aggregate Posterior"),
        (axes[1], ddpm_2d, RED,  "DDPM Prior",
         agg_2d,  BLUE, "Aggregate Posterior q(z)",
         "Learned DDPM Prior  vs  Aggregate Posterior"),
    ]

    for ax, pts_a, col_a, lbl_a, pts_b, col_b, lbl_b, title in panel_cfg:
        ax.set_facecolor("#F2F2F2")
        ax.grid(True, color="white", linewidth=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        _kde_contour(ax, pts_a, col_a, lbl_a)
        _kde_contour(ax, pts_b, col_b, lbl_b)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.85, fontsize=9)

    plt.tight_layout(pad=2.0)
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
