# Advanced Machine Learning — DTU 02460

Generative modelling on MNIST: VAEs, DDPM, and a Latent DDPM that runs diffusion in the latent space of a β-VAE with Gaussian likelihood p(x | z).

## Group Members

| Student ID | GitHub |
|------------|--------|
| s253711 | InfiniteLobster |
| s253820 | Berk Yozkan |

---

- Training all generative models (VAE variants, DDPM U-Net, Latent DDPM)
- 4 representative samples from each model
- FID evaluation for DDPM, Latent DDPM, and VAE
- FID vs β sweep for the Latent DDPM (including β = 1e-6)
- Wall-clock sampling speed (samples / second) for each model
- Latent space analysis — KDE plots of VAE prior, aggregate posterior, and learned DDPM prior

---

## Project Structure

```
Advanced-Machine-learning/
│
├── README.md
├── main.py                         # Minimal script entry point
├── pyproject.toml                  # Project dependencies (uv / pip)
│
├── DDPM/                           # Original reference DDPM code
│   ├── ddpm.py                     # Core DDPM class & training loop
│   ├── unet.py                     # U-Net noise-prediction network
│   └── mnist_ddpm_colab.ipynb      # Original Colab training notebook
│
├── LatentDDPMwNB/                  # Main project package
│   ├── experiments.ipynb           # PRIMARY NOTEBOOK — all experiments
│   ├── compare.py                  # Sampling, FID, speed, latent plots
│   ├── data.py                     # MNIST data loaders
│   ├── fid.py                      # FID computation & MNIST classifier
│   ├── mnist_classifier.pth        # Pre-trained classifier for FID
│   │
│   ├── models/
│   │   ├── base.py                 # Abstract GenerativeModel interface
│   │   ├── registry.py             # Model registry
│   │   ├── vae.py                  # VAE (Gaussian / MoG / Flow prior, Gaussian / Bernoulli decoder)
│   │   ├── ddpm.py                 # DDPM core + FcNetwork (ported from DDPM/)
│   │   ├── ddpm_unet.py            # Pixel-space DDPM with U-Net backbone
│   │   └── latent_ddpm.py          # Latent DDPM: β-VAE encoder → DDPM in z-space → decoder
│   │
│   ├── data/                       # Cached MNIST dataset
│   └── project/outputs/            # Saved model checkpoints & figures
│       ├── ddpm_unet.pt
│       ├── latent_ddpm.pt
│       ├── vae_gauss.pt
│       ├── vae_flow.pt
│       ├── vae_mog.pt
│       └── vae_bernoulli.pt
│
└── src/
    └── Part A/                     # Part A code (VAE & flows baseline)
        └── Szymon/                 # Flow, VAE, priors reference implementations
```

---

## Models

| Model | Description |
|-------|-------------|
| **VAE (Gaussian Prior)** | Standard VAE with N(0, I) prior and Gaussian decoder |
| **VAE (MoG Prior)** | VAE with Mixture-of-Gaussians prior |
| **VAE (Flow Prior)** | VAE with normalizing-flow prior |
| **VAE (Bernoulli Decoder)** | VAE with flow prior and Bernoulli decoder for binarised MNIST |
| **DDPM (U-Net)** | Pixel-space DDPM with U-Net backbone, T=1000 steps |
| **Latent DDPM** | DDPM in the latent space of a β-VAE with Gaussian likelihood |

---

## Setup

```bash
# Install dependencies
pip install -e .

# Or with uv
uv sync
```

Open [`LatentDDPMwNB/experiments.ipynb`](LatentDDPMwNB/experiments.ipynb) and run all cells.  
Set `TEST_MODE = True` in the hyperparameters cell for a fast sanity-check run.
