"""
VAE module — all VAE-related components in one place.

Merges the former ``flow.py``, ``beta_vae.py``, and ``vae_model.py`` into a
single file for simplicity.

Contents
--------
Flow components (ported from ``src/Part A/Szymon/flow.py``):
    GaussianBase, MaskedCouplingLayer, Flow, build_flow_prior

Priors:
    GaussianPrior, MoGPrior

Encoder / Decoders:
    GaussianEncoder, GaussianDecoder, BernoulliDecoder

Core VAE:
    BetaVAE, build_beta_vae, train_beta_vae

Pipeline wrapper:
    VAEModel  (implements GenerativeModel for the comparison pipeline)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm

from .base import GenerativeModel
from .registry import ModelRegistry


# =====================================================================
# 1. Normalizing-flow components
# =====================================================================

class GaussianBase(nn.Module):
    """Standard Gaussian base distribution N(0, I)."""

    def __init__(self, D: int):
        super().__init__()
        self.D = D
        self.register_buffer("mean", torch.zeros(D))
        self.register_buffer("std", torch.ones(D))

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MaskedCouplingLayer(nn.Module):
    """Affine coupling layer with a fixed binary mask."""

    def __init__(self, scale_net: nn.Module, translation_net: nn.Module,
                 mask: torch.Tensor):
        super().__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        """Base -> data direction.  Returns (x, log_det_J)."""
        z_masked = self.mask * z
        s = self.scale_net(z_masked).clamp(-2, 2)
        t = self.translation_net(z_masked)
        x = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
        log_det_J = ((1 - self.mask) * s).sum(dim=-1)
        return x, log_det_J

    def inverse(self, x):
        """Data -> base direction.  Returns (z, log_det_J)."""
        x_masked = self.mask * x
        s = self.scale_net(x_masked).clamp(-2, 2)
        t = self.translation_net(x_masked)
        z = x_masked + (1 - self.mask) * ((x - t) * torch.exp(-s))
        log_det_J = -((1 - self.mask) * s).sum(dim=-1)
        return z, log_det_J


class Flow(nn.Module):
    """
    Normalizing-flow model composed of a base distribution and a
    sequence of invertible transformations.
    """

    def __init__(self, base: GaussianBase, transformations):
        super().__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z=None):
        if z is None:
            return self
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J

    def inverse(self, x):
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J

    def log_prob(self, x):
        """Log-probability under the flow."""
        z, log_det_J = self.inverse(x)
        z = z.to(self.base.mean.device)
        return self.base().log_prob(z) + log_det_J

    def sample(self, sample_shape=(1,)):
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]

    def loss(self, x):
        """Negative mean log-likelihood."""
        return -torch.mean(self.log_prob(x))


def build_flow_prior(latent_dim: int, num_transformations: int = 5,
                     num_hidden: int = 8, device: str = "cpu") -> Flow:
    """Build a normalizing-flow prior for a VAE latent space."""
    base = GaussianBase(latent_dim)
    mask = torch.zeros(latent_dim)
    mask[latent_dim // 2:] = 1

    transformations = []
    for _ in range(num_transformations):
        mask = 1 - mask
        scale_net = nn.Sequential(
            nn.Linear(latent_dim, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, latent_dim),
        )
        translation_net = nn.Sequential(
            nn.Linear(latent_dim, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, latent_dim),
        )
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

    return Flow(base, transformations).to(device)


# =====================================================================
# 2. Priors
# =====================================================================

class GaussianPrior(nn.Module):
    """Standard N(0, I) prior."""

    def __init__(self, M: int):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(M), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(self.mean, self.std), 1)


class MoGPrior(nn.Module):
    """
    Mixture-of-Gaussians prior over the latent space.

    Ported from ``src/Part A/Szymon/priors.py``.
    """

    def __init__(self, M: int, n_components: int = 10, init_std: float = 1.0):
        super().__init__()
        self.M = M
        self.n_components = n_components
        self.logits = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, M) * 0.05)
        self.log_stds = nn.Parameter(
            torch.full((n_components, M), torch.log(torch.tensor(init_std)))
        )

    def forward(self):
        mix = td.Categorical(logits=self.logits)
        std = torch.nn.functional.softplus(self.log_stds) + 1e-5
        comp = td.Independent(td.Normal(loc=self.means, scale=std), 1)
        return td.MixtureSameFamily(mix, comp)


# =====================================================================
# 3. Encoder / Decoders
# =====================================================================

class GaussianEncoder(nn.Module):
    """q(z|x): diagonal Gaussian parameterised by an encoder network."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        mean, log_std = torch.chunk(self.net(x), 2, dim=-1)
        return td.Independent(td.Normal(mean, torch.exp(log_std.clamp(-5, 2))), 1)


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
        self.log_var = nn.Parameter(torch.zeros(output_dim), requires_grad=False)

    def forward(self, z):
        mean = self.net(z)
        std = torch.exp(0.5 * self.log_var).expand_as(mean)
        return td.Independent(td.Normal(mean, std), 1)


class BernoulliDecoder(nn.Module):
    """
    p(x|z): Bernoulli decoder for binarised data.

    Ported from ``src/Part A/Szymon/vae.py``.
    The decoder network is expected to output **logits** (before sigmoid).
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, z):
        logits = self.net(z)
        # validate_args=False: MNIST pixels are continuous [0,1], not
        # binary {0,1}, so the standard Bernoulli support check must be
        # skipped.  The BCE loss is still well-defined for continuous targets.
        return td.Independent(td.Bernoulli(logits=logits, validate_args=False), 1)


# =====================================================================
# 4. BetaVAE core
# =====================================================================

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
        if isinstance(self.prior, Flow):
            log_qz = q.log_prob(z)
            log_pz = self.prior.log_prob(z)
            kl = log_qz - log_pz
        elif isinstance(self.prior, MoGPrior):
            log_qz = q.log_prob(z)
            log_pz = self.prior().log_prob(z)
            kl = log_qz - log_pz
        else:
            kl = td.kl_divergence(q, self.prior())
        return (recon - self.beta * kl).mean()

    def forward(self, x):
        """Return negative ELBO (loss)."""
        return -self.elbo(x)

    def sample(self, n_samples: int = 1):
        if isinstance(self.prior, Flow):
            z = self.prior.sample((n_samples,))
        else:
            z = self.prior().sample((n_samples,))
        p_x = self.decoder(z)
        if hasattr(p_x, 'mean'):
            return p_x.mean
        return p_x.base_dist.probs  # Bernoulli probs

    def encode(self, x):
        """Return posterior q(z|x)."""
        return self.encoder(x)

    def decode_mean(self, z):
        """Deterministic decoder mean."""
        return self.decoder(z).mean


# =====================================================================
# 5. Network builders
# =====================================================================

def make_encoder_net(input_dim: int, latent_dim: int, hidden: int = 512) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, latent_dim * 2),
    )


def make_decoder_net(latent_dim: int, output_dim: int, hidden: int = 512,
                     decoder_type: str = "gaussian") -> nn.Module:
    """
    Build the decoder network.

    For ``decoder_type='gaussian'``, the final activation is Sigmoid so that
    the Gaussian mean lies in [0, 1].  For ``decoder_type='bernoulli'``,
    raw logits are returned (the Bernoulli distribution applies sigmoid
    internally).
    """
    layers = [
        nn.Linear(latent_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output_dim),
    ]
    if decoder_type == "gaussian":
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def build_beta_vae(latent_dim: int = 32, beta: float = 1.0,
                   hidden: int = 512, input_dim: int = 784,
                   device: str = "cpu",
                   prior_type: str = "flow",
                   decoder_type: str = "gaussian",
                   num_flow_transformations: int = 5,
                   flow_hidden: int = 8,
                   mog_components: int = 10) -> BetaVAE:
    """
    Convenience constructor for a fully-connected Beta-VAE on MNIST.

    Parameters
    ----------
    prior_type : ``"gaussian"`` | ``"flow"`` | ``"mog"``
    decoder_type : ``"gaussian"`` | ``"bernoulli"``
    """
    if prior_type == "flow":
        prior = build_flow_prior(latent_dim, num_flow_transformations,
                                 flow_hidden, device)
    elif prior_type == "mog":
        prior = MoGPrior(latent_dim, n_components=mog_components)
    else:
        prior = GaussianPrior(latent_dim)
    enc_net = make_encoder_net(input_dim, latent_dim, hidden)
    dec_net = make_decoder_net(latent_dim, input_dim, hidden, decoder_type=decoder_type)
    encoder = GaussianEncoder(enc_net)
    if decoder_type == "bernoulli":
        decoder = BernoulliDecoder(dec_net)
    else:
        decoder = GaussianDecoder(dec_net, output_dim=input_dim)
    model = BetaVAE(prior, encoder, decoder, beta=beta).to(device)
    return model


# =====================================================================
# 6. Training loop
# =====================================================================

def train_beta_vae(model: BetaVAE, train_loader, epochs: int,
                   lr: float = 1e-3, device: str = "cpu",
                   beta_warmup_epochs: int = 0, gradient_clip: float = 1.0):
    """
    Train β-VAE with β-warmup, cosine LR schedule, and gradient clipping.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total = len(train_loader) * epochs
    pbar = tqdm(range(total), desc="β-VAE training")
    initial_beta = model.beta

    for epoch in range(epochs):
        if beta_warmup_epochs > 0:
            warmup_progress = min(epoch / beta_warmup_epochs, 1.0)
            model.beta = initial_beta * warmup_progress

        for x in train_loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            pbar.set_postfix(
                loss=f"{loss.item():12.4f}",
                beta=f"{model.beta:.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                epoch=f"{epoch+1}/{epochs}"
            )
            pbar.update()

        scheduler.step()

    model.beta = initial_beta
    pbar.close()


# =====================================================================
# 7. VAEModel — GenerativeModel wrapper for the comparison pipeline
# =====================================================================

@ModelRegistry.register("vae")
class VAEModel(GenerativeModel):
    """
    Standalone VAE with configurable prior and decoder.

    Parameters
    ----------
    prior_type : ``"flow"`` | ``"gaussian"`` | ``"mog"``
    decoder_type : ``"gaussian"`` | ``"bernoulli"``
    """

    name = "vae"

    def __init__(self, device: str = "cpu", beta: float = 1.0,
                 latent_dim: int = 32, hidden: int = 512,
                 prior_type: str = "flow",
                 decoder_type: str = "gaussian",
                 num_flow_transformations: int = 5, flow_hidden: int = 8,
                 mog_components: int = 10,
                 model_name: str | None = None):
        super().__init__(device)
        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.prior_type = prior_type
        self.decoder_type = decoder_type
        self.num_flow_transformations = num_flow_transformations
        self.flow_hidden = flow_hidden
        self.mog_components = mog_components

        if model_name is not None:
            self.name = model_name

        self.vae: BetaVAE = build_beta_vae(
            latent_dim=latent_dim,
            beta=beta,
            hidden=hidden,
            device=device,
            prior_type=prior_type,
            decoder_type=decoder_type,
            num_flow_transformations=num_flow_transformations,
            flow_hidden=flow_hidden,
            mog_components=mog_components,
        )
        self.model = self.vae

    # ---- GenerativeModel API ----

    def train_model(self, train_loader, epochs: int = 50, lr: float = 1e-3,
                    beta_warmup_epochs: int | None = None, **kwargs):
        if beta_warmup_epochs is None:
            beta_warmup_epochs = epochs // 2
        train_beta_vae(
            self.vae, train_loader, epochs,
            lr=lr, device=str(self.device),
            beta_warmup_epochs=beta_warmup_epochs,
        )
        self._is_trained = True

    def sample(self, n_samples: int) -> torch.Tensor:
        """Return (n_samples, 1, 28, 28) images in [0, 1]."""
        self.vae.eval()
        with torch.no_grad():
            flat = self.vae.sample(n_samples).cpu()
        return flat.clamp(0.0, 1.0).view(n_samples, 1, 28, 28)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "vae_state": self.vae.state_dict(),
            "config": {
                "beta": self.beta,
                "latent_dim": self.latent_dim,
                "hidden": self.hidden,
                "prior_type": self.prior_type,
                "decoder_type": self.decoder_type,
                "num_flow_transformations": self.num_flow_transformations,
                "flow_hidden": self.flow_hidden,
                "mog_components": self.mog_components,
            },
        }, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt["config"]
        if (cfg["latent_dim"] != self.latent_dim
                or cfg["hidden"] != self.hidden
                or cfg.get("num_flow_transformations") != self.num_flow_transformations):
            self.__init__(device=str(self.device), **cfg)  # type: ignore[misc]
        self.vae.load_state_dict(ckpt["vae_state"])
        self._is_trained = True
