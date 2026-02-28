import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MoGPrior(nn.Module):
    def __init__(self, M, n_components=10, init_std=1.0):
        """
        Define a Mixture of Gaussians prior over the latent space.

        Parameters:
        M: [int]
           Dimension of the latent space.
        n_components: [int]
           Number of Gaussian components in the mixture.
        init_std: [float]
           Initial standard deviation for each component.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.n_components = n_components

        self.logits = nn.Parameter(torch.zeros(self.n_components))
        self.means = nn.Parameter(torch.randn(self.n_components, self.M) * 0.05)
        self.log_stds = nn.Parameter(
            torch.full((self.n_components, self.M), torch.log(torch.tensor(init_std)))
        )

    def forward(self):
        """
        Return the mixture prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        mix = td.Categorical(logits=self.logits)
        std = F.softplus(self.log_stds) + 1e-5
        comp = td.Independent(td.Normal(loc=self.means, scale=std), 1)
        return td.MixtureSameFamily(mix, comp)
