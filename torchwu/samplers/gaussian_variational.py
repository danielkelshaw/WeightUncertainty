import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class GaussianVariational(nn.Module):

    """Gaussian Variational Weight Sampler.

    Section 3.2 of the 'Weight Uncertainty in Neural Networks' paper
    proposes the use of a Gaussian posterior in order to sample weights
    from the network for use in variational inference.
    """

    def __init__(self, mu: Tensor, rho: Tensor) -> None:

        """Gaussian Variational Weight Sampler.

        Parameters
        ----------
        mu : Tensor
            Mu used to shift the samples drawn from a unit Gaussian.
        rho : Tensor
            Rho used to generate the pointwise parameterisation of the
            standard deviation - used to scale the samples drawn a unit
            Gaussian.
        """

        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)

        self.w = None
        self.sigma = None

        self.normal = torch.distributions.Normal(0, 1)

    def sample(self) -> Tensor:

        """Draws a sample from the posterior distribution.

        Samples a weight using:
            w = mu + log(1 + exp(rho)) * epsilon
                where epsilon ~ N(0, 1)

        Returns
        -------
        Tensor
            Sampled weight from the posterior distribution.
        """

        device = self.mu.device
        epsilon = self.normal.sample(self.mu.size()).to(device)
        self.sigma = torch.log(1 + torch.exp(self.rho)).to(device)
        self.w = self.mu + self.sigma * epsilon

        return self.w

    def log_posterior(self) -> Tensor:

        """Log Likelihood for each weight sampled from the distribution.

        Calculates the Gaussian log likelihood of the sampled weight
        given the the current mean, mu, and standard deviation, sigma:

            LL = -log((2pi * sigma^2)^0.5) - 0.5(w - mu)^2 / sigma^2

        Returns
        -------
        Tensor
            Gaussian log likelihood for the weights sampled.
        """

        if self.w is None:
            raise ValueError('self.w must have a value.')

        log_const = np.log(np.sqrt(2 * np.pi))
        log_exp = ((self.w - self.mu) ** 2) / (2 * self.sigma ** 2)
        log_posterior = -log_const - torch.log(self.sigma) - log_exp

        return log_posterior.mean()
