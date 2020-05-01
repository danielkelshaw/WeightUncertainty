from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_bayesian import BayesianModule


class BayesLinearLRT(BayesianModule):

    """Bayesian Linear Layer with Local Reparameterisation Trick.

    Implementation of a Bayesian Linear Layer utilising the 'local
    reparameterisation trick' in order to sample directly from the
    activations.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 std_prior: Optional[float] = 1.0) -> None:

        """Bayesian Linear Layer with Local Reparameterisation Trick.

        Parameters
        ----------
        in_features : int
            Number of features to feed into the layer.
        out_features : int
            Number of features produced by the layer.
        std_prior : float
            Sigma to be used for the normal distribution in the prior.
        """

        super().__init__()

        self.in_feature = in_features
        self.out_feature = out_features
        self.std_prior = std_prior

        self.mu = nn.Pararmeter(
            torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        )

        self.rho = nn.Pararmeter(
            torch.empty(out_features, in_features).uniform_(-5.0, -4.0)
        )

        self.epsilon_normal = torch.distributions.Normal(0, 1)

        self.kl_divergence = 0.0

    def forward(self, x: Tensor) -> Tensor:

        """Calculates the forward pass through the linear layer.

        The local reparameterisation trick is used to estimate the
        gradients with respect to the parameters of a distribution - it
        takes advantage of the fact that, for a fixed input and Gaussian
        distributions over the weights, the resulting distribution over
        the activations is also Gaussian.

        Instead of sampling the weights individually and using them to
        compute a sample from the activation - we can sample from the
        distribution over activations. This yields a lower variance
        gradient estimator which makes training faster and more stable.

        Parameters
        ----------
        x : Tensor
            Inputs to the Bayesian Linear Layer.

        Returns
        -------
        Tensor
            Output from the Bayesian Linear Layer.
        """

        std = torch.log(1 + torch.exp(self.rho))

        act_mu = F.linear(x, self.mu)
        act_std = torch.sqrt(F.linear(x.pow(2), std.pow(2)))

        eps = self.epsilon_normal.sample(act_mu.size())

        w = act_mu + act_std * eps

        self.kl_divergence = self.kld(
            mu_prior=0.0,
            std_prior=self.std_prior,
            mu_posterior=self.mu,
            std_posterior=std
        )

        return w

    def kld(self,
            mu_prior: float,
            std_prior: float,
            mu_posterior: Tensor,
            std_posterior: Tensor) -> Tensor:

        """Calculates the KL Divergence.

        The only 'downside' to the local reparameterisation trick is
        that, as the weights are not being sampled directly, the KL
        Divergence can not be calculated through the use of MC sampling.
        Instead, the closed form of the KL Divergence must be used;
        this restricts the prior and posterior to be Gaussian.

        However, the use of a Gaussian prior / posterior results in a
        lower variance and hence faster convergence.

        Parameters
        ----------
        mu_prior : float
            Mu of the prior normal distribution.
        std_prior : float
            Sigma of the prior normal distribution.
        mu_posterior : Tensor
            Mu to approximate the posterior normal distribution.
        std_posterior : Tensor
            Sigma to approximate the posterior normal distribution.

        Returns
        -------
        Tensor
            Calculated KL Divergence.
        """

        kl_divergence = 0.5 * (
                2 * torch.log(std_prior / std_posterior) -
                1 +
                (std_posterior / std_prior).pow(2) +
                ((mu_prior - mu_posterior) / std_prior).pow(2)
        ).sum()

        return kl_divergence
