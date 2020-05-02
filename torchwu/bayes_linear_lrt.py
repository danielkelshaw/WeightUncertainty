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

        w_mu = torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        self.w_mu = nn.Parameter(w_mu)

        w_rho = torch.empty(out_features, in_features).uniform_(-5.0, -4.0)
        self.w_rho = nn.Parameter(w_rho)

        bias_mu = torch.empty(out_features).uniform_(-0.2, 0.2)
        self.bias_mu = nn.Parameter(bias_mu)

        bias_rho = torch.empty(out_features).uniform_(-5.0, -4.0)
        self.bias_rho = nn.Parameter(bias_rho)

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

        w_std = torch.log(1 + torch.exp(self.w_rho))
        b_std = torch.log(1 + torch.exp(self.bias_rho))

        act_mu = F.linear(x, self.w_mu)
        act_std = torch.sqrt(F.linear(x.pow(2), w_std.pow(2)))

        w_eps = self.epsilon_normal.sample(act_mu.size())
        bias_eps = self.epsilon_normal.sample(b_std.size())

        w_out = act_mu + act_std * w_eps
        b_out = self.bias_mu + b_std * bias_eps

        output = w_out + b_out.unsqueeze(0).expand(x.shape[0], -1)

        w_kl = self.kld(
            mu_prior=0.0,
            std_prior=self.std_prior,
            mu_posterior=self.mu,
            std_posterior=w_std
        )

        bias_kl = self.kld(
            mu_prior=0.0,
            std_prior=0.1,
            mu_posterior=self.bias_mu,
            std_posterior=b_std
        )

        self.kl_divergence = w_kl + bias_kl

        return output

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
