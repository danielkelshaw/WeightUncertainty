from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base_bayesian import BayesianModule
from .samplers.gaussian_variational import GaussianVariational
from .samplers.scale_mixture import ScaleMixture


class BayesLinear(BayesianModule):

    """Bayesian Linear Layer.

    Implementation of a Bayesian Linear Layer as described in the
    'Weight Uncertainty in Neural Networks' paper.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 prior_pi: Optional[float] = 0.5,
                 prior_sigma1: Optional[float] = 1.0,
                 prior_sigma2: Optional[float] = 0.005) -> None:

        """Bayesian Linear Layer.

        Parameters
        ----------
        in_features : int
            Number of features to feed in to the layer.
        out_features : out
            Number of features produced by the layer.
        prior_pi : float
            Pi weight to be used for the ScaleMixture prior.
        prior_sigma1 : float
            Sigma for the first normal distribution in the prior.
        prior_sigma2 : float
            Sigma for the second normal distribution in the prior.
        """

        super().__init__()

        w_mu = torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        w_rho = torch.empty(out_features, in_features).uniform_(-5.0, -4.0)

        bias_mu = torch.empty(out_features).uniform_(-0.2, 0.2)
        bias_rho = torch.empty(out_features).uniform_(-5.0, -4.0)

        self.w_posterior = GaussianVariational(w_mu, w_rho)
        self.bias_posterior = GaussianVariational(bias_mu, bias_rho)

        self.w_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)
        self.bias_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)

        self.kl_divergence = 0.0

    def forward(self, x: Tensor) -> Tensor:

        """Calculates the forward pass through the linear layer.

        Parameters
        ----------
        x : Tensor
            Inputs to the Bayesian Linear layer.

        Returns
        -------
        Tensor
            Output from the Bayesian Linear layer.
        """

        w = self.w_posterior.sample()
        b = self.bias_posterior.sample()

        w_log_prior = self.w_prior.log_prior(w)
        b_log_prior = self.bias_prior.log_prior(b)

        w_log_posterior = self.w_posterior.log_posterior()
        b_log_posterior = self.bias_posterior.log_posterior()

        total_log_prior = w_log_prior + b_log_prior
        total_log_posterior = w_log_posterior + b_log_posterior
        self.kl_divergence = self.kld(total_log_prior, total_log_posterior)

        return F.linear(x, w)

    def kld(self, log_prior: Tensor, log_posterior: Tensor) -> Tensor:

        """Calculates the KL Divergence.

        Uses the weight sampled from the posterior distribution to
        calculate the KL Divergence between the prior and posterior.

        Parameters
        ----------
        log_prior : Tensor
            Log likelihood drawn from the prior.
        log_posterior : Tensor
            Log likelihood drawn from the approximate posterior.

        Returns
        -------
        Tensor
            Calculated KL Divergence.
        """

        return log_posterior - log_prior
