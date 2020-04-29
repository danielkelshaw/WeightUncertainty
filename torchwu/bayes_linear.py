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

        mu = torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        rho = torch.empty(out_features, in_features).uniform_(-5.0, -4.0)

        self.prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)
        self.posterior = GaussianVariational(mu, rho)

        self.log_prior = 0.0
        self.log_posterior = 0.0

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

        w = self.posterior.sample()

        self.log_prior = self.prior.log_prior(w)
        self.log_posterior = self.posterior.log_posterior()

        return F.linear(x, w)