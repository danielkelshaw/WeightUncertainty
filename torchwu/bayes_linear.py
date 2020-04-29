from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base_bayesian import BayesianModule
from .samplers.gaussian_variational import GaussianVariational
from .samplers.scale_mixture import ScaleMixture


class BayesLinear(BayesianModule):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 prior_pi: Optional[float] = 0.5,
                 prior_sigma1: Optional[float] = 1.0,
                 prior_sigma2: Optional[float] = 0.005) -> None:

        super().__init__()

        mu = torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        rho = torch.empty(out_features, in_features).uniform_(-5.0, -4.0)

        self.prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)
        self.posterior = GaussianVariational(mu, rho)

        self.log_prior = 0.0
        self.log_posterior = 0.0

    def forward(self, x: Tensor) -> Tensor:

        w = self.posterior.sample()

        self.log_prior = self.prior.log_prior(w)
        self.log_posterior = self.posterior.log_posterior()

        return F.linear(x, w)
