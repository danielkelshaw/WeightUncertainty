import functools as ft

import torch
import torch.nn as nn
from torch import Tensor


class ScaleMixture(nn.Module):

    """Scale Mixture Prior.

    Section 3.3 of the 'Weight Uncertainty in Neural Networks' paper
    proposes the use of a Scale Mixture prior for use in variational
    inference - this being a fixed-form prior.

    The authors note that, should the parameters be allowed to adjust
    during training, the prior changes rapidly and attempts to capture
    the empirical distribution of the weights. As a result the prior
    learns to fit poor initial parameters and struggles to improve.
    """

    def __init__(self, pi: float, sigma1: float, sigma2: float) -> None:

        """Scale Mixture Prior.

        The authors of 'Weight Uncertainty in Neural Networks' note:

            sigma1 > sigma2:
                provides a heavier tail in the prior density than is
                seen in a plain Gaussian prior.
            sigma2 << 1.0:
                causes many of the weights to a priori tightly
                concentrate around zero.

        Parameters
        ----------
        pi : float
            Parameter used to scale the two Gaussian distributions.
        sigma1 : float
            Standard deviation of the first normal distribution.
        sigma2 : float
            Standard deviation of the second normal distribution.
        """

        super().__init__()

        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        self.normal1 = torch.distributions.Normal(0, sigma1)
        self.normal2 = torch.distributions.Normal(0, sigma2)

    def log_prior(self, w: Tensor) -> Tensor:

        """Log Likelihood of the weight according to the prior.

        Calculates the log likelihood of the supplied weight given the
        prior distribution - the scale mixture of two Gaussians.

        Parameters
        ----------
        w : Tensor
            Weight to be used to calculate the log likelihood.

        Returns
        -------
        Tensor
            Log likelihood of the weights from the prior distribution.
        """

        likelihood_n1 = torch.exp(self.normal1.log_prob(w))
        likelihood_n2 = torch.exp(self.normal2.log_prob(w))

        p_scalemixture = self.pi * likelihood_n1 + (1 - self.pi) * likelihood_n2
        log_prob = ft.reduce(lambda x, y: x * y, torch.log(p_scalemixture))

        return log_prob
