import pytest
import torch

from torchwu.samplers.gaussian_variational import GaussianVariational


class TestGaussianVariational:

    @pytest.fixture
    def gaussian_variational(self):

        mu = torch.empty(10, 10).uniform_(-1.0, 1.0)
        rho = torch.empty(10, 10).uniform_(-1.0, 1.0)

        return GaussianVariational(mu, rho)

    def test_sample(self, gaussian_variational):

        w_sample = gaussian_variational.sample()
        assert isinstance(w_sample, torch.Tensor)

    def test_log_posterior(self, gaussian_variational):

        w_sample = gaussian_variational.sample()
        log_posterior = gaussian_variational.log_posterior()

        assert isinstance(log_posterior, torch.Tensor)
