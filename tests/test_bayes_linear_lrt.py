import pytest
import torch

from torchwu.base_bayesian import BayesianModule
from torchwu.bayes_linear_lrt import BayesLinearLRT


class TestBayesLinear:

    @pytest.fixture
    def bayes_linear_lrt(self):
        return BayesLinearLRT(5, 3)

    def test_forward(self, bayes_linear_lrt):

        to_feed = torch.ones(5)
        ret_tensor = bayes_linear_lrt.forward(to_feed)

        assert len(ret_tensor) == 3
        assert isinstance(ret_tensor, torch.Tensor)
        assert isinstance(bayes_linear_lrt, BayesianModule)

    def test_kld(self, bayes_linear_lrt):

        std_prior = 0.5
        mu_prior = 0.5

        std_posterior = torch.tensor([[0.5, 0.5, 0.5]])
        mu_posterior = torch.tensor([[0.7, 0.75, 0.75]])

        ret_kl = bayes_linear_lrt.kld(
            mu_prior, std_prior, mu_posterior, std_posterior
        )

        assert isinstance(ret_kl, torch.Tensor)
        assert ret_kl.item() == pytest.approx(0.33, 1e-3)
