import pytest
import torch

from torchwu.base_bayesian import BayesianModule
from torchwu.bayes_linear import BayesLinear


class TestBayesLinear:

    @pytest.fixture
    def bayes_linear(self):
        return BayesLinear(5, 3)

    def test_forward(self, bayes_linear):

        to_feed = torch.ones(5)
        ret_tensor = bayes_linear.forward(to_feed)

        assert len(ret_tensor) == 3
        assert isinstance(ret_tensor, torch.Tensor)
        assert isinstance(bayes_linear, BayesianModule)

    def test_kld(self, bayes_linear):

        ll_posterior = torch.tensor([[3.0, 3.0, 3.0]])
        ll_prior = torch.tensor([[2.0, 2.0, 2.0]])

        ret_kl = bayes_linear.kld(ll_prior, ll_posterior)

        assert isinstance(ret_kl, torch.Tensor)
        assert torch.eq(ret_kl, torch.tensor([[1.0, 1.0, 1.0]])).all()
